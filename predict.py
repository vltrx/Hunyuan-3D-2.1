import os
import shutil
import subprocess
import sys
import time
import json
import traceback
import io
import requests
import zipfile
from typing import List, Optional, Union
import gc
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from threading import Semaphore
import tempfile
import queue
from tqdm import tqdm

import torch
from PIL import Image
from torch import cuda, Generator
from cog import BasePredictor, BaseModel, Input, Path  # Path here is for model outputs
import pathlib

# HuggingFace-style environment setup (from their gradio_app.py)
def setup_environment():
    """Setup environment variables for optimal CUDA performance"""
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["PATH"] = f"{os.environ.get('CUDA_HOME', '/usr/local/cuda')}/bin:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('CUDA_HOME', '/usr/local/cuda')}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    
    # Critical environment variables for production stability
    os.environ["OMP_NUM_THREADS"] = "1"
    # U2NET_HOME will be set later when U2NET_PATH is defined
    
    # Ensure CUDA toolkit is available
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
    print(f"PATH: {os.environ.get('PATH')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    print(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST')}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"U2NET_HOME: {os.environ.get('U2NET_HOME')}")

# Setup environment before importing anything else
setup_environment()

# Add paths for the model modules
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Apply torchvision fix early
try:
    from torchvision_fix import apply_fix
    apply_fix()
    print("Applied torchvision compatibility fix")
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.postprocessors import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline, export_to_trimesh
from hy3dshape.models.autoencoders import SurfaceExtractors
from hy3dshape.utils import logger

# Use HF-style import pattern for texture generation
try:
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    print("Using HF-style textureGenPipeline import")
except ImportError:
    # Fallback to full path if needed
    from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    print("Using fallback hy3dpaint.textureGenPipeline import")

# Global variables for lazy loading (HF-style but conditional)
rmbg_worker = None
i23d_worker = None
tex_pipeline = None
floater_remove_worker = None
degenerate_face_remove_worker = None
face_reduce_worker = None
mesh_simplifier = None

# Thread-safe storage for per-worker model instances
worker_models = {}
worker_models_lock = threading.Lock()

# Post-processing serialization lock (only for mesh processing)
mesh_processing_lock = threading.Lock()

# Volume decoding serialization lock - prevents memory collision during parallel processing
volume_decoding_lock = threading.Lock()

# Texture generation lock - prevents VRAM overload during parallel texture generation
texture_generation_lock = threading.Lock()

# Global gate to limit heavy shape generation to 1 GPU at a time
shape_gpu_gate = Semaphore(1)

def ensure_directory_exists(path):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _comprehensive_texture_pipeline_reset(tex_pipeline):
    """Comprehensive reset of all texture generation pipeline state to prevent contamination."""
    try:
        logger.info("  🧹 Performing SELECTIVE texture pipeline reset (preserving current image data)...")
        
        # 1. Reset MeshRender state (most critical for geometry contamination)
        if hasattr(tex_pipeline, 'render') and tex_pipeline.render is not None:
            render = tex_pipeline.render
            
            # DON'T clear mesh geometry data - that's the INPUT for the current image!
            # The mesh data (vtx_pos, pos_idx, vtx_uv, uv_idx) is specific to current image
            # Only clear internal caches that could cause cross-image contamination
            
            # Clear internal texture processing caches
            if hasattr(render, 'tex_position'):
                render.tex_position = None
            if hasattr(render, 'tex_normal'):
                render.tex_normal = None
            if hasattr(render, 'tex_grid'):
                render.tex_grid = None
            if hasattr(render, 'texture_indices'):
                render.texture_indices = None
                
            # DON'T clear actual texture data - that's the OUTPUT, not contamination!
            # The texture storage (tex, tex_mr, tex_normalMap) should remain intact
            # Only clear internal state that could cause cross-contamination
                
            # DON'T clear mesh-specific state (scale_factor, mesh_normalize_*) - that's for current image!
            # These are computed for the specific mesh being processed, not contamination state
                    
        # 2. Reset multiview model pipeline state
        if hasattr(tex_pipeline, 'models') and 'multiview_model' in tex_pipeline.models:
            multiview_model = tex_pipeline.models['multiview_model']
            if hasattr(multiview_model, 'pipeline'):
                pipeline = multiview_model.pipeline
                
                # Clear UNet cached conditions and embeddings
                if hasattr(pipeline, 'unet'):
                    # Force clear any cached states in UNet
                    if hasattr(pipeline.unet, '_cached_condition'):
                        pipeline.unet._cached_condition = None
                    if hasattr(pipeline.unet, 'conv_in_cache'):
                        pipeline.unet.conv_in_cache = None
                        
                # Reset scheduler state to prevent lazy wrapper issues
                if hasattr(pipeline, 'scheduler'):
                    scheduler = pipeline.scheduler
                    # Clear any step-dependent state
                    if hasattr(scheduler, 'timesteps'):
                        # Don't clear timesteps but clear any cached computations
                        pass
                    if hasattr(scheduler, '_step_index'):
                        scheduler._step_index = None
                    if hasattr(scheduler, 'model_outputs'):
                        scheduler.model_outputs = None
                        
        # 3. Clear CUDA cache to prevent memory state contamination
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("  ✅ SELECTIVE texture pipeline reset completed (current image data preserved)")
        
    except Exception as e:
        logger.warning(f"  ⚠️ Error during texture pipeline reset: {e}")

def _aggressive_gpu_state_reset(tex_pipeline):
    """Aggressive GPU state reset specifically for warmed-up GPU contamination issues."""
    try:
        logger.info("  🔥 Performing AGGRESSIVE GPU state reset for warmed-up GPU...")
        
        # 1. First do standard reset
        _comprehensive_texture_pipeline_reset(tex_pipeline)
        
        # 2. AGGRESSIVE GPU memory management
        import torch
        if torch.cuda.is_available():
            # Force complete GPU memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Clean up inter-process communication
            
            # Force garbage collection to free Python references
            import gc
            gc.collect()
            
            # Additional CUDA cache clearing for warmed-up GPUs
            torch.cuda.reset_peak_memory_stats()
            
        # 3. FORCE MODEL RE-INITIALIZATION (most aggressive approach)
        # This targets the "warmed up" GPU kernel caching issue
        if hasattr(tex_pipeline, 'models'):
            models = tex_pipeline.models
            
            # Re-initialize multiview model pipeline components that cache GPU state
            if 'multiview_model' in models and hasattr(models['multiview_model'], 'pipeline'):
                pipeline = models['multiview_model'].pipeline
                
                # ULTRA-AGGRESSIVE scheduler reinitialization for lazy wrapper issues
                if hasattr(pipeline, 'scheduler'):
                    # Backup scheduler config
                    scheduler_class = type(pipeline.scheduler)
                    scheduler_config = pipeline.scheduler.config if hasattr(pipeline.scheduler, 'config') else {}
                    
                    # Force complete scheduler destruction and recreation
                    try:
                        # Delete old scheduler completely
                        old_scheduler = pipeline.scheduler
                        pipeline.scheduler = None
                        del old_scheduler
                        
                        # Force garbage collection
                        import gc
                        gc.collect()
                        
                        # Clear any PyTorch lazy tensor caches
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            # Clear lazy tensor registry if accessible
                            if hasattr(torch._C, '_clear_lazy_graph'):
                                try:
                                    torch._C._clear_lazy_graph()
                                except:
                                    pass
                        
                        # Create completely fresh scheduler instance
                        pipeline.scheduler = scheduler_class.from_config(scheduler_config) if scheduler_config else scheduler_class()
                        
                                                # Ensure scheduler is on the correct device
                        if hasattr(pipeline.scheduler, 'to') and torch.cuda.is_available():
                            pipeline.scheduler = pipeline.scheduler.to(torch.cuda.current_device())
                        
                        # Special handling for UniPC multistep scheduler (source of lazy wrapper error)
                        if 'UniPC' in str(type(pipeline.scheduler)):
                            # Force clear any internal state that could cause lazy wrapper issues
                            if hasattr(pipeline.scheduler, 'model_outputs'):
                                pipeline.scheduler.model_outputs = None
                            if hasattr(pipeline.scheduler, '_step_index'):
                                pipeline.scheduler._step_index = None
                            if hasattr(pipeline.scheduler, 'lower_order_nums'):
                                pipeline.scheduler.lower_order_nums = 0
                            # Clear any matrix caches that could be corrupted
                            for attr in ['_R', '_b', '_rhos_c']:
                                if hasattr(pipeline.scheduler, attr):
                                    setattr(pipeline.scheduler, attr, None)
                            logger.info("  🎯 Applied UniPC-specific lazy wrapper fixes")
                            
                        logger.info("  🔄 ULTRA-AGGRESSIVE scheduler reinitialization completed")
                    except Exception as e:
                        logger.warning(f"  ⚠️ Could not reinitialize scheduler: {e}")
                        # Fallback to basic reset
                        try:
                            pipeline.scheduler = scheduler_class.from_config(scheduler_config) if scheduler_config else scheduler_class()
                        except:
                            pass
                
                # Clear VAE decoder caches that persist on warmed GPUs
                if hasattr(pipeline, 'vae'):
                    # Force VAE to clear its internal caches
                    if hasattr(pipeline.vae, 'decoder') and hasattr(pipeline.vae.decoder, 'conv_cache'):
                        pipeline.vae.decoder.conv_cache = None
                    
                # Force UNet to clear warmed-up optimization caches
                if hasattr(pipeline, 'unet'):
                    unet = pipeline.unet
                    # Clear any persistent attention caches
                    for name, module in unet.named_modules():
                        if hasattr(module, '_attention_cache'):
                            module._attention_cache = None
                        if hasattr(module, '_conv_cache'):
                            module._conv_cache = None
                        if hasattr(module, '_cached_weights'):
                            module._cached_weights = None
        
        # 4. AGGRESSIVE PyTorch lazy tensor state clearing
        try:
            import torch
            if torch.cuda.is_available():
                # Clear any global PyTorch caches that could affect lazy tensors
                torch.cuda.empty_cache()
                
                # Clear autograd computation graph
                if hasattr(torch.autograd, 'graph'):
                    try:
                        torch.autograd.graph.clear()
                    except:
                        pass
                
                # Force synchronization to ensure all operations complete
                torch.cuda.synchronize()
                
                # Clear compilation caches if using torch.compile
                if hasattr(torch, '_dynamo'):
                    try:
                        torch._dynamo.reset()
                    except:
                        pass
                        
        except Exception as e:
            logger.warning(f"  ⚠️ Error during PyTorch state clearing: {e}")
        
        # 5. FORCE CUDA CONTEXT REFRESH (for warmed-up GPU issues)
        if torch.cuda.is_available():
            # This is aggressive - forces CUDA to refresh its context
            current_device = torch.cuda.current_device()
            
            # Temporarily switch device to force context refresh
            try:
                if torch.cuda.device_count() > 1:
                    torch.cuda.set_device((current_device + 1) % torch.cuda.device_count())
                    torch.cuda.empty_cache()
                    torch.cuda.set_device(current_device)
                    logger.info("  🔄 Forced CUDA context refresh")
            except:
                pass
            
            # Final comprehensive cache clear
            torch.cuda.empty_cache()
        
        logger.info("  ✅ AGGRESSIVE GPU state reset completed")
        
    except Exception as e:
        logger.warning(f"  ⚠️ Error during aggressive GPU state reset: {e}")
        # Fallback to standard reset
        _comprehensive_texture_pipeline_reset(tex_pipeline)

def _aggressive_shape_generation_reset(shape_pipeline):
    """Aggressive state reset for shape generation models to prevent geometry contamination."""
    try:
        logger.info("  🎯 Performing AGGRESSIVE shape generation reset for warmed-up GPU...")
        
        # 1. Reset Diffusion Model State
        if hasattr(shape_pipeline, 'model') and hasattr(shape_pipeline.model, 'named_modules'):
            for name, module in shape_pipeline.model.named_modules():
                for attr in ['_attention_cache', '_key_cache', '_value_cache', '_conv_cache', '_cached_hidden_states']:
                    if hasattr(module, attr):
                        setattr(module, attr, None)
        
        # 2. Reset VAE State (critical for geometry contamination)
        if hasattr(shape_pipeline, 'vae'):
            vae = shape_pipeline.vae
            if hasattr(vae, 'encoder') and hasattr(vae.encoder, 'named_modules'):
                for name, module in vae.encoder.named_modules():
                    for attr in ['_cached_features', '_attention_cache']:
                        if hasattr(module, attr):
                            setattr(module, attr, None)
            
            decoder = getattr(vae, 'decoder', None) or getattr(vae, 'geo_decoder', None)
            if decoder and hasattr(decoder, 'named_modules'):
                for name, module in decoder.named_modules():
                    for attr in ['_cached_features', '_attention_cache']:
                        if hasattr(module, attr):
                            setattr(module, attr, None)
            
            if hasattr(vae, 'volume_decoder'):
                volume_decoder = vae.volume_decoder
                if hasattr(volume_decoder, 'named_modules'):
                    for name, module in volume_decoder.named_modules():
                        for attr in ['_cached_volume', '_grid_cache']:
                            if hasattr(module, attr):
                                setattr(module, attr, None)
                else:
                    logger.info(f"    🔧 Clearing custom volume decoder: {type(volume_decoder).__name__}")
                    for attr in ['_cached_volume', '_grid_cache', '_volume_cache', '_output_cache', 
                                 '_last_latent', '_cached_geometry', '_cached_features', '_internal_state']:
                        if hasattr(volume_decoder, attr):
                            setattr(volume_decoder, attr, None)

        # 3. Reset Conditioner State
        if hasattr(shape_pipeline, 'conditioner') and hasattr(shape_pipeline.conditioner, 'named_modules'):
            for name, module in shape_pipeline.conditioner.named_modules():
                for attr in ['_feature_cache', '_embedding_cache', 'last_hidden_state']:
                    if hasattr(module, attr):
                        setattr(module, attr, None)
        
        # 4. Reset Scheduler State
        if hasattr(shape_pipeline, 'scheduler'):
            scheduler = shape_pipeline.scheduler
            scheduler_class = type(scheduler)
            scheduler_config = scheduler.config if hasattr(scheduler, 'config') else {}
            try:
                shape_pipeline.scheduler = scheduler_class.from_config(scheduler_config) if scheduler_config else scheduler_class()
                logger.info("  🔄 Forced shape generation scheduler reinitialization")
            except:
                for attr in ['_step_index', 'model_outputs', 'sample']:
                    if hasattr(scheduler, attr):
                        setattr(scheduler, attr, None)
                logger.warning("  ⚠️ Manual scheduler state clear (reinitialization failed)")
        
        # 5. Clear cached trimesh geometry from pipeline object
        logger.info("    🔍 Deep cleaning shape pipeline for cached trimesh objects...")
        if hasattr(shape_pipeline, '__dict__'):
            for attr in list(shape_pipeline.__dict__.keys()):
                if 'trimesh' in str(type(getattr(shape_pipeline, attr, None))):
                    try:
                        delattr(shape_pipeline, attr)
                        logger.info(f"      🧹 CRITICAL: Cleared cached trimesh object: {attr}")
                    except:
                        pass

        # 6. FORCE MEMORY CLEANUP
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.synchronize()
        
        logger.info("  ✅ AGGRESSIVE shape generation reset completed")
        
    except Exception as e:
        logger.warning(f"  ⚠️ Error during aggressive shape generation reset: {e}")

class VRAMMonitor:
    """Utility class for thread-safe CUDA VRAM queries."""

    def __init__(self):
        self._lock = threading.Lock()

    def get_available_vram(self) -> float:
        """Return available VRAM (GB) on the current CUDA device in a thread-safe way."""
        with self._lock:
            if not torch.cuda.is_available():
                return 0.0
            device_id = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device_id).total_memory / 1024 ** 3
            allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3
            return total - allocated

    def get_used_vram(self) -> float:
        """Return used VRAM (GB) on the current CUDA device in a thread-safe way."""
        with self._lock:
            if not torch.cuda.is_available():
                return 0.0
            device_id = torch.cuda.current_device()
            return torch.cuda.memory_allocated(device_id) / 1024 ** 3

    def check_parallel_safety(self, required_per_worker: float, num_workers: int) -> bool:
        """Check whether enough free VRAM exists to run <num_workers> jobs that each
        need <required_per_worker> GB. Adds a small safety buffer."""
        available = self.get_available_vram()
        total_required = required_per_worker * num_workers
        safety_buffer = 4.0  # GB
        return available >= (total_required + safety_buffer)

# Monkey patch volume decoding to prevent memory collision during parallel processing
def _patched_latents2mesh(self, latents: torch.FloatTensor, **kwargs):
    """Patched latents2mesh method that serializes volume decoding operations"""
    import time
    
    # Serialize volume decoding to prevent memory collision
    with volume_decoding_lock:
        logger.info("  🔒 Acquiring volume decoding lock for serialized processing...")
        start_time = time.time()
        grid_logits = self.volume_decoder(latents, self.geo_decoder, **kwargs)
        decode_time = time.time() - start_time
        logger.info(f"  🔓 Volume decoding completed in {decode_time:.1f}s, releasing lock...")
    
    # Surface extraction can still run in parallel
    start_time = time.time()
    outputs = self.surface_extractor(grid_logits, **kwargs)
    extract_time = time.time() - start_time
    logger.info(f"  Surface extraction completed in {extract_time:.1f}s")
    return outputs

# Apply the monkey patch
def _apply_volume_decoding_patch():
    """Apply volume decoding serialization patch for parallel processing safety"""
    try:
        from hy3dshape.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        # Store original method
        original_latents2mesh = Hunyuan3DDiTFlowMatchingPipeline.latents2mesh

        # Apply the patch
        Hunyuan3DDiTFlowMatchingPipeline.latents2mesh = _patched_latents2mesh
        logger.info("✅ Volume decoding serialization patch applied successfully")
    except ImportError as e:
        logger.warning(f"⚠️ Could not apply volume decoding patch: {e}")
    except Exception as e:
        logger.warning(f"⚠️ Volume decoding patch failed: {e}")

# Model loading state tracking
_models_loading_state = {
    'rembg': False,
    'shape': False,
    'texture': False,
    'postprocessing': False
}

# Legacy initialize_models function replaced with lazy loading methods above

# Constants
CHECKPOINTS_PATH = "/src/checkpoints"
HUNYUAN3D_MODEL_PATH = "tencent/Hunyuan3D-2.1"
U2NET_PATH = os.path.join(CHECKPOINTS_PATH, ".u2net/")
U2NET_URL = "https://weights.replicate.delivery/default/comfy-ui/rembg/u2net.onnx.tar"
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# Absolute path to the RealESRGAN checkpoint (robust to CWD changes)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
REALESRGAN_CKPT = PROJECT_ROOT / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
MULTIVIEW_CFG = PROJECT_ROOT / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml"

# Set U2NET_HOME now that U2NET_PATH is defined
os.environ["U2NET_HOME"] = U2NET_PATH

def download_if_not_exists(url, dest):
    if not os.path.exists(dest):
        start = time.time()
        os.makedirs(dest, exist_ok=True)
        logger.info(f"downloading url: {url}")
        logger.info(f"downloading to: {dest}")
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
        duration = time.time() - start
        logger.info(f"downloading took: {duration:.2f}s")

def download_file_if_not_exists(url, dest_path):
    if not os.path.exists(dest_path):
        start = time.time()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        logger.info(f"downloading url: {url}")
        logger.info(f"downloading to: {dest_path}")
        subprocess.check_call(["wget", "-O", dest_path, url], close_fds=False)
        duration = time.time() - start
        logger.info(f"downloading took: {duration:.2f}s")

# Lazy Loading Architecture - Critical for Replicate cold start performance
def _ensure_rembg_loaded():
    """Ensure background removal model is loaded"""
    global rmbg_worker, _models_loading_state
    if rmbg_worker is None and not _models_loading_state['rembg']:
        _models_loading_state['rembg'] = True
        logger.info("Loading background removal model on-demand...")
        rmbg_worker = BackgroundRemover()
        logger.info("Background removal model loaded")
    return rmbg_worker

def _ensure_shape_model_loaded():
    """Ensure shape generation model is loaded"""
    global i23d_worker, _models_loading_state
    if i23d_worker is None and not _models_loading_state['shape']:
        _models_loading_state['shape'] = True
        logger.info("Loading shape generation model on-demand...")
        i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2.1"
        )
        logger.info("Shape generation model loaded")
    return i23d_worker

def _ensure_texture_model_loaded():
    """Ensure texture generation model is loaded"""
    global tex_pipeline, _models_loading_state
    if tex_pipeline is None and not _models_loading_state['texture']:
        _models_loading_state['texture'] = True
        logger.info("Loading texture generation model on-demand...")
        max_num_view = 6
        resolution = 512
        tex_conf = Hunyuan3DPaintConfig(max_num_view, resolution)
        tex_conf.realesrgan_ckpt_path = str(REALESRGAN_CKPT)
        tex_conf.multiview_cfg_path = str(MULTIVIEW_CFG)
        tex_conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"

        # Fallback: Download RealESRGAN model if missing
        if not os.path.exists(tex_conf.realesrgan_ckpt_path):
            logger.info("RealESRGAN model not found, downloading...")
            os.makedirs(os.path.dirname(REALESRGAN_CKPT), exist_ok=True)
            subprocess.run([
                "wget", 
                REALESRGAN_URL,
                "-O", str(REALESRGAN_CKPT)
            ], check=True)

        tex_pipeline = Hunyuan3DPaintPipeline(tex_conf)
        logger.info("Texture generation model loaded")
    return tex_pipeline

def _ensure_postprocessing_loaded():
    """Ensure mesh post-processing workers are loaded"""
    global floater_remove_worker, degenerate_face_remove_worker, face_reduce_worker, mesh_simplifier, _models_loading_state
    if floater_remove_worker is None and not _models_loading_state['postprocessing']:
        _models_loading_state['postprocessing'] = True
        logger.info("Loading mesh processing tools on-demand...")
        floater_remove_worker = FloaterRemover()
        degenerate_face_remove_worker = DegenerateFaceRemover()
        face_reduce_worker = FaceReducer()
        mesh_simplifier = MeshSimplifier()
        logger.info("Mesh processing tools loaded")
    return floater_remove_worker, degenerate_face_remove_worker, face_reduce_worker, mesh_simplifier

# Per-Worker Model Loading Functions for Thread Safety
def _ensure_worker_models_loaded():
    """Ensure each worker has its own model instances (thread-safe)"""
    thread_id = threading.current_thread().ident
    
    with worker_models_lock:
        if thread_id not in worker_models:
            logger.info(f"🔧 [Worker-{thread_id}] Loading dedicated model instances...")
            
            # Initialize worker model storage
            worker_models[thread_id] = {}
            
            # Load background removal model
            logger.info(f"  [Worker-{thread_id}] Loading BackgroundRemover...")
            worker_models[thread_id]['rembg'] = BackgroundRemover()
            
            # Load shape generation model 
            logger.info(f"  [Worker-{thread_id}] Loading Hunyuan3D shape model...")
            worker_models[thread_id]['shape'] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2.1"
            )
            
            # Load texture generation model
            logger.info(f"  [Worker-{thread_id}] Loading texture generation model...")
            max_num_view = 6
            resolution = 512
            tex_conf = Hunyuan3DPaintConfig(max_num_view, resolution)
            tex_conf.realesrgan_ckpt_path = str(REALESRGAN_CKPT)
            tex_conf.multiview_cfg_path = str(MULTIVIEW_CFG)
            tex_conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
            
            worker_models[thread_id]['texture'] = Hunyuan3DPaintPipeline(tex_conf)
            
            # Load post-processing tools (these can be shared as they're stateless)
            logger.info(f"  [Worker-{thread_id}] Loading mesh processing tools...")
            worker_models[thread_id]['floater_remover'] = FloaterRemover()
            worker_models[thread_id]['degenerate_remover'] = DegenerateFaceRemover()
            worker_models[thread_id]['face_reducer'] = FaceReducer()
            worker_models[thread_id]['mesh_simplifier'] = MeshSimplifier()
            
            logger.info(f"✅ [Worker-{thread_id}] All dedicated models loaded")
    
    return worker_models[thread_id]

def _get_worker_model(model_type):
    """Get specific model instance for current worker thread"""
    thread_id = threading.current_thread().ident
    if thread_id in worker_models and model_type in worker_models[thread_id]:
        return worker_models[thread_id][model_type]
    else:
        # Fallback to global models if worker models not loaded
        if model_type == 'rembg':
            return _ensure_rembg_loaded()
        elif model_type == 'shape':
            return _ensure_shape_model_loaded()
        elif model_type == 'texture':
            return _ensure_texture_model_loaded()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

def _cleanup_worker_models():
    """Clean up worker model instances when done"""
    thread_id = threading.current_thread().ident
    with worker_models_lock:
        if thread_id in worker_models:
            logger.info(f"🧹 [Worker-{thread_id}] Cleaning up worker models...")
            # Models will be garbage collected
            del worker_models[thread_id]

def validate_zip_file(zip_path: Path) -> bool:
    """Validate ZIP file integrity"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Test ZIP file integrity
            bad_file = zip_ref.testzip()
            if bad_file:
                logger.error(f"Corrupted file in ZIP: {bad_file}")
                return False
            
            # Check if ZIP contains any valid image files
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
            has_images = any(
                os.path.splitext(file_info.filename.lower())[1] in image_extensions
                for file_info in zip_ref.filelist
                if not file_info.is_dir()
            )
            
            if not has_images:
                logger.error("ZIP file contains no valid image files")
                return False
                
            return True
    except zipfile.BadZipFile:
        logger.error("Invalid ZIP file format")
        return False
    except Exception as e:
        logger.error(f"ZIP validation error: {str(e)}")
        return False

def validate_image_file(image_path: str) -> bool:
    """Validate individual image file"""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify image integrity
            
            # Re-open for format check (verify() closes the file)
            with Image.open(image_path) as img:
                # Check minimum dimensions
                if img.width < 32 or img.height < 32:
                    logger.warning(f"Image too small: {img.width}x{img.height}")
                    return False
                
                # Check maximum dimensions to prevent memory issues
                if img.width > 4096 or img.height > 4096:
                    logger.warning(f"Image too large: {img.width}x{img.height}")
                    return False
                    
                return True
    except Exception as e:
        logger.error(f"Image validation failed for {image_path}: {str(e)}")
        return False

class Output(BaseModel):
    mesh: Path
    batch_results: Path = None  # For batch processing results

def extract_zip_images(zip_path: Path, extract_dir: str) -> List[str]:
    """Extract images from ZIP file and return list of valid image paths"""
    # Validate ZIP file first
    if not validate_zip_file(zip_path):
        raise ValueError("Invalid or corrupted ZIP file")
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    image_paths = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.filelist:
            if not file_info.is_dir():
                file_ext = os.path.splitext(file_info.filename.lower())[1]
                if file_ext in image_extensions:
                    try:
                        # Extract the file
                        zip_ref.extract(file_info, extract_dir)
                        extracted_path = os.path.join(extract_dir, file_info.filename)
                        
                        # Validate extracted image
                        if validate_image_file(extracted_path):
                            image_paths.append(extracted_path)
                        else:
                            logger.warning(f"Skipping invalid image: {file_info.filename}")
                            # Clean up invalid file
                            try:
                                os.remove(extracted_path)
                            except:
                                pass
                    except Exception as e:
                        logger.warning(f"Failed to extract {file_info.filename}: {str(e)}")
                        continue
    
    if len(image_paths) == 0:
        raise ValueError("No valid images found in ZIP file after validation")
    
    return sorted(image_paths)

def create_batch_zip(meshes_dir: str, results_json_path: str, output_zip_path: str):
    """Create ZIP file containing all batch results"""
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        # Add results JSON
        zip_ref.write(results_json_path, 'batch_results.json')
        
        # Add all mesh files
        if os.path.exists(meshes_dir):
            for filename in os.listdir(meshes_dir):
                if filename.endswith('.glb'):
                    file_path = os.path.join(meshes_dir, filename)
                    zip_ref.write(file_path, f'meshes/{filename}')

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Fast setup for Replicate - models loaded on-demand for optimal cold start"""
        
        logger.info("Setup started - using lazy loading for optimal performance")
        
        # Apply volume decoding serialization patch for parallel processing safety
        _apply_volume_decoding_patch()
        
        # Initialize VRAM monitor and cleanup lock for thread safety
        self.vram_monitor = VRAMMonitor()
        self._cleanup_lock = threading.Lock()
        
        # Initial GPU memory cleanup
        self._cleanup_gpu_memory()
        
        # Download critical dependencies if needed (non-blocking for models)
        download_if_not_exists(U2NET_URL, U2NET_PATH)
        
        logger.info("Setup completed - models will load on-demand")
    
    def _cleanup_gpu_memory(self):
        """Thread-safe GPU memory cleanup"""
        with self._cleanup_lock:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
            
    # HF-style shape generation function (mimicking their exact pattern)
    def _hf_style_gen_shape(self, image, steps=50, guidance_scale=5.5, seed=1234, 
                           octree_resolution=512, num_chunks=200000):
        """Generate shape using HF Space pattern with lazy-loaded model"""
        
        logger.info(f"HF-style shape generation: steps={steps}, guidance_scale={guidance_scale}")
        
        # Ensure shape model is loaded
        shape_worker = _ensure_shape_model_loaded()
        
        # Use lazy-loaded worker (HF pattern)
        generator = torch.Generator()
        generator = generator.manual_seed(int(seed))
        
        # Direct call to lazy-loaded worker (exactly like HF Space)
        outputs = shape_worker(
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            output_type='mesh'
        )
        
        # Convert to trimesh using HF demo pattern
        logger.info("  Converting Latent2MeshOutput to trimesh...")
        mesh = export_to_trimesh(outputs)[0]
        logger.info(f"  Converted to mesh - Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
        
        return [mesh]  # Return as list to match expected format

    def _generate_shape(self, image, steps, guidance_scale, seed, octree_resolution, num_chunks):
        """Generate 3D shape from image"""
        import time
        start_time = time.time()
        
        logger.info(f"  Starting shape generation with {steps} steps, guidance_scale={guidance_scale}")
        logger.info(f"  GPU Memory before generation: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
        
        # DEBUG: Log generator creation
        logger.info("  DEBUG: Creating generator...")
        generator = torch.Generator()
        generator = generator.manual_seed(int(seed))
        logger.info(f"  DEBUG: Generator created with seed {seed}")
        
        # DEBUG: Log before model pipeline call
        logger.info("  DEBUG: About to call i23d_worker pipeline...")
        
        outputs = self.i23d_worker(
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            output_type='mesh'
        )
        
        # DEBUG: Log after model pipeline call
        logger.info("  DEBUG: i23d_worker pipeline call completed")
        
        generation_time = time.time() - start_time
        logger.info(f"  Shape generation completed in {generation_time:.1f} seconds")
        logger.info(f"  GPU Memory after generation: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # Clean up GPU memory after generation (HF-style)
        self._cleanup_gpu_memory()
        
        return outputs

    def _check_memory_safety(self, min_required_gb: float = 12.0) -> bool:
        """Check if we have enough memory to proceed safely"""
        available = self.vram_monitor.get_available_vram()
        if available < min_required_gb:
            logger.warning(f"Low VRAM warning: {available:.1f}GB available, {min_required_gb}GB required")
            self._cleanup_gpu_memory()
            available = self.vram_monitor.get_available_vram()
            
            # More aggressive cleanup if still low
            if available < min_required_gb:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                available = self.vram_monitor.get_available_vram()
                
            return available >= min_required_gb
        return True

    def _log_analytics_event(self, event_name, params=None):
        """Analytics stub - safe to call even if analytics service unavailable"""
        try:
            # In production, you might want to log to analytics service
            logger.info(f"Analytics: {event_name} - {params}")
        except Exception as e:
            # Never let analytics logging break the main pipeline
            logger.debug(f"Analytics logging failed: {e}")
            pass

    def _process_single_image_worker(self, 
                            image_input: Union[Path, str], 
                            output_dir: str,
                            image_idx: int,
                            **kwargs) -> dict:
        """
        Process a single image using worker-specific model instances (thread-safe)
        Returns metadata dict for the image
        """
        start_time = time.time()
        thread_id = threading.current_thread().ident
        
        # CRITICAL: Always preserve original filename (without extension) for output mesh
        if isinstance(image_input, str):
            image_name = os.path.splitext(os.path.basename(image_input))[0]
        else:
            image_name = os.path.splitext(os.path.basename(str(image_input)))[0]
        
        metadata = {
            "input_image": image_name,
            "output_mesh": f"{image_name}.glb",
            "status": "error",
            "duration": 0.0,
            "face_count": 0,
            "vertex_count": 0,
            "error": None,
            "error_type": None
        }
        
        try:
            # Load and preprocess image with validation
            if isinstance(image_input, str):
                if not validate_image_file(image_input):
                    raise ValueError(f"Invalid image file: {image_input}")
                input_image = Image.open(image_input).convert("RGB")
            else:
                input_image = Image.open(str(image_input)).convert("RGB")

            # Background removal with worker-specific model
            if kwargs.get('remove_background', True):
                logger.info(f"  [Worker-{thread_id}] Removing background for {image_name}")
                rmbg_worker = _get_worker_model('rembg')
                processed_image = rmbg_worker(input_image)
            else:
                processed_image = input_image
            
            # Shape generation with worker-specific model
            logger.info(f"  [Worker-{thread_id}] Starting shape generation for {image_name}")
            shape_worker = _get_worker_model('shape')
            
            # AGGRESSIVE state cleanup for warmed-up GPU contamination issues
            _aggressive_shape_generation_reset(shape_worker)
            
            # Use worker-specific shape generation
            generator = torch.Generator()
            generator = generator.manual_seed(int(kwargs.get('seed', 1234)) + image_idx)
            
            outputs = shape_worker(
                image=processed_image,
                num_inference_steps=kwargs.get('steps', 50),
                guidance_scale=kwargs.get('guidance_scale', 5.5),
                generator=generator,
                octree_resolution=kwargs.get('octree_resolution', 512),
                num_chunks=kwargs.get('num_chunks', 200000),
                output_type='mesh'
            )
            
            # Clean up GPU memory after generation
            self._cleanup_gpu_memory()
            
            # Check if mesh generation was successful
            if outputs is None or len(outputs) == 0:
                raise RuntimeError("Shape generation failed - no mesh output")
            
            # Convert to trimesh
            from hy3dshape.pipelines import export_to_trimesh
            mesh = export_to_trimesh(outputs)[0]
            if mesh is None or not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                raise RuntimeError("Shape generation failed - empty mesh")
            
            logger.info(f"  [Worker-{thread_id}] Generated mesh - Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
            
            # Post-process mesh with worker-specific models
            logger.info(f"  [Worker-{thread_id}] Post-processing mesh for {image_name}")
            # Use a local reference so we don't overwrite the global `worker_models` dict that
            # stores the pre-loaded per-thread model sets used for parallel execution.
            worker_models_local = _ensure_worker_models_loaded()
            
            # Apply post-processing pipeline using worker models
            mesh_output = worker_models_local['floater_remover'](mesh)
            if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                raise RuntimeError("Mesh became empty after floater removal")
                
            mesh_output = worker_models_local['degenerate_remover'](mesh_output)
            if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                raise RuntimeError("Mesh became empty after degenerate face removal")
            
            # Face reduction (always needed)
            mesh_output = worker_models_local['face_reducer'](mesh_output, max_facenum=kwargs.get('max_facenum', 40000))
            if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                raise RuntimeError("Mesh became empty after face reduction")
                
            self._cleanup_gpu_memory()

            # Save intermediate mesh
            temp_mesh_path = os.path.join(output_dir, f"{image_name}_temp.obj")
            mesh_output.export(temp_mesh_path)

            # Apply texturing with worker-specific model
            logger.info(f"  [Worker-{thread_id}] Generating texture for {image_name}")
            tex_pipeline = worker_models_local['texture']
            textured_mesh_path = tex_pipeline(
                mesh_path=temp_mesh_path,
                image_path=input_image,
                output_mesh_path=os.path.join(output_dir, f"{image_name}_textured.obj")
            )

            # Export final GLB with thread-safe filename to prevent parallel processing conflicts
            from trimesh import load as load_trimesh
            final_mesh = load_trimesh(textured_mesh_path)
            # Ensure completely unique output path to prevent file overwrites during parallel processing
            unique_output_path = os.path.join(output_dir, f"{image_name}_{thread_id}.glb")
            final_mesh.export(unique_output_path, include_normals=True)
            
            # Rename to final expected filename atomically
            final_output_path = os.path.join(output_dir, f"{image_name}.glb")
            os.rename(unique_output_path, final_output_path)

            # Update metadata with success
            metadata.update({
                "status": "success",
                "duration": time.time() - start_time,
                "face_count": len(final_mesh.faces),
                "vertex_count": len(final_mesh.vertices),
                "error": None,
                "error_type": None
            })

            # Cleanup intermediate files
            try:
                os.remove(temp_mesh_path)
                if os.path.exists(textured_mesh_path):
                    os.remove(textured_mesh_path)
            except:
                pass  # Don't fail if cleanup fails

            logger.info(f"  [Worker-{thread_id}] ✅ {image_name} completed in {metadata['duration']:.1f}s, faces: {metadata['face_count']}")
            
            return metadata

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            metadata.update({
                "status": "error",
                "duration": time.time() - start_time,
                "error": error_msg,
                "error_type": error_type
            })
            
            logger.error(f"[Worker-{thread_id}] Failed to process {image_name}: {error_msg}")
            logger.error(traceback.format_exc())
            return metadata
        finally:
            self._cleanup_gpu_memory()

    def _process_single_image_worker_direct(self, 
                            image_input: Union[Path, str], 
                            output_dir: str,
                            image_idx: int,
                            worker_key: str,
                            **kwargs) -> dict:
        """
        Process a single image using pre-loaded worker-specific models (lock-free)
        Returns metadata dict for the image
        """
        start_time = time.time()
        thread_id = threading.current_thread().ident
        
        # CRITICAL: Always preserve original filename (without extension) for output mesh
        if isinstance(image_input, str):
            image_name = os.path.splitext(os.path.basename(image_input))[0]
        else:
            image_name = os.path.splitext(os.path.basename(str(image_input)))[0]
        
        metadata = {
            "input_image": image_name,
            "output_mesh": f"{image_name}.glb",
            "status": "error",
            "duration": 0.0,
            "face_count": 0,
            "vertex_count": 0,
            "error": None,
            "error_type": None
        }
        
        try:
            # Load and preprocess image with validation
            if isinstance(image_input, str):
                if not validate_image_file(image_input):
                    raise ValueError(f"Invalid image file: {image_input}")
                input_image = Image.open(image_input).convert("RGB")
            else:
                input_image = Image.open(str(image_input)).convert("RGB")

            # Get pre-loaded models for this worker (no locking needed)
            worker_models_set = worker_models[worker_key]

            # Background removal with worker-specific model
            if kwargs.get('remove_background', True):
                logger.info(f"  [Worker-{thread_id}] Removing background for {image_name}")
                processed_image = worker_models_set['rembg'](input_image)
            else:
                processed_image = input_image
            
            # Shape generation with worker-specific model
            logger.info(f"  [Worker-{thread_id}] Starting shape generation for {image_name}")
            
            # AGGRESSIVE state cleanup for warmed-up GPU contamination issues
            _aggressive_shape_generation_reset(worker_models_set['shape'])
            
            # Use worker-specific shape generation (no locking!)
            generator = torch.Generator()
            generator = generator.manual_seed(int(kwargs.get('seed', 1234)) + image_idx)
            
            with shape_gpu_gate:
                outputs = worker_models_set['shape'](
                    image=processed_image,
                    num_inference_steps=kwargs.get('steps', 50),
                    guidance_scale=kwargs.get('guidance_scale', 5.5),
                    generator=generator,
                    octree_resolution=kwargs.get('octree_resolution', 512),
                    num_chunks=kwargs.get('num_chunks', 200000),
                    output_type='mesh'
                )
            
            # Clean up GPU memory after generation
            self._cleanup_gpu_memory()
            
            # Check if mesh generation was successful
            if outputs is None or len(outputs) == 0:
                raise RuntimeError("Shape generation failed - no mesh output")
            
            # Convert to trimesh
            from hy3dshape.pipelines import export_to_trimesh
            mesh = export_to_trimesh(outputs)[0]
            if mesh is None or not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                raise RuntimeError("Shape generation failed - empty mesh")
            
            logger.info(f"  [Worker-{thread_id}] Generated mesh - Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
            
            # SERIALIZE ONLY mesh processing to prevent memory overload
            logger.info(f"  [Worker-{thread_id}] Waiting for mesh processing slot...")
            with mesh_processing_lock:
                logger.info(f"  [Worker-{thread_id}] Starting SERIALIZED mesh processing for {image_name}")
                
                # Apply post-processing pipeline using worker models (SERIALIZED)
                mesh_output = worker_models_set['floater_remover'](mesh)
                if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                    raise RuntimeError("Mesh became empty after floater removal")
                    
                mesh_output = worker_models_set['degenerate_remover'](mesh_output)
                if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                    raise RuntimeError("Mesh became empty after degenerate face removal")
                
                # Face reduction (always needed)
                mesh_output = worker_models_set['face_reducer'](mesh_output, max_facenum=kwargs.get('max_facenum', 40000))
                if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                    raise RuntimeError("Mesh became empty after face reduction")
                    
                self._cleanup_gpu_memory()

                # Save intermediate mesh
                temp_mesh_path = os.path.join(output_dir, f"{image_name}_temp.obj")
                mesh_output.export(temp_mesh_path)
                
                logger.info(f"  [Worker-{thread_id}] Completed SERIALIZED mesh processing for {image_name}")

            # MEMORY-AWARE texture generation with robust file handling
            available_vram = self.vram_monitor.get_available_vram()
            texture_memory_estimate = 18.0  # GB - conservative estimate for texture generation
            
            # Create worker-specific temp mesh path to prevent file conflicts
            worker_temp_mesh_path = os.path.join(output_dir, f"{image_name}_worker_{thread_id}_temp.obj")
            
            # Ensure worker temp mesh path is absolute to prevent texture pipeline path issues
            import pathlib
            worker_temp_mesh_path = str(pathlib.Path(worker_temp_mesh_path).resolve())
            
            # Robust file copy with validation and retry logic
            max_retries = 3
            copy_success = False
            
            for attempt in range(max_retries):
                try:
                    # Validate source file exists and is readable
                    if not os.path.exists(temp_mesh_path):
                        raise FileNotFoundError(f"Source mesh file not found: {temp_mesh_path}")
                    
                    if os.path.getsize(temp_mesh_path) == 0:
                        raise ValueError(f"Source mesh file is empty: {temp_mesh_path}")
                    
                    # Copy with verification
                    import shutil
                    shutil.copy2(temp_mesh_path, worker_temp_mesh_path)
                    
                    # Verify copy was successful
                    if not os.path.exists(worker_temp_mesh_path):
                        raise FileNotFoundError(f"Worker temp file was not created: {worker_temp_mesh_path}")
                    
                    if os.path.getsize(worker_temp_mesh_path) != os.path.getsize(temp_mesh_path):
                        raise ValueError(f"Worker temp file size mismatch")
                    
                    copy_success = True
                    break
                    
                except Exception as e:
                    logger.warning(f"  [Worker-{thread_id}] File copy attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(0.1)  # Brief pause before retry (time module already imported at top of file)
                    else:
                        raise RuntimeError(f"Failed to create worker temp mesh file after {max_retries} attempts: {e}")
            
            if not copy_success:
                raise RuntimeError("File copy failed after all retry attempts")
            
            if available_vram > texture_memory_estimate + 5.0:  # 5GB safety buffer
                # Sufficient VRAM for parallel texture generation
                logger.info(f"  [Worker-{thread_id}] Generating texture for {image_name} (PARALLEL - {available_vram:.1f}GB available)")
                
                # AGGRESSIVE state cleanup for warmed-up GPU contamination issues
                _aggressive_gpu_state_reset(worker_models_set['texture'])
                
                # Ensure absolute path for texture output to prevent path duplication
                output_textured_path = str(pathlib.Path(os.path.join(output_dir, f"{image_name}_worker_{thread_id}_textured.obj")).resolve())
                textured_mesh_path = worker_models_set['texture'](
                    mesh_path=worker_temp_mesh_path,
                    image_path=input_image,
                    output_mesh_path=output_textured_path
                )
                logger.info(f"  [Worker-{thread_id}] Completed PARALLEL texture generation for {image_name}")

            else:
                # Insufficient VRAM - use serialized texture generation
                with texture_generation_lock:
                    logger.info(f"  [Worker-{thread_id}] Generating texture for {image_name} (SERIALIZED for VRAM safety)")
                    
                    # AGGRESSIVE state cleanup for warmed-up GPU contamination issues
                    _aggressive_gpu_state_reset(worker_models_set['texture'])
                    
                    # Ensure absolute path for texture output to prevent path duplication
                    output_textured_path = str(pathlib.Path(os.path.join(output_dir, f"{image_name}_worker_{thread_id}_textured.obj")).resolve())
                    textured_mesh_path = worker_models_set['texture'](
                        mesh_path=worker_temp_mesh_path,
                        image_path=input_image,
                        output_mesh_path=output_textured_path
                    )
                    logger.info(f"  [Worker-{thread_id}] Completed SERIALIZED texture generation for {image_name}")

            # Clean up worker-specific temp file
            try:
                os.remove(worker_temp_mesh_path)
            except:
                pass  # Don't fail if cleanup fails

            # Export final GLB with thread-safe filename to prevent parallel processing conflicts
            from trimesh import load as load_trimesh
            final_mesh = load_trimesh(textured_mesh_path)
            # Ensure completely unique output path to prevent file overwrites during parallel processing
            unique_output_path = os.path.join(output_dir, f"{image_name}_{thread_id}.glb")
            final_mesh.export(unique_output_path, include_normals=True)
            
            # Rename to final expected filename atomically
            final_output_path = os.path.join(output_dir, f"{image_name}.glb")
            os.rename(unique_output_path, final_output_path)

            # Update metadata with success
            metadata.update({
                "status": "success",
                "duration": time.time() - start_time,
                "face_count": len(final_mesh.faces),
                "vertex_count": len(final_mesh.vertices),
                "error": None,
                "error_type": None
            })

            # Cleanup intermediate files
            try:
                os.remove(temp_mesh_path)
                if os.path.exists(textured_mesh_path):
                    os.remove(textured_mesh_path)
            except:
                pass  # Don't fail if cleanup fails

            logger.info(f"  [Worker-{thread_id}] ✅ {image_name} completed in {metadata['duration']:.1f}s, faces: {metadata['face_count']}")
            
            return metadata

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            metadata.update({
                "status": "error",
                "duration": time.time() - start_time,
                "error": error_msg,
                "error_type": error_type
            })
            
            logger.error(f"[Worker-{thread_id}] Failed to process {image_name}: {error_msg}")
            logger.error(traceback.format_exc())
            return metadata
        finally:
            self._cleanup_gpu_memory()

    def _process_single_image(self, 
                            image_input: Union[Path, str], 
                            output_dir: str,
                            image_idx: int,
                            **kwargs) -> dict:
        """
        Process a single image with comprehensive error handling
        Returns metadata dict for the image
        """
        start_time = time.time()
        # CRITICAL: Always preserve original filename (without extension) for output mesh
        if isinstance(image_input, str):
            image_name = os.path.splitext(os.path.basename(image_input))[0]
        else:
            image_name = os.path.splitext(os.path.basename(str(image_input)))[0]
        
        metadata = {
            "input_image": image_name,
            "output_mesh": f"{image_name}.glb",
            "status": "error",
            "duration": 0.0,
            "face_count": 0,
            "vertex_count": 0,
            "error": None,
            "error_type": None
        }
        
        try:
            # Memory safety check
            if not self._check_memory_safety():
                raise RuntimeError(f"Insufficient VRAM available ({self.vram_monitor.get_available_vram():.1f}GB)")

            # Load and preprocess image with validation
            if isinstance(image_input, str):
                if not validate_image_file(image_input):
                    raise ValueError(f"Invalid image file: {image_input}")
                input_image = Image.open(image_input).convert("RGB")
            else:
                input_image = Image.open(str(image_input)).convert("RGB")

            # Background removal with lazy loading
            if kwargs.get('remove_background', True):
                logger.info(f"  Removing background for {image_name}")
                rmbg = _ensure_rembg_loaded()
                processed_image = rmbg(input_image)
            else:
                processed_image = input_image
            
            # Shape generation with lazy loading
            logger.info(f"  Starting shape generation for {image_name}")
            shape_model = _ensure_shape_model_loaded()
            
            # AGGRESSIVE state cleanup for warmed-up GPU contamination issues
            _aggressive_shape_generation_reset(shape_model)
            
            with shape_gpu_gate:
                outputs = self._hf_style_gen_shape(
                    processed_image, 
                    kwargs.get('steps', 50),
                    kwargs.get('guidance_scale', 5.5), 
                    kwargs.get('seed', 1234) + image_idx,  # Incremental seed
                    kwargs.get('octree_resolution', 512),
                    kwargs.get('num_chunks', 200000)
                )
            
            # Clean up GPU memory after generation
            self._cleanup_gpu_memory()
            
            # Check if mesh generation was successful
            if outputs is None or len(outputs) == 0:
                raise RuntimeError("Shape generation failed - no mesh output")
            
            mesh = outputs[0]
            if mesh is None or not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                raise RuntimeError("Shape generation failed - empty mesh")
            
            logger.info(f"  Generated mesh - Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
            
            # Post-process mesh with lazy loading
            logger.info(f"  Post-processing mesh for {image_name}")
            floater_remover, degenerate_remover, face_reducer, mesh_simplifier = _ensure_postprocessing_loaded()
            
            # Apply post-processing pipeline
            mesh_output = floater_remover(mesh)
            if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                raise RuntimeError("Mesh became empty after floater removal")
                
            mesh_output = degenerate_remover(mesh_output)
            if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                raise RuntimeError("Mesh became empty after degenerate face removal")
            
            # Face reduction (always needed)
            mesh_output = face_reducer(mesh_output, max_facenum=kwargs.get('max_facenum', 40000))
            if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                raise RuntimeError("Mesh became empty after face reduction")
                
            self._cleanup_gpu_memory()

            # Save intermediate mesh
            temp_mesh_path = os.path.join(output_dir, f"{image_name}_temp.obj")
            mesh_output.export(temp_mesh_path)

            # Apply texturing with lazy loading
            logger.info(f"  Generating texture for {image_name}")
            tex_pipeline = _ensure_texture_model_loaded()
            
            # AGGRESSIVE state cleanup for warmed-up GPU contamination issues
            _aggressive_gpu_state_reset(tex_pipeline)
            
            textured_mesh_path = tex_pipeline(
                mesh_path=temp_mesh_path,
                image_path=input_image,
                output_mesh_path=os.path.join(output_dir, f"{image_name}_textured.obj")
            )

            # Export final GLB with thread-safe filename to prevent parallel processing conflicts
            from trimesh import load as load_trimesh
            final_mesh = load_trimesh(textured_mesh_path)
            # Use current thread ID for unique filename in sequential processing
            thread_id = threading.current_thread().ident
            unique_output_path = os.path.join(output_dir, f"{image_name}_{thread_id}.glb")
            final_mesh.export(unique_output_path, include_normals=True)
            
            # Rename to final expected filename atomically
            final_output_path = os.path.join(output_dir, f"{image_name}.glb")
            os.rename(unique_output_path, final_output_path)

            # Update metadata with success
            metadata.update({
                "status": "success",
                "duration": time.time() - start_time,
                "face_count": len(final_mesh.faces),
                "vertex_count": len(final_mesh.vertices),
                "error": None,
                "error_type": None
            })

            # Cleanup intermediate files
            try:
                os.remove(temp_mesh_path)
                if os.path.exists(textured_mesh_path):
                    os.remove(textured_mesh_path)
            except:
                pass  # Don't fail if cleanup fails

            logger.info(f"  ✅ {image_name} completed in {metadata['duration']:.1f}s, faces: {metadata['face_count']}")
            
            return metadata

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            metadata.update({
                "status": "error",
                "duration": time.time() - start_time,
                "error": error_msg,
                "error_type": error_type
            })
            
            logger.error(f"Failed to process {image_name}: {error_msg}")
            logger.error(traceback.format_exc())
            return metadata
        finally:
            self._cleanup_gpu_memory()

    def predict(
        self,
        image: Path = Input(description="Input image for generating 3D shape (single image mode)", default=None),
        batch_images: Path = Input(description="ZIP file containing multiple images for batch processing", default=None),
        mesh: Path = Input(description="Optional: Upload a .glb mesh to skip generation and only texture it", default=None),
        prompt: str = Input(description="Text prompt to guide texture generation", default="a detailed texture of a stone sculpture"),
        steps: int = Input(description="Number of inference steps", default=50, ge=20, le=50),
        guidance_scale: float = Input(description="Guidance scale for generation", default=5.5, ge=1.0, le=20.0),
        max_facenum: int = Input(description="Maximum number of faces for mesh generation", default=40000, ge=10000, le=200000),
        num_chunks: int = Input(description="Number of chunks for mesh generation (H100 140GB: use 4-5M chunks)", default=200000, ge=10000, le=5000000),
        seed: int = Input(description="Random seed for generation", default=1234),
        octree_resolution: int = Input(description="Octree resolution for mesh generation", choices=[256, 384, 512], default=512),
        remove_background: bool = Input(description="Whether to remove background from input image", default=True),
        parallel_workers: int = Input(description="Number of parallel workers for batch processing (H100 can handle 2-4)", default=2, ge=1, le=4),
    ) -> Output:
        
        # Centralized output directory management
        self.output_dir = "output"
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(os.path.join(self.output_dir, "meshes"), exist_ok=True)
        
        # Log analytics for every prediction
        self._log_analytics_event("predict_started")
        
        # ======================================================================
        # Route to the correct prediction mode (single, batch, or texture-only)
        # ======================================================================

        if batch_images:
            return self._predict_batch(
                batch_images=batch_images, 
                parallel_workers=parallel_workers, 
                **locals()
            )
        elif image:
            return self._predict_single(**locals())
        elif mesh:
            return self._predict_texture_only(**locals())
        else:
            raise ValueError("You must provide either an 'image', a 'batch_images' ZIP file, or a 'mesh' to texture.")

    def _predict_single(self, **kwargs) -> Output:
        """Single image processing mode"""
        from trimesh import load as load_trimesh

        self._log_analytics_event("predict_mode", {"mode": "single"})

        if os.path.exists("output"):
            shutil.rmtree("output")
        os.makedirs("output", exist_ok=True)

        try:
            if kwargs['mesh']:
                # Mesh-only texturing mode
                self._log_analytics_event("predict_mode", {"mode": "paint_only"})
                mesh_obj = load_trimesh(str(kwargs['mesh']), force="mesh")
                
                # Validate loaded mesh
                if mesh_obj is None:
                    raise ValueError("Failed to load mesh from file")
                if not hasattr(mesh_obj, 'vertices') or len(mesh_obj.vertices) == 0:
                    raise ValueError("Loaded mesh has no vertices")
                if not hasattr(mesh_obj, 'faces') or len(mesh_obj.faces) == 0:
                    raise ValueError("Loaded mesh has no faces")
                
                logger.info(f"Loaded mesh: {len(mesh_obj.vertices)} vertices, {len(mesh_obj.faces)} faces")
                
                # Try mesh simplification with graceful fallback
                try:
                    _, _, _, mesh_simplifier = _ensure_postprocessing_loaded()
                    simplified_mesh = mesh_simplifier(mesh_obj)
                    if simplified_mesh is not None and len(simplified_mesh.vertices) > 0 and len(simplified_mesh.faces) > 0:
                        mesh_obj = simplified_mesh
                        logger.info("Mesh simplification successful")
                    else:
                        logger.warning("Mesh simplification returned empty mesh, using original")
                except Exception as e:
                    logger.warning(f"Mesh simplification failed: {e}, using original mesh")
                
                _, _, face_reducer, _ = _ensure_postprocessing_loaded()
                mesh_obj = face_reducer(mesh_obj, max_facenum=kwargs['max_facenum'])
                self._cleanup_gpu_memory()

                if kwargs['image'] is not None:
                    input_image = Image.open(str(kwargs['image'])).convert("RGB")
                    if kwargs['remove_background']:
                        rmbg = _ensure_rembg_loaded()
                        input_image = rmbg(input_image)
                        self._cleanup_gpu_memory()
                else:
                    raise ValueError("To texture a mesh, an input image must be provided.")

                temp_mesh_path = "output/temp_mesh.obj"
                mesh_obj.export(temp_mesh_path)

                tex_pipeline = _ensure_texture_model_loaded()
                
                # AGGRESSIVE state cleanup for warmed-up GPU contamination issues
                _aggressive_gpu_state_reset(tex_pipeline)
                
                textured_mesh_path = tex_pipeline(
                    mesh_path=temp_mesh_path,
                    image_path=input_image,
                    output_mesh_path="output/textured_mesh.obj"
                )
                final_mesh = load_trimesh(textured_mesh_path)

            else:
                # Full pipeline mode
                if kwargs['image'] is None:
                    raise ValueError("Image must be provided if mesh is not.")

                metadata = self._process_single_image(
                    kwargs['image'], 
                    "output", 
                    0, 
                    **kwargs
                )
                
                if metadata['status'] != 'success':
                    raise RuntimeError(f"Failed to process image: {metadata.get('error', 'Unknown error')}")
                
                output_path = Path("output/mesh.glb")
                if not output_path.exists():
                    raise RuntimeError(f"Failed to generate mesh file at {output_path}")

                self._log_analytics_event("predict_completed", {"duration": time.time() - time.time()})
                return Output(mesh=output_path)

            output_path = Path("output/mesh.glb")
            final_mesh.export(str(output_path), include_normals=True)

            if not output_path.exists():
                raise RuntimeError(f"Failed to generate mesh file at {output_path}")

            self._log_analytics_event("predict_completed", {"duration": time.time() - time.time()})
            return Output(mesh=output_path)

        except Exception as e:
            self._log_analytics_event("predict_error", {"error": str(e)})
            raise

    def _preload_worker_models(self, num_workers: int):
        """Pre-load model instances for all workers with predictable IDs"""
        logger.info(f"🔧 Pre-loading models for {num_workers} workers...")
        
        # Pre-load models sequentially but assign to predictable worker IDs
        for worker_id in range(num_workers):
            worker_key = f"worker_{worker_id}"
            logger.info(f"  Loading models for {worker_key}...")
            
            with worker_models_lock:
                if worker_key not in worker_models:
                    # Initialize worker model storage
                    worker_models[worker_key] = {}
                    
                    # Load background removal model
                    logger.info(f"    Loading BackgroundRemover for {worker_key}...")
                    worker_models[worker_key]['rembg'] = BackgroundRemover()
                    
                    # Load shape generation model 
                    logger.info(f"    Loading Hunyuan3D shape model for {worker_key}...")
                    worker_models[worker_key]['shape'] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                        "tencent/Hunyuan3D-2.1"
                    )
                    
                    # Load texture generation model
                    logger.info(f"    Loading texture generation model for {worker_key}...")
                    max_num_view = 6
                    resolution = 512
                    tex_conf = Hunyuan3DPaintConfig(max_num_view, resolution)
                    tex_conf.realesrgan_ckpt_path = str(REALESRGAN_CKPT)
                    tex_conf.multiview_cfg_path = str(MULTIVIEW_CFG)
                    tex_conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
                    
                    worker_models[worker_key]['texture'] = Hunyuan3DPaintPipeline(tex_conf)
                    
                    # Load post-processing tools
                    logger.info(f"    Loading mesh processing tools for {worker_key}...")
                    worker_models[worker_key]['floater_remover'] = FloaterRemover()
                    worker_models[worker_key]['degenerate_remover'] = DegenerateFaceRemover()
                    worker_models[worker_key]['face_reducer'] = FaceReducer()
                    worker_models[worker_key]['mesh_simplifier'] = MeshSimplifier()
                    
                    logger.info(f"  ✅ {worker_key} models loaded")
        
        logger.info(f"✅ All {num_workers} worker model sets pre-loaded")

    def _cleanup_all_worker_models(self):
        """Clean up all pre-loaded worker model instances"""
        logger.info("🧹 Cleaning up all pre-loaded worker models...")
        with worker_models_lock:
            worker_keys = list(worker_models.keys())
            for worker_key in worker_keys:
                if worker_key.startswith('worker_'):
                    logger.info(f"  Cleaning up {worker_key} models...")
                    del worker_models[worker_key]
        logger.info("✅ All pre-loaded worker models cleaned up")



    def _process_image_worker(self, args_tuple):
        """
        Worker function for parallel processing.
        Acquires a worker from the queue, processes one image, and releases the worker.
        """
        image_path, output_dir, image_idx, worker_queue, kwargs = args_tuple
        
        worker_key = None
        try:
            # Get a worker from the queue (blocks until one is available)
            worker_key = worker_queue.get()
            
            # Process the image using the assigned worker
            return self._process_single_image_worker_direct(
                image_input=image_path,
                output_dir=output_dir,
                image_idx=image_idx,
                worker_key=worker_key,
                **kwargs
            )
        finally:
            # IMPORTANT: Return the worker to the queue so it can be reused
            if worker_key is not None:
                worker_queue.put(worker_key)

    def _process_batch_parallel(self, image_paths: List[str], parallel_workers: int, **kwargs) -> List[dict]:
        """Process a batch of images in parallel using a thread pool."""
        logger.info(f"🔄 Processing batch 1: images 1-{len(image_paths)}")
        vram_before = self.vram_monitor.get_available_vram()
        logger.info(f"💾 VRAM before batch: {vram_before:.1f}GB")

        # Use a thread pool to manage worker execution
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            
            # The worker_key is now managed by the queue, not assigned directly
            args_list = [(
                image_path, 
                kwargs.get('output_mesh_dir', os.path.join(self.output_dir, "meshes")),
                i,
                self.worker_queue, # Pass the queue to the worker
                kwargs
            ) for i, image_path in enumerate(image_paths)]
            
            # Map jobs to threads and collect results
            results = list(tqdm(executor.map(self._process_image_worker, args_list), 
                                total=len(image_paths), desc="Batch Processing"))
        
        return results

    def _initialize_worker_pool(self, num_workers: int):
        """Initializes the stateful worker pool for thread-safe processing."""
        logger.info(f"  🏭 Initializing stateful worker pool with {num_workers} workers...")
        self.worker_locks = {f"worker_{i}": threading.Lock() for i in range(num_workers)}
        self.worker_queue = queue.Queue()
        for i in range(num_workers):
            self.worker_queue.put(f"worker_{i}")
        logger.info("  ✅ Worker pool initialized.")

    def _predict_batch(self, batch_images: Path, parallel_workers: int = 2, **kwargs) -> Output:
        """
        Process a batch of images from a ZIP file.
        This is the main entry point for batch prediction.
        """
        # Determine number of workers
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            parallel_workers = 1
        
        self.active_workers = parallel_workers
        
        # Initialize the stateful worker pool
        self._initialize_worker_pool(self.active_workers)
        
        results_dir = tempfile.mkdtemp()
        meshes_dir = os.path.join(results_dir, "meshes")
        os.makedirs(meshes_dir, exist_ok=True)
        
        # Pre-load models in parallel for all workers
        if self.active_workers > 1:
            self._preload_worker_models(self.active_workers)
        
        image_paths = extract_zip_images(batch_images, self.output_dir)
        num_images = len(image_paths)
        logger.info(f"🚀 Starting batch processing: {num_images} images")
        
        # Log VRAM and processing mode
        self.vram_monitor.get_available_vram()
        if self.active_workers > 1:
            logger.info(f"🔥 Using PARALLEL processing with {self.active_workers} workers")
        else:
            logger.info(f"🔥 Using SEQUENTIAL processing (CPU or single GPU)")
        
        start_time = time.time()
        
        # Choose processing strategy based on number of workers
        if self.active_workers > 1:
            all_results = self._process_batch_parallel(image_paths, self.active_workers, **kwargs)
        else:
            all_results = self._process_batch_sequential(image_paths, **kwargs)
        
        end_time = time.time()
        
        # Final cleanup of pre-loaded models
        if self.active_workers > 1:
            self._cleanup_all_worker_models()
        
        total_time = end_time - start_time
        
        # Create final results zip
        results_json_path = os.path.join(results_dir, "results.json")
        with open(results_json_path, "w") as f:
            json.dump(all_results, f, indent=2)
            
        successful_predictions = [res for res in all_results if res['status'] == 'success']
        output_zip_path = os.path.join(self.output_dir, "batch_results.zip")
        create_batch_zip(meshes_dir, results_json_path, output_zip_path)

        # Log analytics
        success_rate = (len(successful_predictions) / num_images) * 100 if num_images > 0 else 0
        analytics_data = {
            'total_images': num_images,
            'successful': len(successful_predictions),
            'failed': num_images - len(successful_predictions),
            'success_rate_percent': round(success_rate, 1),
            'total_time_minutes': round(total_time / 60, 1)
        }
        self._log_analytics_event("batch_predict_completed", analytics_data)
        
        logger.info("\n🏁 Batch processing completed!")
        logger.info(f"📊 Results: {len(successful_predictions)}/{num_images} successful ({success_rate:.1f}%)")
        logger.info(f"⏱️  Total time: {total_time / 60:.1f} minutes")
        
        return Output(batch_results=Path(output_zip_path), mesh=None)

    def _process_batch_sequential(self, image_paths: List[str], **kwargs) -> List[dict]:
        """Process images sequentially (fallback method)"""
        import pathlib
        output_mesh_dir = kwargs.pop('output_mesh_dir', None)
        if output_mesh_dir is None:
            output_mesh_dir = str(pathlib.Path("output/meshes").resolve())
        else:
            output_mesh_dir = str(pathlib.Path(output_mesh_dir).resolve())
        
        batch_results = []
        
        for idx, image_path in enumerate(image_paths):
            logger.info(f"\n📸 Processing image {idx + 1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Pre-processing safety check
            current_vram = self.vram_monitor.get_available_vram()
            logger.info(f"  VRAM before image {idx + 1}: {current_vram:.1f}GB available")
            
            if not self._check_memory_safety():
                logger.error(f"Insufficient VRAM for image {idx + 1}, skipping remaining images")
                
                # Add error entries for remaining images
                for remaining_idx in range(idx, len(image_paths)):
                    remaining_path = image_paths[remaining_idx]
                    image_name = os.path.splitext(os.path.basename(remaining_path))[0]
                    error_metadata = {
                        "input_image": image_name,
                        "output_mesh": None,
                        "status": "error",
                        "duration": 0.0,
                        "face_count": 0,
                        "vertex_count": 0,
                        "error": "Insufficient VRAM",
                        "error_type": "RuntimeError"
                    }
                    batch_results.append(error_metadata)
                break
            
            # Process single image
            metadata = self._process_single_image(
                image_path,
                output_mesh_dir,
                idx,
                **kwargs
            )
            batch_results.append(metadata)
            
            # Cleanup between images
            self._cleanup_gpu_memory()
        
        return batch_results 