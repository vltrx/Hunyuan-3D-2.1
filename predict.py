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

import torch
from PIL import Image
from torch import cuda, Generator
from cog import BasePredictor, BaseModel, Input, Path

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
        tex_conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        tex_conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        tex_conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"

        # Fallback: Download RealESRGAN model if missing
        if not os.path.exists(tex_conf.realesrgan_ckpt_path):
            logger.info("RealESRGAN model not found, downloading...")
            os.makedirs(os.path.dirname(tex_conf.realesrgan_ckpt_path), exist_ok=True)
            subprocess.run([
                "wget", 
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "-O", tex_conf.realesrgan_ckpt_path
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
            tex_conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
            tex_conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
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

class VRAMMonitor:
    def __init__(self):
        self._lock = threading.Lock()
    
    def get_available_vram(self) -> float:
        """Get available VRAM in GB (thread-safe)"""
        with self._lock:
        if not cuda.is_available():
            return 0.0
        
        total = cuda.get_device_properties(0).total_memory / 1024**3
        allocated = cuda.memory_allocated(0) / 1024**3
        return total - allocated
    
    def get_used_vram(self) -> float:
        """Get used VRAM in GB (thread-safe)"""
        with self._lock:
        if not cuda.is_available():
            return 0.0
        return cuda.memory_allocated(0) / 1024**3
    
    def check_parallel_safety(self, required_per_worker: float, num_workers: int) -> bool:
        """Check if we have enough VRAM for parallel workers"""
        available = self.get_available_vram()
        total_required = required_per_worker * num_workers
        safety_buffer = 4.0  # 4GB safety buffer
        
        return available >= (total_required + safety_buffer)

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
            worker_models = _ensure_worker_models_loaded()
            
            # Apply post-processing pipeline using worker models
            mesh_output = worker_models['floater_remover'](mesh)
            if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                raise RuntimeError("Mesh became empty after floater removal")
                
            mesh_output = worker_models['degenerate_remover'](mesh_output)
            if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                raise RuntimeError("Mesh became empty after degenerate face removal")
            
            # Face reduction (always needed)
            mesh_output = worker_models['face_reducer'](mesh_output, max_facenum=kwargs.get('max_facenum', 40000))
            if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                raise RuntimeError("Mesh became empty after face reduction")
                
            self._cleanup_gpu_memory()

            # Save intermediate mesh
            temp_mesh_path = os.path.join(output_dir, f"{image_name}_temp.obj")
            mesh_output.export(temp_mesh_path)

            # Apply texturing with worker-specific model
            logger.info(f"  [Worker-{thread_id}] Generating texture for {image_name}")
            tex_pipeline = worker_models['texture']
            textured_mesh_path = tex_pipeline(
                mesh_path=temp_mesh_path,
                image_path=input_image,
                output_mesh_path=os.path.join(output_dir, f"{image_name}_textured.obj")
            )

            # Export final GLB
            from trimesh import load as load_trimesh
            final_mesh = load_trimesh(textured_mesh_path)
            output_path = os.path.join(output_dir, f"{image_name}.glb")
            final_mesh.export(output_path, include_normals=True)

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
            
            # Use worker-specific shape generation (no locking!)
            generator = torch.Generator()
            generator = generator.manual_seed(int(kwargs.get('seed', 1234)) + image_idx)
            
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

            # MEMORY-AWARE texture generation with worker-specific file paths
            available_vram = self.vram_monitor.get_available_vram()
            texture_memory_estimate = 18.0  # GB - conservative estimate for texture generation
            
            # Create worker-specific temp mesh path to prevent file conflicts
            worker_temp_mesh_path = os.path.join(output_dir, f"{image_name}_worker_{thread_id}_temp.obj")
            
            # Copy the mesh to worker-specific path to prevent conflicts
            import shutil
            shutil.copy2(temp_mesh_path, worker_temp_mesh_path)
            
            if available_vram > texture_memory_estimate + 5.0:  # 5GB safety buffer
                # Sufficient VRAM for parallel texture generation
                logger.info(f"  [Worker-{thread_id}] Generating texture for {image_name} (PARALLEL - {available_vram:.1f}GB available)")
                textured_mesh_path = worker_models_set['texture'](
                    mesh_path=worker_temp_mesh_path,
                    image_path=input_image,
                    output_mesh_path=os.path.join(output_dir, f"{image_name}_worker_{thread_id}_textured.obj")
                )
                logger.info(f"  [Worker-{thread_id}] Completed PARALLEL texture generation for {image_name}")
            else:
                # Insufficient VRAM - serialize texture generation for safety
                logger.info(f"  [Worker-{thread_id}] Waiting for texture generation slot (VRAM: {available_vram:.1f}GB)")
                with texture_generation_lock:
                    logger.info(f"  [Worker-{thread_id}] Generating texture for {image_name} (SERIALIZED for VRAM safety)")
                    textured_mesh_path = worker_models_set['texture'](
                        mesh_path=worker_temp_mesh_path,
                        image_path=input_image,
                        output_mesh_path=os.path.join(output_dir, f"{image_name}_worker_{thread_id}_textured.obj")
                    )
                    logger.info(f"  [Worker-{thread_id}] Completed SERIALIZED texture generation for {image_name}")

            # Clean up worker-specific temp file
            try:
                os.remove(worker_temp_mesh_path)
            except:
                pass  # Don't fail if cleanup fails

            # Export final GLB
            from trimesh import load as load_trimesh
            final_mesh = load_trimesh(textured_mesh_path)
            output_path = os.path.join(output_dir, f"{image_name}.glb")
            final_mesh.export(output_path, include_normals=True)

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
            textured_mesh_path = tex_pipeline(
                mesh_path=temp_mesh_path,
                image_path=input_image,
                output_mesh_path=os.path.join(output_dir, f"{image_name}_textured.obj")
            )

            # Export final GLB
            from trimesh import load as load_trimesh
            final_mesh = load_trimesh(textured_mesh_path)
            output_path = os.path.join(output_dir, f"{image_name}.glb")
            final_mesh.export(output_path, include_normals=True)

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
        
        start_time = time.time()
        
        # Analytics
        self._log_analytics_event("predict_started")

        # Determine processing mode based on inputs
        if batch_images:
            return self._predict_batch(
                batch_images=batch_images,
                steps=steps,
                guidance_scale=guidance_scale,
                max_facenum=max_facenum,
                num_chunks=num_chunks,
                seed=seed,
                octree_resolution=octree_resolution,
                remove_background=remove_background,
                prompt=prompt,
                parallel_workers=parallel_workers
            )
        else:
            return self._predict_single(
                image=image,
                mesh=mesh,
                steps=steps,
                guidance_scale=guidance_scale,
                max_facenum=max_facenum,
                num_chunks=num_chunks,
                seed=seed,
                octree_resolution=octree_resolution,
                remove_background=remove_background,
                prompt=prompt
            )

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
                    tex_conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
                    tex_conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
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
        """Worker function for parallel image processing with pre-loaded models"""
        idx, image_path, output_dir, kwargs, worker_id = args_tuple
        
        # Each worker gets its own thread ID for logging
        thread_id = threading.current_thread().ident
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        worker_key = f"worker_{worker_id}"
        
        logger.info(f"🔄 [Worker-{thread_id}] Processing {image_name} ({idx + 1}) using {worker_key} models")
        
        try:
            # Pre-processing VRAM check for this worker
            current_vram = self.vram_monitor.get_available_vram()
            logger.info(f"  [Worker-{thread_id}] VRAM: {current_vram:.1f}GB available")
            
            if current_vram < 25.0:  # Require 25GB per worker for dedicated models
                raise RuntimeError(f"Insufficient VRAM for worker: {current_vram:.1f}GB available")
            
            # Verify pre-loaded models are available
            if worker_key not in worker_models:
                raise RuntimeError(f"Pre-loaded models not found for {worker_key}")
            
            # Process the image using pre-loaded worker-specific models
            metadata = self._process_single_image_worker_direct(
                image_path,
                output_dir,
                idx,
                worker_key,
                **kwargs
            )
            
            logger.info(f"✅ [Worker-{thread_id}] Completed {image_name}")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ [Worker-{thread_id}] Failed {image_name}: {str(e)}")
            
            # Return error metadata
            return {
                "input_image": image_name,
                "output_mesh": None,
                "status": "error",
                "duration": 0.0,
                "face_count": 0,
                "vertex_count": 0,
                "error": str(e),
                "error_type": type(e).__name__
            }
        finally:
            # Always cleanup after worker
            self._cleanup_gpu_memory()
            # Don't cleanup worker models here - they're shared/pre-loaded

    def _predict_batch(self, batch_images: Path, parallel_workers: int = 2, **kwargs) -> Output:
        """
        Batch processing mode - extract ZIP, process images, create results
        """
        batch_start_time = time.time()
        
        self._log_analytics_event("predict_mode", {"mode": "batch"})
        
        # Setup output directory structure
        if os.path.exists("output"):
            shutil.rmtree("output")
        os.makedirs("output", exist_ok=True)
        os.makedirs("output/meshes", exist_ok=True)
        
        # Extract images from ZIP
        logger.info("Extracting images from ZIP file...")
        extract_dir = "output/extracted"
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            image_paths = extract_zip_images(batch_images, extract_dir)
            logger.info(f"Extracted {len(image_paths)} images from ZIP")
        except Exception as e:
            raise ValueError(f"Failed to extract ZIP file: {str(e)}")
        
        if len(image_paths) == 0:
            raise ValueError("No valid images found in ZIP file")

        logger.info(f"🚀 Starting batch processing: {len(image_paths)} images")
        logger.info(f"💾 Available VRAM: {self.vram_monitor.get_available_vram():.1f}GB")
        
        # Determine if we can use parallel processing (higher VRAM requirement for dedicated models)
        vram_per_worker = 30.0  # Estimated VRAM per worker with dedicated models (GB)
        can_parallel = (
            parallel_workers > 1 and 
            len(image_paths) > 1 and
            self.vram_monitor.check_parallel_safety(vram_per_worker, parallel_workers)
        )
        
        if can_parallel:
            logger.info(f"🔥 Using PARALLEL processing with {parallel_workers} workers")
            batch_results = self._process_batch_parallel(image_paths, parallel_workers, **kwargs)
        else:
            if parallel_workers > 1:
                logger.info(f"⚠️  Falling back to SEQUENTIAL processing (insufficient VRAM for {parallel_workers} workers)")
            else:
                logger.info(f"📋 Using SEQUENTIAL processing")
            batch_results = self._process_batch_sequential(image_paths, **kwargs)
        
        # Post-process results to update output mesh paths
        for metadata in batch_results:
            if metadata['status'] == 'success':
                metadata['output_mesh'] = f"{metadata['input_image']}.glb"
            else:
                metadata['output_mesh'] = None
        
        # Generate batch results JSON
        results_json_path = "output/batch_results.json"
        with open(results_json_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        # Create batch ZIP file
        batch_zip_path = "output/batch_meshes.zip"
        create_batch_zip("output/meshes", results_json_path, batch_zip_path)
        
        # Final statistics
        total_time = time.time() - batch_start_time
        successful_count = len([r for r in batch_results if r['status'] == 'success'])
        success_rate = successful_count / len(image_paths) * 100
        
        logger.info(f"\n🏁 Batch processing completed!")
        logger.info(f"📊 Results: {successful_count}/{len(image_paths)} successful ({success_rate:.1f}%)")
        logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
        
        self._log_analytics_event("batch_predict_completed", {
            "total_images": len(image_paths),
            "successful": successful_count,
            "failed": len(image_paths) - successful_count,
            "success_rate_percent": round(success_rate, 1),
            "total_time_minutes": round(total_time / 60, 1)
        })
        
        return Output(
            mesh=Path(batch_zip_path),
            batch_results=Path(results_json_path)
        )
    
    def _process_batch_parallel(self, image_paths: List[str], parallel_workers: int, **kwargs) -> List[dict]:
        """Process images in parallel using ThreadPoolExecutor with dedicated model instances"""
        
        # Pre-load models for all workers before parallel execution
        logger.info(f"🔧 Pre-loading dedicated models for {parallel_workers} workers...")
        self._preload_worker_models(parallel_workers)
        logger.info(f"✅ All {parallel_workers} worker model sets pre-loaded")
        
        # Prepare arguments for workers (include worker_id for model assignment)
        worker_args = [
            (idx, image_path, "output/meshes", kwargs, idx % parallel_workers)
            for idx, image_path in enumerate(image_paths)
        ]
        
        batch_results = []
        
        # Process in batches to manage memory better
        batch_size = parallel_workers * 2  # Process 2 batches per worker set
        
        for batch_start in range(0, len(worker_args), batch_size):
            batch_end = min(batch_start + batch_size, len(worker_args))
            current_batch = worker_args[batch_start:batch_end]
            
            logger.info(f"🔄 Processing batch {batch_start//batch_size + 1}: images {batch_start + 1}-{batch_end}")
            
            # Check VRAM before each batch
            current_vram = self.vram_monitor.get_available_vram()
            logger.info(f"💾 VRAM before batch: {current_vram:.1f}GB")
            
            if current_vram < 40.0:  # Need substantial VRAM for parallel processing with dedicated models
                logger.warning("⚠️  Low VRAM detected, falling back to sequential for remaining images")
                
                # Process remaining images sequentially
                remaining_args = worker_args[batch_start:]
                for idx, image_path, output_dir, kwargs_dict, _ in remaining_args:
                    metadata = self._process_single_image(image_path, output_dir, idx, **kwargs_dict)
                    batch_results.append(metadata)
                break
            
            # Execute parallel batch with pre-loaded models
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                # Submit all jobs in current batch
                future_to_args = {
                    executor.submit(self._process_image_worker, args): args 
                    for args in current_batch
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_args):
                    try:
                        result = future.result(timeout=600)  # 10 minute timeout per image
                        batch_results.append(result)
                    except Exception as e:
                        # Handle worker failure
                        args = future_to_args[future]
                        _, image_path, _, _, _ = args
                        image_name = os.path.splitext(os.path.basename(image_path))[0]
                        
                        error_metadata = {
                            "input_image": image_name,
                            "output_mesh": None,
                            "status": "error",
                            "duration": 0.0,
                            "face_count": 0,
                            "vertex_count": 0,
                            "error": f"Worker timeout or error: {str(e)}",
                            "error_type": type(e).__name__
                        }
                        batch_results.append(error_metadata)
                        logger.error(f"❌ Worker failed for {image_name}: {str(e)}")
            
            # Cleanup between batches
            self._cleanup_gpu_memory()
            time.sleep(1)  # Brief pause between batches
        
        # Cleanup all pre-loaded worker models
        self._cleanup_all_worker_models()
        
        return batch_results
    
    def _process_batch_sequential(self, image_paths: List[str], **kwargs) -> List[dict]:
        """Process images sequentially (fallback method)"""
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
                "output/meshes",
                idx,
                **kwargs
            )
            batch_results.append(metadata)
            
            # Cleanup between images
            self._cleanup_gpu_memory()
        
        return batch_results 