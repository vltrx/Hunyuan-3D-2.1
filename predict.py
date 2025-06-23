import os
import shutil
import subprocess
import sys
import time
import json
import traceback
import io
import requests
from typing import List, Optional, Union
import gc
import numpy as np

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
    
    # Ensure CUDA toolkit is available
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
    print(f"PATH: {os.environ.get('PATH')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    print(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST')}")

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

def initialize_models():
    """Initialize models lazily (HF-style but on-demand)"""
    global rmbg_worker, i23d_worker, tex_pipeline
    global floater_remove_worker, degenerate_face_remove_worker, face_reduce_worker, mesh_simplifier
    
    if rmbg_worker is not None:
        return  # Already initialized
    
    print("Initializing models at runtime (HF-style)...")

    # Initialize background removal worker
    print("Loading background removal model...")
    rmbg_worker = BackgroundRemover()

    # Initialize shape generation model  
    print("Loading shape generation model...")
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1"
    )

    # Initialize texture generation model
    print("Loading texture generation model...")
    max_num_view = 6
    resolution = 512
    tex_conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    tex_conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    tex_conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    tex_conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"

    # Fallback: Download RealESRGAN model if missing
    if not os.path.exists(tex_conf.realesrgan_ckpt_path):
        print("RealESRGAN model not found, downloading as fallback...")
        os.makedirs(os.path.dirname(tex_conf.realesrgan_ckpt_path), exist_ok=True)
        subprocess.run([
            "wget", 
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "-O", tex_conf.realesrgan_ckpt_path
        ], check=True)
        print("RealESRGAN model downloaded successfully")

    tex_pipeline = Hunyuan3DPaintPipeline(tex_conf)

    # Initialize mesh processing workers
    print("Loading mesh processing tools...")
    floater_remove_worker = FloaterRemover()
    degenerate_face_remove_worker = DegenerateFaceRemover()
    face_reduce_worker = FaceReducer()
    mesh_simplifier = MeshSimplifier()

    print("All models initialized successfully (HF-style)")

# Constants
CHECKPOINTS_PATH = "/src/checkpoints"
HUNYUAN3D_MODEL_PATH = "tencent/Hunyuan3D-2.1"
U2NET_PATH = os.path.join(CHECKPOINTS_PATH, ".u2net/")
U2NET_URL = "https://weights.replicate.delivery/default/comfy-ui/rembg/u2net.onnx.tar"
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

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

class Output(BaseModel):
    # Single mode outputs
    mesh: Optional[Path] = None
    
    # Batch mode outputs (when batch_mode=True)
    meshes: Optional[List[Path]] = None
    failed_images: Optional[List[str]] = None
    processing_stats: Optional[dict] = None

class VRAMMonitor:
    """Monitor VRAM usage for batch processing safety"""
    
    @staticmethod
    def get_available_vram() -> float:
        """Get available VRAM in GB"""
        if not cuda.is_available():
            return 0.0
        
        total = cuda.get_device_properties(0).total_memory / 1024**3
        allocated = cuda.memory_allocated(0) / 1024**3
        return total - allocated
    
    @staticmethod
    def get_used_vram() -> float:
        """Get used VRAM in GB"""
        if not cuda.is_available():
            return 0.0
        return cuda.memory_allocated(0) / 1024**3

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        logger.info("Setup started")
        
        # Lazy model initialization (HF-style but on-demand)
        initialize_models()
        
        # Use the now-initialized global workers
        logger.info("Using initialized global workers (HF-style)")
        self.rmbg_worker = rmbg_worker
        self.i23d_worker = i23d_worker
        self.tex_pipeline = tex_pipeline
        self.floater_remove_worker = floater_remove_worker
        self.degenerate_face_remove_worker = degenerate_face_remove_worker
        self.face_reduce_worker = face_reduce_worker
        self.mesh_simplifier = mesh_simplifier
        
        # Initialize VRAM monitor
        self.vram_monitor = VRAMMonitor()
        
        # Initial GPU memory cleanup
        self._cleanup_gpu_memory()
        
        logger.info("Setup completed using HF-style lazy-loaded workers")
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory between predictions"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            
    # HF-style shape generation function (mimicking their exact pattern)
    def _hf_style_gen_shape(self, image, steps=50, guidance_scale=5.0, seed=1234, 
                           octree_resolution=512, num_chunks=8000):
        """Generate shape using HF Space pattern with module-level workers"""
        
        logger.info(f"HF-style shape generation: steps={steps}, guidance_scale={guidance_scale}")
        
        # Use module-level worker directly (HF pattern)
        generator = torch.Generator()
        generator = generator.manual_seed(int(seed))
        
        # Direct call to module-level worker (exactly like HF Space)
        outputs = i23d_worker(
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

    def _check_memory_safety(self, min_required_gb: float = 30.0) -> bool:
        """Check if we have enough memory to proceed safely"""
        available = self.vram_monitor.get_available_vram()
        if available < min_required_gb:
            logger.warning(f"Low VRAM warning: {available:.1f}GB available, {min_required_gb}GB required")
            self._cleanup_gpu_memory()
            available = self.vram_monitor.get_available_vram()
            return available >= min_required_gb
        return True

    def _log_analytics_event(self, event_name, params=None):
        # In production, you might want to log to analytics service
        logger.info(f"Analytics: {event_name} - {params}")

    def _process_single_image(self, 
                            image_input: Union[Path, Image.Image], 
                            output_dir: str,
                            image_idx: int,
                            **kwargs) -> Optional[Path]:
        """
        Process a single image with comprehensive error handling
        """
        start_time = time.time()
        
        try:
            # Memory safety check
            if not self._check_memory_safety():
                raise RuntimeError(f"Insufficient VRAM available ({self.vram_monitor.get_available_vram():.1f}GB)")

            # Ensure models are initialized (safety check)
            if rmbg_worker is None:
                logger.info("Models not initialized, initializing now...")
                initialize_models()

            # Load and preprocess image
            if isinstance(image_input, Path):
                input_image = Image.open(str(image_input)).convert("RGB")
                image_name = os.path.splitext(os.path.basename(image_input))[0]
            else:
                input_image = image_input
                image_name = f"image_{image_idx}"

            # Background removal
            logger.info(f"  Removing background for {image_name}")
            processed_image = rmbg_worker(input_image)  # Use module-level worker directly
            
            # Log transparency info
            if hasattr(processed_image, 'getchannel'):
                alpha = processed_image.getchannel('A')
                alpha_array = np.array(alpha)
                opaque_pixels = np.sum(alpha_array > 0)
                total_pixels = alpha_array.size
                transparency_ratio = opaque_pixels / total_pixels
                logger.info(f"  Image transparency: {transparency_ratio*100:.2f}% opaque pixels")
                
                # IMMEDIATE DEBUG: Test if we can reach this line
                logger.info("  DEBUG: Line immediately after transparency logging reached")
                
                # IMMEDIATE DEBUG: Test variable cleanup
                del alpha, alpha_array, opaque_pixels, total_pixels, transparency_ratio
                logger.info("  DEBUG: Variables cleaned up successfully")
                
                # Force garbage collection to prevent memory issues
                import gc
                gc.collect()
                logger.info("  DEBUG: Garbage collection completed")
                
                # Clear any potential GPU memory issues
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("  DEBUG: CUDA cache cleared")
            else:
                logger.info("  DEBUG: No alpha channel found, skipping transparency check")
            
            # IMMEDIATE DEBUG: Test if we can reach shape generation section
            logger.info("  DEBUG: About to start shape generation section")
            
            # DEBUG: Add checkpoint logs to identify where process hangs
            logger.info("  DEBUG: About to start shape generation timing")
            
            # Start timing shape generation
            shape_start_time = time.time()
            logger.info(f"  Beginning shape generation at {time.strftime('%H:%M:%S')}")
            
            # DEBUG: Log before calling HF-style generation
            logger.info("  DEBUG: About to call HF-style shape generation")
            
            # Use HF-style shape generation (exactly like HF Space)
            outputs = self._hf_style_gen_shape(
                processed_image, 
                kwargs.get('steps', 50),
                kwargs.get('guidance_scale', 5.0), 
                kwargs.get('seed', 1234) + image_idx,
                kwargs.get('octree_resolution', 512),
                kwargs.get('num_chunks', 8000)
            )
            
            # DEBUG: Log after HF-style generation returns
            logger.info("  DEBUG: HF-style shape generation returned")
            
            shape_total_time = time.time() - shape_start_time
            logger.info(f"  Total shape pipeline time: {shape_total_time:.1f} seconds")
            
            # Clean up GPU memory after generation (HF-style)
            self._cleanup_gpu_memory()
            
            # Check if mesh generation was successful
            if outputs is None:
                raise RuntimeError("Mesh generation returned None - surface extraction failed")
            
            # Add detailed mesh diagnostics
            logger.info(f"  Generated mesh - Vertices: {len(outputs[0].vertices)}, Faces: {len(outputs[0].faces)}")
            if hasattr(outputs[0], 'bounds'):
                bounds = outputs[0].bounds
                size = bounds[1] - bounds[0]
                logger.info(f"  Mesh bounds: {bounds}, Size: {size}")
                logger.info(f"  Mesh volume: {outputs[0].volume:.6f}")
                
                # Check if mesh is degenerate (very flat) and retry if needed
                min_dimension = min(size)
                max_dimension = max(size)
                dimension_ratio = min_dimension / max_dimension if max_dimension > 0 else 0
                
                # More sensitive detection for flat geometry
                is_flat = (
                    min_dimension < 0.1 or  # Increased threshold
                    outputs[0].volume < 0.01 or  # Increased threshold  
                    dimension_ratio < 0.15  # New: detect when one dimension is much smaller
                )
                
                if is_flat:
                    logger.warning(f"  ⚠️  Detected flat mesh - min dimension: {min_dimension:.6f}, volume: {outputs[0].volume:.6f}, ratio: {dimension_ratio:.3f}")
                    logger.info(f"  🔄 Retrying with enhanced parameters...")
                    
                    # Retry with parameters that help generate more 3D geometry
                    try:
                        retry_outputs = self._hf_style_gen_shape(
                            processed_image,
                            min(kwargs.get('steps', 50) + 10, 60),  # More steps
                            min(kwargs.get('guidance_scale', 5.0) + 3.0, 12.0),  # Much higher guidance
                            kwargs.get('seed', 1234) + image_idx + 999,  # Different seed
                            min(kwargs.get('octree_resolution', 512) + 128, 640),  # Higher resolution
                            max(kwargs.get('num_chunks', 8000) - 2000, 6000)  # Fewer chunks for more detail
                        )
                        retry_mesh = retry_outputs[0]
                        
                        if retry_mesh is not None and hasattr(retry_mesh, 'bounds'):
                            retry_bounds = retry_mesh.bounds
                            retry_size = retry_bounds[1] - retry_bounds[0]
                            retry_min_dim = min(retry_size)
                            retry_max_dim = max(retry_size)
                            retry_ratio = retry_min_dim / retry_max_dim if retry_max_dim > 0 else 0
                            retry_volume = retry_mesh.volume
                            
                            # Check if retry improved the geometry (focus on dimension ratio and volume)
                            improved_ratio = retry_ratio > dimension_ratio * 1.5  # At least 50% better ratio
                            improved_volume = retry_volume > outputs[0].volume * 2.0  # At least 2x volume
                            improved_min_dim = retry_min_dim > min_dimension * 1.5  # At least 50% thicker
                            
                            if improved_ratio or improved_volume or improved_min_dim:
                                logger.info(f"  ✅ Retry improved geometry - ratio: {retry_ratio:.3f} (was {dimension_ratio:.3f}), volume: {retry_volume:.6f} (was {outputs[0].volume:.6f})")
                                outputs[0] = retry_mesh
                            else:
                                logger.info(f"  ⚠️  Retry didn't improve significantly - ratio: {retry_ratio:.3f}, volume: {retry_volume:.6f}, keeping original")
                        else:
                            logger.info(f"  ⚠️  Retry failed, keeping original mesh")
                            
                    except Exception as retry_error:
                        logger.warning(f"  ⚠️  Retry failed with error: {str(retry_error)}, keeping original")
                
            else:
                logger.info("  Mesh bounds info not available")
            
            # Post-process mesh
            logger.info(f"  Post-processing mesh for {image_name}")
            
            # Basic cleanup using module-level workers (HF pattern)
            mesh_output = floater_remove_worker(outputs[0])
            if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                raise RuntimeError("Mesh became empty after floater removal")
                
            mesh_output = degenerate_face_remove_worker(mesh_output)
            if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                raise RuntimeError("Mesh became empty after degenerate face removal")
            
            # Optional mesh simplification (skip if binary missing)
            mesh_before_simplify = mesh_output
            try:
                simplified_mesh = mesh_simplifier(mesh_output)
                if simplified_mesh is None or len(simplified_mesh.vertices) == 0 or len(simplified_mesh.faces) == 0:
                    logger.warning("  Mesh simplifier returned empty mesh, skipping simplification")
                    mesh_output = mesh_before_simplify  # Keep original mesh
                else:
                    logger.info(f"  Mesh simplified successfully")
                    mesh_output = simplified_mesh
            except Exception as e:
                logger.warning(f"  Mesh simplification failed ({str(e)}), continuing without simplification")
                mesh_output = mesh_before_simplify  # Keep original mesh
            
            # Face reduction (always needed)
            mesh_output = face_reduce_worker(mesh_output, max_facenum=kwargs.get('max_facenum', 40000))
            if mesh_output is None or len(mesh_output.vertices) == 0 or len(mesh_output.faces) == 0:
                raise RuntimeError("Mesh became empty after face reduction")
                
            self._cleanup_gpu_memory()

            # Save intermediate mesh
            temp_mesh_path = os.path.join(output_dir, f"{image_name}_temp.obj")
            mesh_output.export(temp_mesh_path)

            # Apply texturing using module-level worker (HF pattern)
            logger.info(f"  Generating texture for {image_name}")
            textured_mesh_path = tex_pipeline(
                mesh_path=temp_mesh_path,
                image_path=input_image,
                output_mesh_path=os.path.join(output_dir, f"{image_name}_textured.obj")
            )

            # Export final GLB
            from trimesh import load as load_trimesh
            final_mesh = load_trimesh(textured_mesh_path)
            output_path = Path(os.path.join(output_dir, f"{image_name}.glb"))
            final_mesh.export(str(output_path), include_normals=True)

            # Cleanup intermediate files
            try:
                os.remove(temp_mesh_path)
                if os.path.exists(textured_mesh_path):
                    os.remove(textured_mesh_path)
            except:
                pass  # Don't fail if cleanup fails

            processing_time = time.time() - start_time
            logger.info(f"  ✅ {image_name} completed in {processing_time:.1f}s, faces: {len(final_mesh.faces)}")
            
            return output_path

        except Exception as e:
            error_msg = f"Failed to process image {image_idx}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return None
        finally:
            self._cleanup_gpu_memory()

    def predict(
        self,
        image: Path = Input(
            description="Input image for generating 3D shape (single mode)",
            default=None,
        ),
        images: str = Input(
            description="Comma-separated URLs or base64 images for batch processing",
            default=None,
        ),
        mesh: Path = Input(
            description="Optional: Upload a .glb mesh to skip generation and only texture it",
            default=None,
        ),
        prompt: str = Input(
            description="Text prompt to guide texture generation",
            default="a detailed texture",
        ),
        steps: int = Input(
            description="Number of inference steps",
            default=50,
            ge=20,
            le=50,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for generation",
            default=5.0,
            ge=1.0,
            le=20.0,
        ),
        max_facenum: int = Input(
            description="Maximum number of faces for mesh generation",
            default=40000,
            ge=10000,
            le=200000,
        ),
        num_chunks: int = Input(
            description="Number of chunks for mesh generation",
            default=8000,
            ge=1000,
            le=200000,
        ),
        seed: int = Input(
            description="Random seed for generation",
            default=1234,
        ),
        octree_resolution: int = Input(
            description="Octree resolution for mesh generation",
            choices=[256, 384, 512],
            default=512,
        ),
        remove_background: bool = Input(
            description="Whether to remove background from input image",
            default=True,
        ),
        batch_mode: bool = Input(
            description="Enable batch processing mode",
            default=False,
        ),
        max_batch_size: int = Input(
            description="Maximum number of images to process in batch mode",
            default=10,
            ge=1,
            le=25,
        ),
    ) -> Output:
        
        start_time = time.time()

        # Determine processing mode
        if batch_mode and images:
            return self._predict_batch(
                images=images,
                steps=steps,
                guidance_scale=guidance_scale,
                max_facenum=max_facenum,
                num_chunks=num_chunks,
                seed=seed,
                octree_resolution=octree_resolution,
                remove_background=remove_background,
                max_batch_size=max_batch_size,
                prompt=prompt
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
        """Original single image processing logic"""
        from trimesh import load as load_trimesh

        self._log_analytics_event("predict_started", kwargs)

        if os.path.exists("output"):
            shutil.rmtree("output")
        os.makedirs("output", exist_ok=True)

        generator = Generator().manual_seed(kwargs['seed'])

        try:
            if kwargs['mesh']:
                # Texture-only mode
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
                    simplified_mesh = self.mesh_simplifier(mesh_obj)
                    if simplified_mesh is not None and len(simplified_mesh.vertices) > 0 and len(simplified_mesh.faces) > 0:
                        mesh_obj = simplified_mesh
                        logger.info("Mesh simplification successful")
                    else:
                        logger.warning("Mesh simplification returned empty mesh, using original")
                except Exception as e:
                    logger.warning(f"Mesh simplification failed: {e}, using original mesh")
                
                mesh_obj = self.face_reduce_worker(mesh_obj, max_facenum=kwargs['max_facenum'])
                self._cleanup_gpu_memory()

                if kwargs['image'] is not None:
                    input_image = Image.open(str(kwargs['image'])).convert("RGB")
                    if kwargs['remove_background']:
                        input_image = self.rmbg_worker(input_image)
                        self._cleanup_gpu_memory()
                else:
                    raise ValueError("To texture a mesh, an input image must be provided.")

                temp_mesh_path = "output/temp_mesh.obj"
                mesh_obj.export(temp_mesh_path)

                textured_mesh_path = self.tex_pipeline(
                    mesh_path=temp_mesh_path,
                    image_path=input_image,
                    output_mesh_path="output/textured_mesh.obj"
                )
                final_mesh = load_trimesh(textured_mesh_path)

            else:
                # Full pipeline mode
                if kwargs['image'] is None:
                    raise ValueError("Image must be provided if mesh is not.")

                result_path = self._process_single_image(
                    kwargs['image'], 
                    "output", 
                    0, 
                    **kwargs
                )
                
                if result_path is None:
                    raise RuntimeError("Failed to process image")
                
                return Output(mesh=result_path)

            output_path = Path("output/mesh.glb")
            final_mesh.export(str(output_path), include_normals=True)

            if not output_path.exists():
                raise RuntimeError(f"Failed to generate mesh file at {output_path}")

            return Output(mesh=output_path)

        except Exception as e:
            self._log_analytics_event("predict_error", {"error": str(e)})
            raise

    def _predict_batch(self, 
                      images: str, 
                      max_batch_size: int,
                      **kwargs) -> Output:
        """
        Sequential batch processing with robust error handling
        """
        batch_start_time = time.time()
        
        # Parse image inputs
        if isinstance(images, str):
            image_list = [img.strip() for img in images.split(',') if img.strip()]
        else:
            image_list = [images]
        
        # Validate and limit batch size
        if len(image_list) > max_batch_size:
            logger.warning(f"Batch size {len(image_list)} exceeds limit {max_batch_size}, truncating")
            image_list = image_list[:max_batch_size]
        
        if len(image_list) == 0:
            raise ValueError("No valid images provided for batch processing")

        # Setup output directory
        if os.path.exists("output"):
            shutil.rmtree("output")
        os.makedirs("output", exist_ok=True)
        
        logger.info(f"🚀 Starting batch processing: {len(image_list)} images")
        logger.info(f"💾 Available VRAM: {self.vram_monitor.get_available_vram():.1f}GB")
        
        self._log_analytics_event("batch_started", {
            "num_images": len(image_list),
            "max_batch_size": max_batch_size,
            **kwargs
        })

        # Process images sequentially
        successful_meshes = []
        failed_images = []
        processing_times = []
        
        for idx, image_input in enumerate(image_list):
            logger.info(f"\n📸 Processing image {idx + 1}/{len(image_list)}")
            
            # Pre-processing safety check
            if not self._check_memory_safety():
                logger.error(f"Insufficient VRAM for image {idx + 1}, skipping remaining images")
                failed_images.extend(image_list[idx:])
                break
            
            image_start_time = time.time()
            
            try:
                # Handle different input types (URL, file path, etc.)
                if image_input.startswith(('http://', 'https://')):
                    # Download image from URL
                    response = requests.get(image_input)
                    image_obj = Image.open(io.BytesIO(response.content))
                else:
                    # Assume it's a file path
                    image_obj = Path(image_input)
                
                result_path = self._process_single_image(
                    image_obj,
                    "output",
                    idx,
                    **kwargs
                )
                
                if result_path is not None:
                    successful_meshes.append(result_path)
                    processing_time = time.time() - image_start_time
                    processing_times.append(processing_time)
                    
                    # Progress update
                    avg_time = sum(processing_times) / len(processing_times)
                    remaining = len(image_list) - idx - 1
                    eta_minutes = (remaining * avg_time) / 60
                    
                    logger.info(f"✅ Image {idx + 1} completed in {processing_time:.1f}s")
                    logger.info(f"📊 Progress: {len(successful_meshes)}/{len(image_list)} successful")
                    if remaining > 0:
                        logger.info(f"⏱️  ETA: {eta_minutes:.1f} minutes")
                else:
                    failed_images.append(image_input)
                    logger.error(f"❌ Failed to process image {idx + 1}")
                    
            except Exception as e:
                failed_images.append(image_input)
                logger.error(f"❌ Error processing image {idx + 1}: {str(e)}")
                continue
            
            # Cleanup between images
            self._cleanup_gpu_memory()
        
        # Final statistics
        total_time = time.time() - batch_start_time
        success_rate = len(successful_meshes) / len(image_list) * 100
        avg_time_per_image = sum(processing_times) / len(processing_times) if processing_times else 0
        
        stats = {
            "total_images": len(image_list),
            "successful": len(successful_meshes),
            "failed": len(failed_images),
            "success_rate_percent": round(success_rate, 1),
            "total_time_minutes": round(total_time / 60, 1),
            "average_time_per_image_seconds": round(avg_time_per_image, 1),
            "peak_vram_usage_gb": round(self.vram_monitor.get_used_vram(), 1)
        }
        
        logger.info(f"\n🏁 Batch processing completed!")
        logger.info(f"📊 Results: {len(successful_meshes)}/{len(image_list)} successful ({success_rate:.1f}%)")
        logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
        logger.info(f"⚡ Average per image: {avg_time_per_image:.1f} seconds")
        
        self._log_analytics_event("batch_completed", stats)
        
        return Output(
            meshes=successful_meshes,
            failed_images=failed_images,
            processing_stats=stats
        ) 