import os
import shutil
import subprocess
import sys
import time
from typing import List

from PIL import Image
from torch import cuda, Generator
from cog import BasePredictor, BaseModel, Input, Path

# Add the necessary paths for the new module structure
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.postprocessors import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.models.autoencoders import SurfaceExtractors
from hy3dshape.utils import logger
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

class BatchOutput(BaseModel):
    meshes: List[Path]

class BatchPredictor(BasePredictor):
    def setup(self) -> None:
        # Same setup as original predict.py
        start = time.time()
        logger.info("Setup started")
        os.environ["OMP_NUM_THREADS"] = "1"
        
        # ... (same setup code as original)
        
    def _process_single_image(self, image_path: Path, **kwargs) -> Path:
        """Process a single image and return the mesh path"""
        from trimesh import load as load_trimesh
        
        # Create unique output directory for this image
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = f"output/{image_name}_{int(time.time()*1000)}"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            input_image = Image.open(str(image_path))
            if kwargs.get('remove_background', True):
                input_image = self.rmbg_worker(input_image.convert("RGB"))
                self._cleanup_gpu_memory()

            # Generate shape
            mesh_output = self.i23d_worker(
                image=input_image,
                num_inference_steps=kwargs.get('steps', 50),
                guidance_scale=kwargs.get('guidance_scale', 5.0),
                generator=Generator().manual_seed(kwargs.get('seed', 1234)),
                octree_resolution=kwargs.get('octree_resolution', 512),
                num_chunks=kwargs.get('num_chunks', 8000),
            )[0]

            self._cleanup_gpu_memory()
            
            # Post-process mesh
            mesh_output = self.floater_remove_worker(mesh_output)
            mesh_output = self.degenerate_face_remove_worker(mesh_output)
            mesh_output = self.mesh_simplifier(mesh_output)
            mesh_output = self.face_reduce_worker(mesh_output, max_facenum=kwargs.get('max_facenum', 40000))
            self._cleanup_gpu_memory()

            # Save intermediate mesh
            temp_mesh_path = f"{output_dir}/temp_mesh.obj"
            mesh_output.export(temp_mesh_path)

            # Apply texturing
            textured_mesh_path = self.texgen_worker(
                mesh_path=temp_mesh_path,
                image_path=input_image,
                output_mesh_path=f"{output_dir}/textured_mesh.obj",
                save_glb=False
            )

            # Load and export final mesh
            final_mesh = load_trimesh(textured_mesh_path)
            output_path = Path(f"{output_dir}/mesh.glb")
            final_mesh.export(str(output_path), include_normals=True)
            
            return output_path

        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            raise
        finally:
            self._cleanup_gpu_memory()

    def predict(
        self,
        images: str = Input(
            description="Comma-separated list of image URLs or single image",
            default=None,
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
        max_batch_size: int = Input(
            description="Maximum number of images to process (memory limited)",
            default=3,
            ge=1,
            le=5,
        ),
    ) -> BatchOutput:
        
        start_time = time.time()
        
        # Parse image inputs (could be URLs, file paths, etc.)
        if isinstance(images, str):
            image_list = [img.strip() for img in images.split(',')]
        else:
            image_list = [images]
        
        # Limit batch size to prevent OOM
        if len(image_list) > max_batch_size:
            logger.warning(f"Batch size {len(image_list)} exceeds limit {max_batch_size}, truncating")
            image_list = image_list[:max_batch_size]
        
        if os.path.exists("output"):
            shutil.rmtree("output")
        os.makedirs("output", exist_ok=True)
        
        logger.info(f"Processing batch of {len(image_list)} images")
        
        # Process images sequentially
        results = []
        kwargs = {
            'steps': steps,
            'guidance_scale': guidance_scale,
            'max_facenum': max_facenum,
            'num_chunks': num_chunks,
            'seed': seed,
            'octree_resolution': octree_resolution,
            'remove_background': remove_background,
        }
        
        for i, image_path in enumerate(image_list):
            logger.info(f"Processing image {i+1}/{len(image_list)}: {image_path}")
            
            try:
                # Use different seed for each image
                kwargs['seed'] = seed + i
                result_path = self._process_single_image(Path(image_path), **kwargs)
                results.append(result_path)
                
                logger.info(f"Completed image {i+1}/{len(image_list)} in {time.time() - start_time:.1f}s")
                
            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {str(e)}")
                # Continue with next image instead of failing entire batch
                continue
        
        total_time = time.time() - start_time
        logger.info(f"Batch processing completed: {len(results)}/{len(image_list)} successful in {total_time:.1f}s")
        
        return BatchOutput(meshes=results)

    def _cleanup_gpu_memory(self):
        if cuda.is_available():
            cuda.empty_cache()
            cuda.ipc_collect() 