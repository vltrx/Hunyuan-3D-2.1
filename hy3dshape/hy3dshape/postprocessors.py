# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import tempfile
from typing import Union

import numpy as np
import pymeshlab
import torch
import trimesh

from .models.autoencoders import Latent2MeshOutput
from .utils import synchronize_timer


def load_mesh(path):
    """Load mesh with validation and proper GLB handling"""
    try:
        if path.endswith(".glb"):
            mesh = trimesh.load(path)
            # Handle GLB files that might load as Scene
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) == 0:
                    raise ValueError("GLB file contains no geometry")
                # Combine all geometries into single mesh
                combined_mesh = trimesh.Trimesh()
                for geom in mesh.geometry.values():
                    if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
                        combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
                mesh = combined_mesh
            
            # Validate the loaded trimesh
            if mesh is None:
                raise ValueError("Failed to load mesh from GLB file")
            if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                raise ValueError("Loaded GLB mesh has no vertices")
            if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
                raise ValueError("Loaded GLB mesh has no faces")
                
        else:
            mesh = pymeshlab.MeshSet()
            mesh.load_new_mesh(path)
            # Validate pymeshlab mesh
            if mesh.current_mesh().vertex_number() == 0:
                raise ValueError("Loaded mesh has no vertices")
            if mesh.current_mesh().face_number() == 0:
                raise ValueError("Loaded mesh has no faces")
                
        return mesh
        
    except Exception as e:
        raise ValueError(f"Failed to load mesh from {path}: {str(e)}")


def reduce_face(mesh: pymeshlab.MeshSet, max_facenum: int = 200000):
    if max_facenum > mesh.current_mesh().face_number():
        return mesh

    mesh.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True
    )
    return mesh


def remove_floater(mesh: pymeshlab.MeshSet):
    mesh.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                      nbfaceratio=0.005)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def pymeshlab2trimesh(mesh: pymeshlab.MeshSet):
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
        mesh.save_current_mesh(temp_file.name)
        mesh = trimesh.load(temp_file.name)
    # 检查加载的对象类型
    if isinstance(mesh, trimesh.Scene):
        combined_mesh = trimesh.Trimesh()
        # 如果是Scene，遍历所有的geometry并合并
        for geom in mesh.geometry.values():
            combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
        mesh = combined_mesh
    return mesh


def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        mesh.export(temp_file.name)
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(temp_file.name)
    return mesh


def export_mesh(input, output):
    if isinstance(input, pymeshlab.MeshSet):
        mesh = output
    elif isinstance(input, Latent2MeshOutput):
        output = Latent2MeshOutput()
        output.mesh_v = output.current_mesh().vertex_matrix()
        output.mesh_f = output.current_mesh().face_matrix()
        mesh = output
    else:
        mesh = pymeshlab2trimesh(output)
    return mesh


def import_mesh(mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str]) -> pymeshlab.MeshSet:
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)
    elif isinstance(mesh, Latent2MeshOutput):
        mesh = pymeshlab.MeshSet()
        mesh_pymeshlab = pymeshlab.Mesh(vertex_matrix=mesh.mesh_v, face_matrix=mesh.mesh_f)
        mesh.add_mesh(mesh_pymeshlab, "converted_mesh")

    if isinstance(mesh, (trimesh.Trimesh, trimesh.scene.Scene)):
        mesh = trimesh2pymeshlab(mesh)

    return mesh


class FaceReducer:
    @synchronize_timer('FaceReducer')
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
        max_facenum: int = 40000
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh]:
        ms = import_mesh(mesh)
        ms = reduce_face(ms, max_facenum=max_facenum)
        mesh = export_mesh(mesh, ms)
        return mesh


class FloaterRemover:
    @synchronize_timer('FloaterRemover')
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_mesh(mesh)
        ms = remove_floater(ms)
        mesh = export_mesh(mesh, ms)
        return mesh


class DegenerateFaceRemover:
    @synchronize_timer('DegenerateFaceRemover')
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_mesh(mesh)

        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
            ms.save_current_mesh(temp_file.name)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_file.name)

        mesh = export_mesh(mesh, ms)
        return mesh


def mesh_normalize(mesh):
    """
    Normalize mesh vertices to sphere
    """
    # Validate mesh has vertices
    if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
        raise ValueError("Cannot normalize empty mesh - mesh has no vertices")
    
    vtx_pos = np.asarray(mesh.vertices)
    
    # Additional validation for degenerate cases
    if vtx_pos.size == 0:
        raise ValueError("Cannot normalize mesh with zero vertices")
    
    scale_factor = 1.2
    max_bb = (vtx_pos - 0).max(0)[0]
    min_bb = (vtx_pos - 0).min(0)[0]

    center = (max_bb + min_bb) / 2

    scale = torch.norm(torch.tensor(vtx_pos - center, dtype=torch.float32), dim=1).max() * 2.0

    vtx_pos = (vtx_pos - center) * (scale_factor / float(scale))
    mesh.vertices = vtx_pos

    return mesh


class MeshSimplifier:
    def __init__(self, executable: str = None):
        if executable is None:
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
            executable = os.path.join(CURRENT_DIR, "mesh_simplifier.bin")
        self.executable = executable
        
        # Check if binary exists at initialization
        self.binary_available = os.path.exists(self.executable) and os.access(self.executable, os.X_OK)
        if not self.binary_available:
            print(f"Warning: Mesh simplifier binary not found at {self.executable}")
            print("Mesh simplification will be skipped - this may affect quality but won't cause errors")

    @synchronize_timer('MeshSimplifier')
    def __call__(
        self,
        mesh: Union[trimesh.Trimesh],
    ) -> Union[trimesh.Trimesh]:
        # Validate input mesh
        if mesh is None:
            raise ValueError("Input mesh is None")
        
        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            raise ValueError("Input mesh has no vertices")
            
        if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            raise ValueError("Input mesh has no faces")
        
        # If binary not available, skip simplification but normalize
        if not self.binary_available:
            print(f"Skipping mesh simplification (binary not available), applying normalization only")
            try:
                return mesh_normalize(mesh)
            except Exception as e:
                print(f"Warning: Mesh normalization failed: {e}, returning original mesh")
                return mesh
        
        # Try mesh simplification with binary
        try:
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_input:
                with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_output:
                    # Export input mesh
                    mesh.export(temp_input.name)
                    
                    # Run simplifier binary
                    result = os.system(f'{self.executable} {temp_input.name} {temp_output.name}')
                    
                    # Check if command succeeded
                    if result != 0:
                        print(f"Mesh simplifier binary failed with exit code {result}, skipping simplification")
                        return mesh_normalize(mesh)
                    
                    # Check if output file was created
                    if not os.path.exists(temp_output.name) or os.path.getsize(temp_output.name) == 0:
                        print(f"Mesh simplifier produced empty output, skipping simplification")
                        return mesh_normalize(mesh)
                    
                    # Load simplified mesh
                    ms = trimesh.load(temp_output.name, process=False)
                    if isinstance(ms, trimesh.Scene):
                        combined_mesh = trimesh.Trimesh()
                        for geom in ms.geometry.values():
                            combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
                        ms = combined_mesh
                    
                    # Validate simplified mesh before normalizing
                    if ms is None or len(ms.vertices) == 0 or len(ms.faces) == 0:
                        print("Simplified mesh is empty, returning normalized original")
                        return mesh_normalize(mesh)
                    
                    # Normalize and return
                    ms = mesh_normalize(ms)
                    return ms
                    
        except Exception as e:
            print(f"Mesh simplification failed with error: {e}, returning normalized original")
            try:
                return mesh_normalize(mesh)
            except Exception as norm_e:
                print(f"Warning: Mesh normalization also failed: {norm_e}, returning original mesh")
                return mesh
        finally:
            # Cleanup temp files
            try:
                if 'temp_input' in locals():
                    os.unlink(temp_input.name)
                if 'temp_output' in locals():
                    os.unlink(temp_output.name)
            except:
                pass
