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
from pathlib import Path
from uuid import uuid4
import pymeshlab
import trimesh


def remesh_mesh(mesh_path: str, output_dir_path: str):
    """Create a **unique** remesh filename in the given directory (or the directory part of a path).

    textureGenPipeline historically passes a *file path* like
    ``/tmp/output/meshes/white_mesh_remesh.obj``.  We ignore the filename and
    always generate our own UUID-based name to avoid collisions between
    parallel workers.
    """
    out_dir = Path(output_dir_path).parent if output_dir_path.endswith(".obj") else Path(output_dir_path)

    # Ensure directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always generate a unique OBJ name
    remesh_path = out_dir / f"{uuid4().hex}_remesh.obj"

    mesh_simplify_trimesh(mesh_path, str(remesh_path))
    return str(remesh_path)


def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):
    """
    Simplifies a mesh to a target face count, ensuring thread safety for parallel execution.
    - Creates the output directory if it doesn't exist.
    - Uses a unique temporary file for intermediate processing to avoid race conditions.
    - Preserves the original mesh simplification logic using pymeshlab and trimesh.
    """
    out_path = Path(outputpath).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a unique temporary path for the intermediate mesh to avoid worker collisions
    tmp_path = out_path.with_name(f"{out_path.stem}_{uuid4().hex}.obj")

    try:
        # 1. Load and clean mesh using pymeshlab, saving to a unique temp file
        ms = pymeshlab.MeshSet()
        if inputpath.endswith(".glb"):
            ms.load_new_mesh(inputpath, load_in_a_single_layer=True)
        else:
            ms.load_new_mesh(inputpath)
        ms.save_current_mesh(str(tmp_path), save_textures=False)
        del ms

        # 2. Load the cleaned mesh in trimesh for simplification
        courent = trimesh.load(str(tmp_path), force="mesh")
        face_num = len(courent.faces)

        # 3. Simplify the mesh if it exceeds the target face count
        if face_num > target_count:
            courent = courent.simplify_quadric_decimation(target_count)

        # 4. Ensure directory still exists (race-safe) before export
        Path(outputpath).parent.mkdir(parents=True, exist_ok=True)
        #    Export the final, simplified mesh
        courent.export(outputpath)
        del courent

    finally:
        # 5. Ensure the temporary file is always deleted
        tmp_path.unlink(missing_ok=True)

    return outputpath
