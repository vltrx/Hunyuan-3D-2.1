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

from pathlib import Path
from uuid import uuid4
import pymeshlab
import trimesh


def remesh_mesh(mesh_path, remesh_path):
    mesh_simplify_trimesh(mesh_path, remesh_path)


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

        # 4. Export the final, simplified mesh to the intended output path
        courent.export(outputpath)
        del courent

    finally:
        # 5. Ensure the temporary file is always deleted
        tmp_path.unlink(missing_ok=True)

    return outputpath
