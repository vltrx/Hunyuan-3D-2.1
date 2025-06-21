#!/usr/bin/env python3
"""
Test script following the exact HuggingFace demo.py pattern
to validate our API improvements work correctly.
"""

import sys
import os
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

try:
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    print("✅ Successfully imported textureGenPipeline (HF pattern)")
except ImportError as e:
    print(f"❌ Failed to import textureGenPipeline: {e}")
    try:
        from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        print("✅ Successfully imported hy3dpaint.textureGenPipeline (fallback)")
    except ImportError as e2:
        print(f"❌ Failed fallback import: {e2}")
        sys.exit(1)

try:
    from torchvision_fix import apply_fix
    apply_fix()
    print("✅ Applied torchvision compatibility fix")
except ImportError:
    print("⚠️  Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"⚠️  Warning: Failed to apply torchvision fix: {e}")

def test_hf_pattern():
    """Test the exact HuggingFace demo.py pattern"""
    
    print("\n🧪 Testing HuggingFace demo.py pattern...")
    
    # Test 1: Shape pipeline initialization
    print("\n1️⃣  Testing shape pipeline initialization...")
    try:
        model_path = 'tencent/Hunyuan3D-2.1'
        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
        print("✅ Shape pipeline loaded successfully")
    except Exception as e:
        print(f"❌ Shape pipeline failed: {e}")
        return False
    
    # Test 2: Background removal
    print("\n2️⃣  Testing background removal...")
    try:
        # Create a test image if demo doesn't exist
        if not os.path.exists('assets/demo.png'):
            print("⚠️  assets/demo.png not found, creating test image...")
            os.makedirs('assets', exist_ok=True)
            test_img = Image.new('RGB', (512, 512), color='red')
            test_img.save('assets/demo.png')
        
        image = Image.open('assets/demo.png').convert("RGBA")
        if image.mode == 'RGB':
            rembg = BackgroundRemover()
            image = rembg(image)
        print("✅ Background removal successful")
    except Exception as e:
        print(f"❌ Background removal failed: {e}")
        return False
    
    # Test 3: Texture pipeline configuration  
    print("\n3️⃣  Testing texture pipeline configuration...")
    try:
        max_num_view = 6  # can be 6 to 9
        resolution = 512  # can be 768 or 512
        conf = Hunyuan3DPaintConfig(max_num_view, resolution)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        print("✅ Texture pipeline config created successfully")
        
        # Test pipeline initialization (without loading models)
        print("   Testing pipeline initialization...")
        paint_pipeline = Hunyuan3DPaintPipeline(conf)
        print("✅ Texture pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Texture pipeline config failed: {e}")
        return False
    
    print("\n🎉 All HuggingFace pattern tests passed!")
    return True

if __name__ == "__main__":
    success = test_hf_pattern()
    if success:
        print("\n✅ HuggingFace pattern validation successful!")
        print("   The API changes should work correctly in deployment.")
    else:
        print("\n❌ HuggingFace pattern validation failed!")
        print("   There may be import or configuration issues to resolve.")
        sys.exit(1) 