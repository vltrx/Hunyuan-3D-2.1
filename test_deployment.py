#!/usr/bin/env python3
"""
Test script for Hunyuan3D-2.1 Replicate deployment with batch processing
"""

import os
import replicate
import argparse


def test_image_to_3d(model_name, image_path):
    """Test basic image-to-3D generation"""
    print(f"Testing image-to-3D generation with model: {model_name}")
    print(f"Input image: {image_path}")
    
    try:
        with open(image_path, "rb") as image_file:
            output = replicate.run(
                model_name,
                input={
                    "image": image_file,
                    "steps": 30,  # Reduced for faster testing
                    "guidance_scale": 5.0,
                    "octree_resolution": 256,  # Reduced for faster testing
                    "remove_background": True,
                    "max_facenum": 20000,  # Reduced for faster testing
                    "seed": 1234,
                    "batch_mode": False
                }
            )
        
        print(f"✅ Success! Generated mesh: {output.mesh}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_batch_processing(model_name, image_paths):
    """Test batch processing mode"""
    print(f"Testing batch processing with model: {model_name}")
    print(f"Input images: {len(image_paths)} images")
    
    try:
        # Convert image paths to comma-separated string
        images_str = ",".join(image_paths)
        
        output = replicate.run(
            model_name,
            input={
                "images": images_str,
                "batch_mode": True,
                "max_batch_size": len(image_paths),
                "steps": 25,  # Reduced for faster testing
                "guidance_scale": 5.0,
                "octree_resolution": 256,
                "remove_background": True,
                "max_facenum": 15000,  # Reduced for faster testing
                "seed": 1234
            }
        )
        
        print(f"✅ Success! Batch processing completed:")
        print(f"   📊 Generated meshes: {len(output.meshes)}")
        print(f"   ❌ Failed images: {len(output.failed_images)}")
        print(f"   📈 Success rate: {output.processing_stats.get('success_rate_percent', 0)}%")
        print(f"   ⏱️  Total time: {output.processing_stats.get('total_time_minutes', 0)} minutes")
        print(f"   ⚡ Avg per image: {output.processing_stats.get('average_time_per_image_seconds', 0)} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_texture_only(model_name, image_path, mesh_path):
    """Test texture-only mode with uploaded mesh"""
    print(f"Testing texture-only mode with model: {model_name}")
    print(f"Reference image: {image_path}")
    print(f"Input mesh: {mesh_path}")
    
    try:
        with open(image_path, "rb") as image_file, open(mesh_path, "rb") as mesh_file:
            output = replicate.run(
                model_name,
                input={
                    "image": image_file,
                    "mesh": mesh_file,
                    "prompt": "detailed realistic texture",
                    "remove_background": True,
                    "batch_mode": False
                }
            )
        
        print(f"✅ Success! Textured mesh: {output.mesh}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_performance_modes(model_name, image_path):
    """Test different performance/quality modes"""
    print(f"Testing different performance modes with model: {model_name}")
    
    modes = {
        "fast": {"octree_resolution": 256, "max_facenum": 15000, "steps": 25},
        "balanced": {"octree_resolution": 384, "max_facenum": 30000, "steps": 35},
        "quality": {"octree_resolution": 512, "max_facenum": 40000, "steps": 50}
    }
    
    success_count = 0
    
    for mode_name, settings in modes.items():
        print(f"\n🎯 Testing {mode_name} mode...")
        try:
            with open(image_path, "rb") as image_file:
                output = replicate.run(
                    model_name,
                    input={
                        "image": image_file,
                        "batch_mode": False,
                        **settings,
                        "seed": 1234
                    }
                )
            
            print(f"   ✅ {mode_name} mode successful: {output.mesh}")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ {mode_name} mode failed: {str(e)}")
    
    return success_count == len(modes)


def main():
    parser = argparse.ArgumentParser(description="Test Hunyuan3D-2.1 Replicate deployment")
    parser.add_argument("model_name", help="Replicate model name (e.g., 'vltrx/hunyuan3d-2-1')")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--images", nargs='+', help="Multiple images for batch testing")
    parser.add_argument("--mesh", help="Path to test mesh for texture-only mode")
    parser.add_argument("--test-mode", 
                       choices=["single", "batch", "texture", "performance", "all"], 
                       default="all",
                       help="Which test mode to run")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        return
    
    if args.images:
        for img in args.images:
            if not os.path.exists(img):
                print(f"❌ Error: Image file not found: {img}")
                return
    
    if args.mesh and not os.path.exists(args.mesh):
        print(f"❌ Error: Mesh file not found: {args.mesh}")
        return
    
    print("🚀 Starting Hunyuan3D-2.1 Replicate deployment tests...")
    print("=" * 60)
    
    success_count = 0
    total_tests = 0
    
    # Test single image processing
    if args.test_mode in ["single", "all"]:
        total_tests += 1
        print("\n📸 Test 1: Single Image-to-3D Generation")
        print("-" * 40)
        if test_image_to_3d(args.model_name, args.image):
            success_count += 1
    
    # Test batch processing
    if args.test_mode in ["batch", "all"] and args.images:
        total_tests += 1
        print("\n🔄 Test 2: Batch Processing")
        print("-" * 40)
        if test_batch_processing(args.model_name, args.images):
            success_count += 1
    elif args.test_mode in ["batch", "all"] and not args.images:
        print("\n⚠️  Skipping batch test: no multiple images provided (use --images)")
    
    # Test texture-only mode
    if args.test_mode in ["texture", "all"] and args.mesh:
        total_tests += 1
        print("\n🎨 Test 3: Texture-Only Mode")
        print("-" * 40)
        if test_texture_only(args.model_name, args.image, args.mesh):
            success_count += 1
    elif args.test_mode in ["texture", "all"] and not args.mesh:
        print("\n⚠️  Skipping texture-only test: no mesh file provided (use --mesh)")
    
    # Test performance modes
    if args.test_mode in ["performance", "all"]:
        total_tests += 1
        print("\n⚡ Test 4: Performance Modes")
        print("-" * 40)
        if test_performance_modes(args.model_name, args.image):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Deployment is working correctly.")
        print("\n🚀 Ready for production deployment!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    # Usage examples
    print("\n📖 Usage Examples:")
    print("=" * 30)
    print("# Single image processing:")
    print("replicate run vltrx/hunyuan3d-2-1 image=@image.jpg")
    print("\n# Batch processing:")
    print("replicate run vltrx/hunyuan3d-2-1 batch_mode=true images=\"img1.jpg,img2.jpg,img3.jpg\"")
    print("\n# Texture only:")
    print("replicate run vltrx/hunyuan3d-2-1 image=@texture_ref.jpg mesh=@mesh.glb")


if __name__ == "__main__":
    main() 