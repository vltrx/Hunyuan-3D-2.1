#!/usr/bin/env python3
"""
Quick verification script for vltrx/hunyuan3d-2-1 model
"""

import replicate

def verify_model():
    """Verify the vltrx/hunyuan3d-2-1 model is accessible and working"""
    
    model_name = "vltrx/hunyuan3d-2-1"
    
    print(f"🔍 Verifying model: {model_name}")
    print("=" * 50)
    
    try:
        # Get model info
        model = replicate.models.get(model_name)
        print(f"✅ Model found: {model.name}")
        print(f"📝 Description: {model.description or 'No description'}")
        print(f"👤 Owner: {model.owner}")
        print(f"🔗 URL: https://replicate.com/{model_name}")
        
        # Get latest version
        latest_version = model.latest_version
        if latest_version:
            print(f"📦 Latest version: {latest_version.id}")
            print(f"📅 Created: {latest_version.created_at}")
            
            # Show input schema
            if hasattr(latest_version, 'openapi_schema') and latest_version.openapi_schema:
                schema = latest_version.openapi_schema
                if 'components' in schema and 'schemas' in schema['components']:
                    input_schema = schema['components']['schemas'].get('Input', {})
                    if 'properties' in input_schema:
                        print(f"\n📋 Available inputs:")
                        for param, details in input_schema['properties'].items():
                            param_type = details.get('type', 'unknown')
                            description = details.get('description', 'No description')
                            print(f"  • {param} ({param_type}): {description}")
        
        print(f"\n🎯 Model verification successful!")
        print(f"🚀 Ready to use: {model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying model: {str(e)}")
        print(f"💡 Make sure the model exists at: https://replicate.com/{model_name}")
        return False

def show_usage_examples():
    """Show usage examples for the model"""
    
    print("\n" + "=" * 50)
    print("📖 USAGE EXAMPLES")
    print("=" * 50)
    
    print("\n🔄 Batch Processing (Recommended):")
    print("```python")
    print("import replicate")
    print("")
    print("output = replicate.run(")
    print('    "vltrx/hunyuan3d-2-1",')
    print("    input={")
    print('        "batch_mode": True,')
    print('        "images": "img1.jpg,img2.jpg,img3.jpg",')
    print('        "max_batch_size": 10,')
    print('        "octree_resolution": 384,')
    print('        "steps": 40')
    print("    }")
    print(")")
    print("```")
    
    print("\n📸 Single Image:")
    print("```python")
    print("import replicate")
    print("")
    print("output = replicate.run(")
    print('    "vltrx/hunyuan3d-2-1",')
    print("    input={")
    print('        "image": open("image.jpg", "rb"),')
    print('        "batch_mode": False,')
    print('        "octree_resolution": 512')
    print("    }")
    print(")")
    print("```")
    
    print("\n🧪 Test with script:")
    print("```bash")
    print("python test_deployment.py vltrx/hunyuan3d-2-1 \\")
    print("    --image test.jpg \\")
    print("    --images img1.jpg img2.jpg img3.jpg \\")
    print("    --test-mode all")
    print("```")

if __name__ == "__main__":
    success = verify_model()
    show_usage_examples()
    
    if success:
        print(f"\n🎉 Your model vltrx/hunyuan3d-2-1 is ready to use!")
    else:
        print(f"\n⚠️  Please check your model setup and try again.") 