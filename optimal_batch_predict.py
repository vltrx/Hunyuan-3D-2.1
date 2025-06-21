import os
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import torch
from torch import cuda

class OptimalBatchPredictor:
    """Optimized batch processing for L40S (48GB VRAM)"""
    
    # Batch size configurations for different scenarios
    BATCH_CONFIGS = {
        "fast": {"max_batch": 5, "octree_res": 256, "max_faces": 20000},
        "balanced": {"max_batch": 10, "octree_res": 384, "max_faces": 30000}, 
        "quality": {"max_batch": 3, "octree_res": 512, "max_faces": 40000},
        "bulk": {"max_batch": 20, "octree_res": 256, "max_faces": 15000}
    }
    
    def __init__(self):
        self.vram_monitor = VRAMMonitor()
        
    def determine_optimal_batch_size(self, 
                                   num_images: int, 
                                   quality_mode: str = "balanced") -> Dict[str, Any]:
        """Determine optimal batch size based on available VRAM and requirements"""
        
        config = self.BATCH_CONFIGS[quality_mode]
        available_vram = self.vram_monitor.get_available_vram()
        
        # Conservative batch sizing based on VRAM headroom
        if available_vram < 35:  # Less than 35GB available
            recommended_batch = min(config["max_batch"] // 2, num_images)
        elif available_vram < 42:  # Less than 42GB available  
            recommended_batch = min(config["max_batch"], num_images)
        else:  # Plenty of VRAM
            recommended_batch = min(config["max_batch"] * 2, num_images, 25)  # Cap at 25
            
        return {
            "batch_size": recommended_batch,
            "octree_resolution": config["octree_res"],
            "max_faces": config["max_faces"],
            "estimated_time": recommended_batch * 3,  # 3 min per image estimate
            "estimated_vram": 29,  # Peak per image
            "processing_mode": "sequential_with_cleanup"
        }
    
    def predict_optimized_batch(self, 
                              images: List[str],
                              quality_mode: str = "balanced",
                              enable_checkpointing: bool = True) -> List[str]:
        """
        Process batch with optimal L40S utilization
        """
        
        num_images = len(images)
        config = self.determine_optimal_batch_size(num_images, quality_mode)
        
        print(f"🚀 Optimized Batch Processing on L40S")
        print(f"📊 Images: {num_images}")
        print(f"⚙️  Batch size: {config['batch_size']}")
        print(f"📐 Octree resolution: {config['octree_resolution']}")
        print(f"⏱️  Estimated time: {config['estimated_time']} minutes")
        print(f"💾 Peak VRAM: {config['estimated_vram']}GB per image")
        
        # Split into optimal chunks
        chunks = self._chunk_images(images, config['batch_size'])
        all_results = []
        
        for chunk_idx, chunk in enumerate(chunks):
            print(f"\n🔄 Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} images)")
            
            chunk_results = self._process_chunk_sequential(
                chunk, 
                config,
                enable_checkpointing
            )
            
            all_results.extend(chunk_results)
            
            # Memory cleanup between chunks
            self._aggressive_cleanup()
            
            # Progress report
            completed = sum(len(chunks[:i+1]) for i in range(chunk_idx + 1))
            print(f"✅ Progress: {completed}/{num_images} images completed")
        
        return all_results
    
    def _chunk_images(self, images: List[str], chunk_size: int) -> List[List[str]]:
        """Split images into processing chunks"""
        return [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]
    
    def _process_chunk_sequential(self, 
                                chunk: List[str], 
                                config: Dict[str, Any],
                                enable_checkpointing: bool) -> List[str]:
        """Process a chunk of images sequentially with memory management"""
        
        results = []
        
        for idx, image_path in enumerate(chunk):
            start_time = time.time()
            
            # Pre-processing VRAM check
            available_vram = self.vram_monitor.get_available_vram()
            if available_vram < 30:  # Need at least 30GB free
                print(f"⚠️  Low VRAM ({available_vram}GB), forcing cleanup...")
                self._aggressive_cleanup()
            
            try:
                # Process single image with config settings
                result = self._process_single_optimized(
                    image_path, 
                    config,
                    f"chunk_image_{idx}"
                )
                
                results.append(result)
                
                # Post-processing cleanup
                self._cleanup_gpu_memory()
                
                processing_time = time.time() - start_time
                remaining = len(chunk) - idx - 1
                eta = remaining * processing_time
                
                print(f"  ✅ Image {idx+1}/{len(chunk)} completed in {processing_time:.1f}s")
                print(f"  📊 VRAM: {self.vram_monitor.get_used_vram():.1f}GB used")
                if remaining > 0:
                    print(f"  ⏱️  ETA for chunk: {eta/60:.1f} minutes")
                
                # Optional checkpointing
                if enable_checkpointing and (idx + 1) % 5 == 0:
                    self._save_checkpoint(results, f"checkpoint_{idx+1}")
                    
            except Exception as e:
                print(f"❌ Failed to process {image_path}: {str(e)}")
                # Continue with next image instead of failing entire chunk
                continue
        
        return results
    
    def _process_single_optimized(self, 
                                image_path: str, 
                                config: Dict[str, Any],
                                output_name: str) -> str:
        """Process single image with optimized settings"""
        
        # Implementation would use the config parameters:
        # - config['octree_resolution'] 
        # - config['max_faces']
        # - Optimized for L40S memory patterns
        
        # Placeholder for actual processing
        result_path = f"output/{output_name}.glb"
        
        # Simulate processing with memory monitoring
        peak_vram = self.vram_monitor.get_used_vram()
        print(f"    📈 Peak VRAM: {peak_vram:.1f}GB")
        
        return result_path
    
    def _cleanup_gpu_memory(self):
        """Aggressive GPU memory cleanup"""
        if cuda.is_available():
            cuda.empty_cache()
            cuda.ipc_collect()
            # Force garbage collection
            import gc
            gc.collect()
    
    def _aggressive_cleanup(self):
        """Maximum cleanup between chunks"""
        self._cleanup_gpu_memory()
        time.sleep(1)  # Give GPU time to actually free memory
        print(f"🧹 Memory cleanup: {self.vram_monitor.get_available_vram():.1f}GB available")
    
    def _save_checkpoint(self, results: List[str], checkpoint_name: str):
        """Save intermediate results"""
        checkpoint_path = f"checkpoints/{checkpoint_name}.json"
        os.makedirs("checkpoints", exist_ok=True)
        
        import json
        with open(checkpoint_path, 'w') as f:
            json.dump({"results": results, "timestamp": time.time()}, f)
        print(f"💾 Checkpoint saved: {checkpoint_path}")


class VRAMMonitor:
    """Monitor VRAM usage on L40S"""
    
    def get_available_vram(self) -> float:
        """Get available VRAM in GB"""
        if not cuda.is_available():
            return 0.0
        
        total = cuda.get_device_properties(0).total_memory / 1024**3
        allocated = cuda.memory_allocated(0) / 1024**3
        return total - allocated
    
    def get_used_vram(self) -> float:
        """Get used VRAM in GB"""
        if not cuda.is_available():
            return 0.0
        return cuda.memory_allocated(0) / 1024**3
    
    def get_total_vram(self) -> float:
        """Get total VRAM in GB"""
        if not cuda.is_available():
            return 0.0
        return cuda.get_device_properties(0).total_memory / 1024**3


# Usage example
def main():
    predictor = OptimalBatchPredictor()
    
    # Example batch processing scenarios
    test_images = [f"image_{i}.jpg" for i in range(15)]
    
    # Different processing modes
    modes = {
        "fast": "Quick processing, lower quality",
        "balanced": "Good balance of speed and quality", 
        "quality": "Best quality, slower processing",
        "bulk": "Maximum throughput for large batches"
    }
    
    for mode, description in modes.items():
        print(f"\n🎯 Mode: {mode} - {description}")
        config = predictor.determine_optimal_batch_size(len(test_images), mode)
        print(f"   Recommended batch size: {config['batch_size']}")
        print(f"   Estimated processing time: {config['estimated_time']} minutes")

if __name__ == "__main__":
    main() 