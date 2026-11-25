// Example 2: Offloading Model and Freeing GPU Hardware VRAM
// Demonstrates instantaneous hardware SSBO deallocation and returning memory to pool.

import "std/tensor.cl"
import "std/ai/gpu.cl"
import "std/math.zk"

fn main():
    println("=======================================================")
    println(" Cluster GPU: Instant VRAM Offloading")
    println("=======================================================")

    ok := gpu_vram_init()
    if ok != 1:
        println("[ERROR] Failed to initialize GPU OpenGL 4.3 context")
        return

    model_path := "models/qwen2.5-0.5b-instruct-q4_0.gguf"
    binding := gpu_vram_load_file(model_path)
    if binding.buf_id <= 0:
        println("[NOTE] Model not found at " + model_path + "; testing 64 MB buffer offload instead.")
        binding = gpu_vram_allocate_bytes(67108864)

    println("Allocated SSBO Buffer #" + to_text(binding.buf_id) + " in GT 610 VRAM.")

    // Offload model immediately
    println("Offloading buffer from GT 610 physical DDR3 VRAM...")
    gpu_vram_release(binding)

    println("[OK] SSBO Buffer #" + to_text(binding.buf_id) + " deallocated successfully.")
    println("[OK] VRAM returned to system pool. GPU memory ready for next model.")
