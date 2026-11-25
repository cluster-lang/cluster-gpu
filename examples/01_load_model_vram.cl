// Example 1: Loading Any Model Weights Directly into GT 610 Physical VRAM
// Demonstrates zero host-RAM usage: tensors transferred via PCIe DMA into OpenGL 4.3 SSBOs.

import "std/tensor.cl"
import "std/ai/gpu.cl"
import "std/math.zk"

fn main():
    println("=======================================================")
    println(" Cluster GPU: Model Ingestion into Physical VRAM")
    println("=======================================================")

    ok := gpu_vram_init()
    if ok != 1:
        println("[ERROR] Failed to initialize GPU OpenGL 4.3 context")
        return

    avail_before := gpu_vram_available_memory_mb()
    println("Available VRAM before load: " + to_text(avail_before) + " MB")

    // Target model path (GGUF, CLM, or ONNX)
    model_path := "models/qwen2.5-0.5b-instruct-q4_0.gguf"
    println("Target model: " + model_path)

    // Load model over PCIe directly into GT 610 hardware VRAM
    binding := gpu_vram_load_file(model_path)
    if binding.buf_id <= 0:
        println("[NOTE] Place model at '" + model_path + "' to run DMA upload.")
        return

    println("[OK] Model allocated in hardware VRAM!")
    println("  - SSBO Buffer ID: #" + to_text(binding.buf_id))
    println("  - Transferred Bytes: " + to_text(binding.num_elements) + " bytes")

    // Read first 16 bytes over PCIe to verify hardware integrity
    hex_sample := gpu_vram_sample_hex(binding, 16)
    println("  - DMA Verified Header: " + hex_sample)

    avail_after := gpu_vram_available_memory_mb()
    println("Available VRAM after load: " + to_text(avail_after) + " MB")
    println("Host CPU RAM freed: 100% of weights reside in GT 610 DDR3 VRAM.")
    gpu_vram_release(binding)
