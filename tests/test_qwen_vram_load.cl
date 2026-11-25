// ============================================================================
//  Test: Load Qwen 0.5B GGUF Model into NVIDIA GT 610 Discrete VRAM
// ============================================================================

import "std/tensor.cl"
import "std/ai/gpu.cl"

fn main():
    println("==================================================================")
    println("       QWEN 2.5 0.5B: DIRECT HARDWARE VRAM INGESTION TEST        ")
    println("==================================================================")
    
    ok := gpu_vram_init()
    if ok != 1:
        println("[ERROR] Failed to init GPU")
        return
        
    renderer := gpu_vram_query_renderer()
    vendor := gpu_vram_query_vendor()
    total_mb := gpu_vram_total_memory_mb()
    avail_mb := gpu_vram_available_memory_mb()
    println("GPU: " + renderer + " (" + vendor + ")")
    println("Total Dedicated VRAM: " + to_text(total_mb) + " MB")
    println("Available VRAM:       " + to_text(avail_mb) + " MB")

    qwen_path := "models/qwen2.5-0.5b-instruct-q4_0.gguf"
    println("\nLoading Qwen 2.5 0.5B into GT 610 VRAM: " + qwen_path)
    
    qwen_binding := gpu_vram_load_file(qwen_path)
    if qwen_binding.buf_id > 0:
        println("-> [SUCCESS] Allocated Hardware SSBO ID: " + to_text(qwen_binding.buf_id))
        println("-> [SUCCESS] Ingested " + to_text(qwen_binding.num_elements) + " bytes (~409 MB) into GT 610 DDR3 VRAM")
        
        // Sample first 16 bytes directly from physical VRAM over PCIe
        hex := gpu_vram_sample_hex(qwen_binding, 16)
        println("-> VRAM Memory-Mapped Sample: " + hex)
        println("-> Expected GGUF Magic:       47 47 55 46 03 00 00 00 23 01 00 00 00 00 00 00 ")
        
        gpu_vram_release(qwen_binding)
        println("-> Released Qwen 0.5B buffer from VRAM")
    else:
        println("-> [FAILED] Could not load Qwen into VRAM")
    println("==================================================================")
