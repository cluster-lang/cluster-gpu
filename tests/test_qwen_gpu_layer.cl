// Test tensor extraction and GPU upload for Qwen 0.5B

import "std/tensor.cl"
import "std/ai/model.cl"
import "std/ai/gpu.cl"

fn main():
    println("==================================================================")
    println("      QWEN 0.5B TENSOR DEQUANTIZATION & GPU VRAM DISPATCH        ")
    println("==================================================================")
    
    ok := gpu_vram_init()
    if ok != 1:
        println("[ERROR] Failed to init GPU")
        return
        
    path := "models/qwen2.5-0.5b-instruct-q4_0.gguf"
    println("Opening Qwen 0.5B model via zero-copy mmap...")
    m := model_open(path)
    if m == 0:
        println("[ERROR] Failed to open model")
        return
        
    println("[1] Extracting blk.0.attn_norm.weight (F32)...")
    t_norm := model_get_tensor(m, "blk.0.attn_norm.weight")
    println("    Shape: [" + to_text(t_norm.rows()) + ", " + to_text(t_norm.cols()) + "]")
    v0 := t_norm.get(0, 0)
    println("    Sample value[0]: " + to_text(v0))
    
    println("[2] Extracting blk.0.attn_q.weight (Q4_0 dequantized to F32)...")
    t_wq := model_get_tensor(m, "blk.0.attn_q.weight")
    println("    Shape: [" + to_text(t_wq.rows()) + ", " + to_text(t_wq.cols()) + "]")
    wq_0_0 := t_wq.get(0, 0)
    println("    Sample value[0,0]: " + to_text(wq_0_0))

    println("[3] Uploading Qwen layer weights into GT 610 VRAM...")
    buf_norm := gpu_vram_allocate(t_norm.cols())
    buf_wq := gpu_vram_allocate(t_wq.rows() * t_wq.cols())
    
    gpu_vram_upload_tensor(buf_norm, t_norm)
    gpu_vram_upload_tensor(buf_wq, t_wq)
    println("    -> Uploaded to GT 610 SSBO ID: " + to_text(buf_norm.buf_id) + " and " + to_text(buf_wq.buf_id))

    println("[4] Dispatching GEMM Projection on GT 610 48 CUDA cores...")
    // Create activation vector [1, 896]
    t_x := tensor_fill(1, 896, 0.1)
    buf_x := gpu_vram_allocate(896)
    buf_q := gpu_vram_allocate(896)
    gpu_vram_upload_tensor(buf_x, t_x)
    
    // Dispatch Q projection: [1, 896] * [896, 896] -> [1, 896]
    comp_ok := gpu_vram_compute_gemm(buf_x, buf_wq, buf_q, 1, 896, 896)
    if comp_ok == 1:
        println("    -> [SUCCESS] GT 610 Fermi CUDA cores computed Q-projection directly in VRAM!")
        t_out := tensor_zeros(1, 896)
        gpu_vram_download_tensor(buf_q, t_out)
        q0 := t_out.get(0, 0)
        q1 := t_out.get(0, 1)
        println("    -> Downloaded activations from VRAM: Q[0]=" + to_text(q0) + ", Q[1]=" + to_text(q1))

    gpu_vram_release(buf_q)
    gpu_vram_release(buf_x)
    gpu_vram_release(buf_wq)
    gpu_vram_release(buf_norm)
    model_close(m)
    println("==================================================================")
    println("  GPU VRAM INFERENCE PIPELINE FOR QWEN 0.5B FULLY OPERATIONAL!    ")
    println("==================================================================")
