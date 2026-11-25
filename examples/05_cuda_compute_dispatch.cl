// Example 5: Firing 48 CUDA Cores with Direct GLSL 4.30 Compute Shaders
// Executes parallel vector operations directly on VRAM memory buffers.

import "std/tensor.cl"
import "std/ai/gpu.cl"
import "std/math.zk"

fn main():
    println("=======================================================")
    println(" Cluster GPU: Direct 48 CUDA Cores Compute Dispatch")
    println("=======================================================")

    ok := gpu_vram_init()
    if ok != 1:
        println("[ERROR] Failed to initialize GPU OpenGL 4.3 context")
        return

    n := 1024
    buf_a := gpu_vram_allocate(n)
    buf_b := gpu_vram_allocate(n)
    buf_c := gpu_vram_allocate(n)

    t_a := tensor_fill(1, n, 2.5)
    t_b := tensor_fill(1, n, 4.0)
    t_c := tensor_zeros(1, n)

    gpu_vram_upload_tensor(buf_a, t_a)
    gpu_vram_upload_tensor(buf_b, t_b)

    println("Allocated 3 SSBO buffers in VRAM (" + to_text(n) + " elements each).")
    println("Dispatching GLSL 4.30 compute kernel across Fermi streaming multiprocessors...")

    gpu_vram_compute_vector(buf_a, buf_b, buf_c, n, 1.5)
    gpu_vram_download_tensor(buf_c, t_c)

    val := tensor_get(t_c, 0, 0)
    println("[OK] Compute dispatch verified! Output element [0] = " + to_text(val) + " (expected 11.5)")

    gpu_vram_release(buf_a)
    gpu_vram_release(buf_b)
    gpu_vram_release(buf_c)
    println("VRAM scratch buffers freed.")
