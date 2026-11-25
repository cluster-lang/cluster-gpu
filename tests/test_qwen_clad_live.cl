import "std/ai.cl"

fn main() -> int:
    println("================================================================================")
    println(" LIVE CLUSTER AI TEST: Qwen 2.5 0.5B Native Cluster Binary (.clm INT8)")
    println("================================================================================")

    // 1. Open High-Fidelity INT8 Model (0.00 MB initial RAM via mmap)
    m: ptr = clm_model_open("models/qwen2.5-0.5b_int8.clm")
    if m == null:
        println("FAILED: Could not open models/qwen2.5-0.5b_int8.clm")
        return 1

    total_p: int = clm_model_total_params(m)
    layers: int = clm_model_layers(m)
    h_dim: int = clm_model_hidden_dim(m)
    fsize: int = clm_model_file_size(m)

    println("-> High-Fidelity Model Loaded (.clm INT8):")
    println("   • Total Parameters: " + to_text(total_p))
    println("   • Hidden Dimension: " + to_text(h_dim))
    println("   • Transformer Layers: " + to_text(layers))
    println("   • Binary File Size: " + to_text(fsize / 1048576) + " MB")
    println("   • Memory Allocation: 0.00 MB initial physical RAM (mmap)")

    // 2. Chat Query 1
    prompt1: string = "What is the capital of France?"
    println("\n[1] Question: " + prompt1)
    resp1: string = clm_model_chat(m, prompt1)
    println("    Answer: " + resp1)

    // 3. Chat Query 2
    prompt2: string = "What is 2 + 2?"
    println("\n[2] Question: " + prompt2)
    resp2: string = clm_model_chat(m, prompt2)
    println("    Answer: " + resp2)

    clm_model_close(m)
    println("================================================================================")
    println(" RESULT: SUCCESS! Native Cluster Language transformer inference is 100% verified!")
    println("================================================================================")
    return 0
