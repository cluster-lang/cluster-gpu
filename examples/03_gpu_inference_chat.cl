// Example 3: Running Real-time Autoregressive Neural Inference on GT 610 VRAM
// Uses built-in CLM neural engine compiled directly by Cluster.

fn main():
    println("=======================================================")
    println(" Cluster GPU: Autoregressive Neural QA on GT 610")
    println("=======================================================")

    model_path := "models/qwen2.5-0.5b_int4.clm"
    println("Opening model: " + model_path)
    m := clm_open(model_path)
    if m == null:
        println("[ERROR] Could not open model at " + model_path)
        return

    prompt := "What is Cluster Language and how does it unlock old GPUs?"
    println("Prompt: '" + prompt + "'")
    println("Generating response...")

    response := clm_chat(m, prompt)
    println("\n[QWEN 2.5 RESPONSE]:\n" + response)

    clm_close(m)
    println("\nInference complete. Context released.")
