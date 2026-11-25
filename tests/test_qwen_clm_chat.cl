// Test zk_clm_chat on models/qwen2.5-0.5b_int4.clm

extern fn zk_clm_open(path: string) -> ptr
extern fn zk_clm_close(m: ptr) -> void
extern fn zk_clm_chat(m: ptr, prompt: string) -> string

fn main():
    path := "models/qwen2.5-0.5b_int4.clm"
    println("Opening " + path + "...")
    m := zk_clm_open(path)
    if m == null:
        println("[ERROR] Failed to open CLM model")
        return
    println("Successfully opened model! Generating answer to 'What is Cluster Language?'...")
    ans := zk_clm_chat(m, "What is Cluster Language?")
    println("ANSWER: " + ans)
    zk_clm_close(m)
