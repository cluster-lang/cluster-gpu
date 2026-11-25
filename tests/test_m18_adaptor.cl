import "std/ai.cl"

fn main() -> int:
    println("--- Testing Milestone 18 (Cluster Syntax): Adapter & 2-bit Depth Mode ---")

    // 1. Open 2-bit Base Model
    m: ptr = clm_model_open("tests/fixtures/tiny_2bit.clm")
    if m == null:
        println("FAIL: Failed to open 2-bit base model")
        return 1

    params: int = clm_model_total_params(m)
    tcount: int = clm_model_tensor_count(m)
    println("Params: " + to_text(params) + ", Tensors: " + to_text(tcount))

    // 2. Attach Adapter
    att_res: int = clm_model_attach_adaptor(m, "tests/fixtures/tiny_adapter.clad")
    if att_res != 0:
        println("FAIL: Could not attach adapter")
        clm_model_close(m)
        return 1

    has_ad: bool = clm_model_has_adaptor(m)
    rank: int = clm_model_adaptor_rank(m)
    is_depth: bool = clm_model_is_depth_mode(m)
    if has_ad and is_depth:
        println("Adapter verified active at rank " + to_text(rank))
    else:
        println("FAIL: Adapter state mismatch")
        clm_model_close(m)
        return 1

    // 3. Chat with Adapter (Depth Mode)
    resp: string = clm_model_chat(m, "Explain relativity")
    println("Chat Response: " + resp)

    // 4. Detach Adapter
    clm_model_detach_adaptor(m)
    has_after: bool = clm_model_has_adaptor(m)
    if has_after:
        println("FAIL: Adapter still present after detach")
        clm_model_close(m)
        return 1

    clm_model_close(m)
    println("PASS: Cluster syntax test_m18_adaptor.cl verified!")
    return 0
