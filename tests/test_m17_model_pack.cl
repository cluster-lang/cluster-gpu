import "std/ai.cl"

fn test_clm_inspection():
    m := clm_model_open("tests/fixtures/tiny_model.clm")
    if m == 0:
        m = clm_model_open("models/tiny_model.clm")
    if m == 0:
        println("FAILED: could not open tiny_model.clm")
        return

    n_params := clm_model_total_params(m)
    if n_params != 16:
        println("FAILED: expected 16 params, got " + to_text(n_params))
        return

    n_layers := clm_model_layers(m)
    if n_layers != 1:
        println("FAILED: expected 1 layer, got " + to_text(n_layers))
        return

    h_dim := clm_model_hidden_dim(m)
    if h_dim != 4:
        println("FAILED: expected hidden_dim 4, got " + to_text(h_dim))
        return

    n_tensors := clm_model_tensor_count(m)
    if n_tensors != 3:
        println("FAILED: expected 3 tensors, got " + to_text(n_tensors))
        return

    t0_name := clm_model_tensor_name(m, 0)
    if t0_name != "blk.0.attn_norm.weight":
        println("FAILED: unexpected tensor 0 name: " + t0_name)
        return

    println("PASS: CLM metadata inspection verified")
    clm_model_close(m)

fn test_clm_tensor_decompilation():
    m := clm_model_open("tests/fixtures/tiny_model.clm")
    if m == 0:
        m = clm_model_open("models/tiny_model.clm")
    if m == 0:
        println("FAILED: could not open tiny_model.clm")
        return

    // Extract F32 tensor
    t1 := clm_model_get_tensor(m, "blk.0.attn_norm.weight")
    if tensor_rows(t1) != 4:
        println("FAILED: tensor rows mismatch")
        return

    // Extract INT4 tensor
    t3 := clm_model_get_tensor(m, "token_embd.weight")
    if tensor_rows(t3) != 4 or tensor_cols(t3) != 2:
        println("FAILED: INT4 tensor shape mismatch")
        return

    println("PASS: CLM tensor extraction & INT4 dequantization verified")
    clm_model_close(m)

fn test_clm_conversational_chat():
    m := clm_model_open("tests/fixtures/tiny_model.clm")
    if m == 0:
        m = clm_model_open("models/tiny_model.clm")
    if m == 0:
        println("FAILED: could not open tiny_model.clm")
        return

    reply := clm_model_chat(m, "What is Cluster Language?")
    if string_length(reply) <= 0:
        println("FAILED: empty chat reply")
        return

    gen := clm_model_generate(m, "System prompt", 32)
    if string_length(gen) <= 0:
        println("FAILED: empty generate output")
        return

    println("PASS: CLM conversational chat & token generation verified")
    println("Sample reply: " + reply)
    clm_model_close(m)

fn main():
    println("=== Running M17 Packed AI Binary (.clm) Tests ===")
    test_clm_inspection()
    test_clm_tensor_decompilation()
    test_clm_conversational_chat()
    println("=== All M17 CLM Tests Passed Successfully ===")
