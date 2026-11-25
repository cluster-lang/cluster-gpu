import "std/clm.cl"

fn main() -> int:
    println("--- Testing std/clm.cl ---")
    m := cl_model_new("nano_transformer", "gpt_causal_lm")
    cl_model_set_param(m, "vocab_size", "256")
    cl_model_set_param(m, "d_model", "64")
    cl_model_set_param(m, "n_heads", "4")
    cl_model_set_param(m, "n_layers", "2")
    
    assert(cl_model_get_param(m, "vocab_size", "") == "256")
    assert(cl_model_get_param_int(m, "d_model", 0) == 64)
    assert(cl_model_get_param_int(m, "n_heads", 0) == 4)
    assert(cl_model_get_param_int(m, "n_layers", 0) == 2)
    
    t_tok_emb := tensor_zeros(4, 4)
    tensor_set(t_tok_emb, 0, 0, 1.25)
    tensor_set(t_tok_emb, 0, 1, -0.5)
    tensor_set(t_tok_emb, 1, 1, 2.75)
    tensor_set(t_tok_emb, 3, 3, 0.125)
    
    t_attn_q := tensor_zeros(4, 4)
    tensor_set(t_attn_q, 0, 0, 0.77)
    tensor_set(t_attn_q, 2, 1, -1.33)
    
    cl_model_add_tensor(m, "tok_emb.weight", t_tok_emb)
    cl_model_add_tensor(m, "layer0.attn_q.weight", t_attn_q)
    
    assert(cl_model_tensor_count(m) == 2)
    assert(cl_model_total_params(m) == 32)
    assert(cl_model_has_tensor(m, "tok_emb.weight") == 1)
    assert(cl_model_has_tensor(m, "missing.weight") == 0)
    
    path := "/tmp/test_std_unit_model.clm"
    res := cl_model_save(path, m)
    assert(res == 1)
    
    loaded := cl_model_load(path)
    assert(loaded.name == "nano_transformer")
    assert(loaded.arch == "gpt_causal_lm")
    assert(cl_model_get_param_int(loaded, "vocab_size", 0) == 256)
    assert(cl_model_get_param_int(loaded, "d_model", 0) == 64)
    assert(cl_model_tensor_count(loaded) == 2)
    assert(cl_model_total_params(loaded) == 32)
    
    t_loaded_emb := cl_model_get_tensor(loaded, "tok_emb.weight")
    assert(t_loaded_emb.rows() == 4)
    assert(t_loaded_emb.cols() == 4)
    let mut diff0: float = tensor_get(t_loaded_emb, 0, 0) - 1.25
    if diff0 < 0.0:
        diff0 = 0.0 - diff0
    assert(diff0 < 0.0001)
    
    println("PASSED: std/clm.cl works 100%!")
    return 0
