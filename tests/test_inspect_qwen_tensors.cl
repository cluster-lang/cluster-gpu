// Inspect Qwen 0.5B tensors & metadata via Cluster runtime

import "std/tensor.cl"
import "std/ai/model.cl"

fn main():
    path := "models/qwen2.5-0.5b-instruct-q4_0.gguf"
    m := model_open(path)
    if m == 0:
        println("[ERROR] Failed to open Qwen 0.5B model")
        return
    
    count := model_tensor_count(m)
    total := model_total_params(m)
    println("Qwen 0.5B Tensor Count: " + to_text(count))
    println("Total Parameters: " + to_text(total))
    
    // Print first 25 tensors
    let mut i: int = 0
    while i < 25 and i < count:
        tname := model_tensor_name(m, i)
        rows := model_tensor_rows(m, i)
        cols := model_tensor_cols(m, i)
        dt := model_tensor_dtype(m, i)
        println("  Tensor[" + to_text(i) + "]: " + tname + " shape=[" + to_text(rows) + ", " + to_text(cols) + "] dtype=" + to_text(dt))
        i = i + 1
        
    model_close(m)
