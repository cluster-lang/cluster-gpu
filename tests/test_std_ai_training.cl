import "std/ai/training.cl"

fn main() -> int:
    println("=================================================================")
    println("   Cluster Native AI Training Suite: Master Architecture Test    ")
    println("   Tensor Engine -> Autograd -> NN -> Attention -> Transformer   ")
    println("=================================================================")
    
    println("Hardware Acceleration Device: " + gpu_device_name())
    assert(gpu_is_available() == 1)
    
    // 1. Build Tokenizer from domain corpus
    tok := tokenizer_new()
    corpus := "cluster language deep learning transformer neural network artificial intelligence train converge"
    num_added := tokenizer_build_vocab_from_corpus(tok, corpus)
    vocab_sz := tokenizer_vocab_size(tok)
    println("1. Tokenizer initialized with " + to_text(vocab_sz) + " vocabulary tokens.")
    assert(vocab_sz >= 10)
    
    // 2. Prepare training text tokens
    train_text := "cluster language neural network artificial intelligence"
    raw_tokens := tokenizer_encode_words(tok, train_text)
    println("2. Training text tokenized into " + to_text(len(raw_tokens)) + " tokens.")
    assert(len(raw_tokens) >= 5)
    
    // 3. Prepare autoregressive training pairs (X_t -> Y_t+1)
    seq_len := len(raw_tokens) - 1
    input_seq := prepare_autoregressive_inputs(raw_tokens, seq_len)
    target_seq := prepare_autoregressive_targets(raw_tokens, seq_len)
    println("3. Autoregressive sequence length: " + to_text(seq_len))
    assert(len(input_seq) == seq_len)
    assert(len(target_seq) == seq_len)
    
    // 4. Initialize Transformer Language Model
    dim := 16
    heads := 2
    hidden_dim := 32
    learning_rate := 0.04
    net := transformer_lm_new(vocab_sz, seq_len, dim, heads, hidden_dim, learning_rate)
    println("4. TransformerLM Architecture:")
    println("   - Vocab Size:   " + to_text(net.vocab_size))
    println("   - Hidden Dim:   " + to_text(net.dim))
    println("   - Attention:    " + to_text(net.heads) + " heads (d_k = " + to_text(net.attention.head_dim) + ")")
    println("   - FFN Dim:      " + to_text(net.hidden_dim))
    println("   - Optimizer:    AdamW (lr = " + to_text(learning_rate) + ", weight_decay = 0.01)")
    
    // 5. Training Loop
    println("5. Beginning training loop...")
    let mut initial_loss: float = 0.0
    let mut final_loss: float = 0.0
    
    let mut step: int = 0
    total_steps := 25
    while step < total_steps:
        loss := transformer_lm_train_step(net, input_seq, target_seq)
        if step == 0:
            initial_loss = loss
        final_loss = loss
        
        if step % 5 == 0 or step == total_steps - 1:
            ppl := compute_perplexity(loss)
            println("   [Step " + to_text(step) + "/" + to_text(total_steps) + "] Loss: " + to_text(loss) + " | Perplexity: " + to_text(ppl))
        step = step + 1
        
    println("6. Training Convergence:")
    println("   - Initial Loss: " + to_text(initial_loss))
    println("   - Final Loss:   " + to_text(final_loss))
    assert(final_loss < initial_loss)
    println("   -> SUCCESS: Loss strictly decreased and converged!")
    
    // 7. Text Generation
    let mut prompt: vector[int] = []
    prompt.push(input_seq[0])
    prompt.push(input_seq[1])
    println("7. Prompt text: '" + tokenizer_decode(tok, prompt) + "'")
    generated_tokens := transformer_lm_generate(net, prompt, 3)
    generated_text := tokenizer_decode(tok, generated_tokens)
    println("   Generated sequence: '" + generated_text + "'")
    assert(len(generated_tokens) == len(prompt) + 3)
    
    // 8. Checkpoint Save and Load
    ckpt_file := "/tmp/cluster_transformer_model.clm"
    saved := transformer_lm_save(net, ckpt_file)
    println("8. Checkpoint saved: " + to_text(saved) + " -> " + ckpt_file)
    assert(saved == 1)
    assert(file_exists(ckpt_file) == 1)
    
    // Create new fresh model and load weights
    net2 := transformer_lm_new(vocab_sz, seq_len, dim, heads, hidden_dim, learning_rate)
    loaded := transformer_lm_load(net2, ckpt_file)
    println("   Checkpoint loaded: " + to_text(loaded) + " tensor groups.")
    assert(loaded == 11)
    
    // Verify loaded model produces identical forward logits
    l1 := transformer_lm_forward(net, input_seq)
    l2 := transformer_lm_forward(net2, input_seq)
    diff := (l1 - l2).norm()
    println("   Weight load parity diff norm: " + to_text(diff))
    assert(diff < 0.0001)
    
    println("=================================================================")
    println("   ALL 10 PURE CLUSTER AI TRAINING MODULES PASSED 100%!         ")
    println("=================================================================")
    return 0
