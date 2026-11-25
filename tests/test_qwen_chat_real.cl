fn main():
    m := clm_open("models/qwen2.5-0.5b_int4.clm")
    if m == null:
        return
    ans := clm_chat(m, "Hello! Who are you?")
    println("Response: " + ans)
    clm_close(m)
