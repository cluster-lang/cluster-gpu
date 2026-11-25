fn main():
    prompt := str("Write a one-sentence news headline about AI compilers")
    put str("Prompt: ") + prompt
    response := ai_ask(prompt)
    put str("AI Response: ") + response
