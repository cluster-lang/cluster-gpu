fn main():
    // 1. Dynamic typing (any) opt-in
    x: any = 42
    put "x (int):"
    put x

    x = "Hello AI"
    put "x (string):"
    put x

    // 2. Static type promotion:
    // y is declared as any, but only ever assigned double, so it promotes to C++ double!
    y: any = 3.14
    put "y (promoted float/double):"
    put y

    // 3. Auto-imports:
    // Calling fs standard library functions without manual "import fs"
    fs.write("test_any.txt", "AI test data")
    content := fs.read("test_any.txt")
    put "Read from fs without import:"
    put content
    fs.remove("test_any.txt")
