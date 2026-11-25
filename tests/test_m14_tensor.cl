// ============================================================================
// Cluster Language - Milestone 14: Native Tensors & SIMD Matrix Math Test Suite
// Verifies native tensor operations, @ matrix operator, and linear algebra
// ============================================================================

fn test_tensor_creation_and_accessors():
    put "--- Test 1: Creation and Dimension Accessors ---"
    t := tensor_zeros(4, 5)
    assert(t.rows() == 4, "rows must be 4")
    assert(t.cols() == 5, "cols must be 5")
    assert(t.size() == 20, "size must be 20")

    o := tensor_ones(3, 3)
    assert(o.rows() == 3, "ones rows must be 3")
    assert(o.cols() == 3, "ones cols must be 3")
    assert(o.get(0, 0) == 1.0, "ones(0,0) must be 1.0")
    assert(o.get(2, 2) == 1.0, "ones(2,2) must be 1.0")

    f := tensor_fill(2, 3, 42.5)
    assert(f.get(0, 0) == 42.5, "fill(0,0) must be 42.5")
    assert(f.get(1, 2) == 42.5, "fill(1,2) must be 42.5")
    put "Test 1 PASSED"

fn test_tensor_get_and_set():
    put "--- Test 2: Get and Set ---"
    t := tensor_zeros(2, 2)
    t.set(0, 0, 10.5)
    t.set(0, 1, 20.25)
    t.set(1, 0, 30.125)
    t.set(1, 1, 40.0)

    assert(t.get(0, 0) == 10.5, "get(0,0)")
    assert(t.get(0, 1) == 20.25, "get(0,1)")
    assert(t.get(1, 0) == 30.125, "get(1,0)")
    assert(t.get(1, 1) == 40.0, "get(1,1)")
    put "Test 2 PASSED"

fn test_matrix_multiplication():
    put "--- Test 3: Matrix Multiplication (@ and @=) ---"
    // [ [1, 2],
    //   [3, 4] ]
    a := tensor_zeros(2, 2)
    a.set(0, 0, 1.0)
    a.set(0, 1, 2.0)
    a.set(1, 0, 3.0)
    a.set(1, 1, 4.0)

    // [ [5, 6],
    //   [7, 8] ]
    b := tensor_zeros(2, 2)
    b.set(0, 0, 5.0)
    b.set(0, 1, 6.0)
    b.set(1, 0, 7.0)
    b.set(1, 1, 8.0)

    // c = a @ b:
    // [ [1*5 + 2*7, 1*6 + 2*8],    = [ [19, 22],
    //   [3*5 + 4*7, 3*6 + 4*8] ]       [43, 50] ]
    c := a @ b
    assert(c.rows() == 2, "c rows == 2")
    assert(c.cols() == 2, "c cols == 2")
    assert(c.get(0, 0) == 19.0, "c(0,0) == 19")
    assert(c.get(0, 1) == 22.0, "c(0,1) == 22")
    assert(c.get(1, 0) == 43.0, "c(1,0) == 43")
    assert(c.get(1, 1) == 50.0, "c(1,1) == 50")

    // Compound operator @=
    accum := tensor_ones(2, 2)
    multiplier := tensor_fill(2, 2, 2.0)
    accum @= multiplier
    // [1, 1] @ [2, 2] = [4, 4]
    // [1, 1]   [2, 2]   [4, 4]
    assert(accum.get(0, 0) == 4.0, "accum(0,0) == 4")
    assert(accum.get(1, 1) == 4.0, "accum(1,1) == 4")
    put "Test 3 PASSED"

fn test_elementwise_and_scalar_math():
    put "--- Test 4: Elementwise & Scalar Math ---"
    u := tensor_fill(2, 2, 8.0)
    v := tensor_fill(2, 2, 3.0)

    add_res := u + v
    assert(add_res.get(0, 0) == 11.0, "8 + 3 == 11")

    sub_res := u - v
    assert(sub_res.get(0, 0) == 5.0, "8 - 3 == 5")

    mul_res := u * v
    assert(mul_res.get(0, 0) == 24.0, "8 * 3 == 24")

    scale_res := u * 0.5
    assert(scale_res.get(0, 0) == 4.0, "8 * 0.5 == 4")
    put "Test 4 PASSED"

fn test_transpose_and_reductions():
    put "--- Test 5: Transpose & Reductions ---"
    // 2x3 matrix:
    // [ [1, 2, 3],
    //   [4, 5, 6] ]
    m := tensor_zeros(2, 3)
    m.set(0, 0, 1.0)
    m.set(0, 1, 2.0)
    m.set(0, 2, 3.0)
    m.set(1, 0, 4.0)
    m.set(1, 1, 5.0)
    m.set(1, 2, 6.0)

    // Sum reduction
    total := m.sum()
    assert(total == 21.0, "sum 1..6 == 21.0")

    // Transpose (3x2)
    mT := m.transpose()
    assert(mT.rows() == 3, "mT rows == 3")
    assert(mT.cols() == 2, "mT cols == 2")
    assert(mT.get(0, 0) == 1.0, "mT(0,0) == 1")
    assert(mT.get(1, 0) == 2.0, "mT(1,0) == 2")
    assert(mT.get(2, 0) == 3.0, "mT(2,0) == 3")
    assert(mT.get(0, 1) == 4.0, "mT(0,1) == 4")
    assert(mT.get(1, 1) == 5.0, "mT(1,1) == 5")
    assert(mT.get(2, 1) == 6.0, "mT(2,1) == 6")

    // Vector dot product
    v1 := tensor_fill(1, 3, 2.0)
    v2 := tensor_fill(1, 3, 3.0)
    d := v1.dot(v2)
    assert(d == 18.0, "dot product (2*3)*3 == 18.0")
    put "Test 5 PASSED"

fn test_tensor_print_and_put():
    put "--- Test 6: Display via put statement ---"
    p := tensor_zeros(2, 2)
    p.set(0, 0, 1.23)
    p.set(0, 1, 4.56)
    p.set(1, 0, 7.89)
    p.set(1, 1, 0.12)
    put p
    put "Test 6 PASSED"

fn main():
    put "=================================================="
    put "Running Milestone 14: Native Tensors Test Suite"
    put "=================================================="
    test_tensor_creation_and_accessors()
    test_tensor_get_and_set()
    test_matrix_multiplication()
    test_elementwise_and_scalar_math()
    test_transpose_and_reductions()
    test_tensor_print_and_put()
    put "=================================================="
    put "ALL MILESTONE 14 TESTS PASSED SUCCESSFULLY!"
    put "=================================================="
