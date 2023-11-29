import moca as mc

from testing import assert_almost_equal, assert_equal
from utils.index import Index


fn test_arange() raises:
    print("# test_arange")
    let x = mc.arange(3)
    for i in range(x.shape()[0]):
        _ = assert_equal(i, x[i])


fn test_eye() raises:
    print("# test_eye")
    let A = mc.eye(3)
    for i in range(A.shape()[0]):
        for j in range(A.shape()[1]):
            if i == j:
                _ = assert_equal(A[i, j], 1.0)
            else:
                _ = assert_equal(A[i, j], 0.0)


fn test_zeros() raises:
    print("# test_zeros")
    let x = mc.zeros(3)
    for i in range(x.shape()[0]):
        _ = assert_equal(x[i], 0.0)


fn test_zeros2() raises:
    print("# test_zeros2")
    let A = mc.zeros(2, 2)
    _ = assert_equal(A.shape()[0], 2)
    _ = assert_equal(A.shape()[1], 2)
    for i in range(A.shape()[0]):
        for j in range(A.shape()[1]):
            _ = assert_equal(A[i, j], 0.0)


fn test_zeros_like() raises:
    print("# test_zeros_like")
    let x = mc.arange(4)
    let y = mc.zeros_like(x)
    for i in range(y.shape()[0]):
        _ = assert_equal(y[i], 0)

    _ = assert_equal(x.shape()[0], y.shape()[0])


fn test_ones() raises:
    print("# test_ones")
    let x = mc.ones(3)
    for i in range(x.shape()[0]):
        _ = assert_equal(x[i], 1.0)


fn test_ones2() raises:
    print("# test_ones2")
    let A = mc.ones(2, 2)
    _ = assert_equal(A.shape()[0], 2)
    _ = assert_equal(A.shape()[1], 2)
    for i in range(A.shape()[0]):
        for j in range(A.shape()[1]):
            _ = assert_equal(A[i, j], 1.0)


fn test_ones_like() raises:
    print("# test_ones_like")
    let x = mc.arange(4)
    let y = mc.ones_like(x)
    for i in range(y.shape()[0]):
        _ = assert_equal(y[i], 1)

    _ = assert_equal(x.shape()[0], y.shape()[0])


fn test_swap() raises:
    print("# test_swap")
    var x = mc.arange(2)
    mc.swap(x, Index(0), Index(1))
    _ = assert_equal(x[0], 1)
    _ = assert_equal(x[1], 0)


fn test_add() raises:
    print("# test_add")
    let x = mc.arange(3)
    let y = mc.add(x, x)
    for i in range(x.shape()[0]):
        _ = assert_almost_equal(2 * x[i], y[i])


fn test_add2() raises:
    print("# test_add2")
    let x = mc.arange(2)
    let y = x[0]
    let z = mc.add(y, x)
    for i in range(x.shape()[0]):
        _ = assert_equal(y + x[i], z[i])


fn test_subtract() raises:
    print("# test_subtract")
    let x = mc.arange(2)
    let y = mc.subtract(x, x)
    for i in range(x.shape()[0]):
        _ = assert_equal(y[i], x[i] - x[i])


fn test_multiply() raises:
    print("# test_multiply")
    let x = mc.arange(2)
    let y = mc.add(Int64(3), x)
    let z = mc.multiply(x, y)
    for i in range(x.shape()[0]):
        _ = assert_almost_equal(x[i] * y[i], z[i])


fn test_multiply2() raises:
    print("# test_multiply2")
    let x = mc.arange(2)
    let y: Int64 = 2
    let z = mc.multiply(y, x)
    for i in range(x.shape()[0]):
        _ = assert_almost_equal(x[i] * y, z[i])


fn test_mat_mat() raises:
    print("# test_mat_mat")
    var A = Tensor[DType.float64](1, 2)
    A[Index(0, 0)] = 1.0
    A[Index(0, 1)] = 2.0

    var B = Tensor[DType.float64](2, 3)
    B[Index(0, 0)] = 3.0
    B[Index(0, 1)] = 4.0
    B[Index(0, 2)] = 5.0
    B[Index(1, 0)] = 6.0
    B[Index(1, 1)] = 7.0
    B[Index(1, 2)] = 8.0

    let C = mc.mat_mat(A, B)
    _ = assert_equal(C.shape()[0], A.shape()[0])
    _ = assert_equal(C.shape()[1], B.shape()[1])

    for i in range(C.shape()[0]):
        for j in range(C.shape()[1]):
            _ = assert_equal(C[Index(i, j)], A[i, 0] * B[0, j] + A[i, 1] * B[1, j])


fn test_matT_mat() raises:
    print("# test_matT_mat")
    var A = Tensor[DType.float64](2, 1)
    A[Index(0, 0)] = 1.0
    A[Index(1, 0)] = 2.0

    var B = Tensor[DType.float64](2, 3)
    B[Index(0, 0)] = 3.0
    B[Index(0, 1)] = 4.0
    B[Index(0, 2)] = 5.0
    B[Index(1, 0)] = 6.0
    B[Index(1, 1)] = 7.0
    B[Index(1, 2)] = 8.0

    let C = mc.matT_mat(A, B)
    _ = assert_equal(C.shape()[0], A.shape()[1])
    _ = assert_equal(C.shape()[1], B.shape()[1])

    for i in range(C.shape()[0]):
        for j in range(C.shape()[1]):
            _ = assert_equal(C[Index(i, j)], A[0, i] * B[0, j] + A[1, i] * B[1, j])


fn test_mat_matT() raises:
    print("# test_mat_matT")
    var A = Tensor[DType.float64](1, 2)
    A[Index(0, 0)] = 1.0
    A[Index(0, 1)] = 2.0

    var B = Tensor[DType.float64](3, 2)
    B[Index(0, 0)] = 3.0
    B[Index(0, 1)] = 4.0
    B[Index(1, 0)] = 5.0
    B[Index(1, 1)] = 6.0
    B[Index(2, 0)] = 7.0
    B[Index(2, 1)] = 8.0

    let C = mc.mat_matT(A, B)
    _ = assert_equal(C.shape()[0], A.shape()[0])
    _ = assert_equal(C.shape()[1], B.shape()[0])

    for i in range(C.shape()[0]):
        for j in range(C.shape()[1]):
            _ = assert_equal(C[Index(i, j)], A[i, 0] * B[j, 0] + A[i, 1] * B[j, 1])


fn test_mat_vec() raises:
    print("# test_mat_vec")
    var A = Tensor[DType.float64](2, 3)
    A[Index(0, 0)] = 1.0
    A[Index(0, 1)] = 2.0
    A[Index(0, 2)] = 3.0
    A[Index(1, 0)] = 4.0
    A[Index(1, 1)] = 5.0
    A[Index(1, 2)] = 6.0

    var x = Tensor[DType.float64](3)
    x[Index(0)] = 7.0
    x[Index(1)] = 8.0
    x[Index(2)] = 9.0

    let b = mc.mat_vec(A, x)
    _ = assert_equal(b.shape()[0], A.shape()[0])

    for i in range(b.shape()[0]):
        _ = assert_equal(b[i], A[i, 0] * x[0] + A[i, 1] * x[1] + A[i, 2] * x[2])


fn test_matT_vec() raises:
    print("# test_matT_vec")
    var A = Tensor[DType.float64](3, 2)
    A[Index(0, 0)] = 1.0
    A[Index(0, 1)] = 2.0
    A[Index(1, 0)] = 3.0
    A[Index(1, 1)] = 4.0
    A[Index(2, 0)] = 5.0
    A[Index(2, 1)] = 6.0

    var x = Tensor[DType.float64](3)
    x[Index(0)] = 7.0
    x[Index(1)] = 8.0
    x[Index(2)] = 9.0

    let b = mc.matT_vec(A, x)
    _ = assert_equal(b.shape()[0], A.shape()[1])

    for i in range(b.shape()[0]):
        _ = assert_equal(b[i], A[0, i] * x[0] + A[1, i] * x[1] + A[2, i] * x[2])


fn test_matT_vec2() raises:
    print("# test_matT_vec")
    var A = Tensor[DType.float64](4, 2)
    A[Index(0, 0)] = 1.0
    A[Index(0, 1)] = 2.0
    A[Index(1, 0)] = 3.0
    A[Index(1, 1)] = 4.0
    A[Index(2, 0)] = 5.0
    A[Index(2, 1)] = 6.0
    A[Index(3, 0)] = 5.0
    A[Index(3, 1)] = 6.0

    let x = SIMD[DType.float64, 4](7, 8, 9, 10)

    let b = mc.matT_vec(A, x)
    _ = assert_equal(b.shape()[0], A.shape()[1])

    for i in range(b.shape()[0]):
        _ = assert_equal(
            b[i], A[0, i] * x[0] + A[1, i] * x[1] + A[2, i] * x[2] + A[3, i] * x[3]
        )


fn test_vecT_vec() raises:
    print("# test_vecT_vec")
    let x = mc.arange(4)

    let xtx = mc.vecT_vec(x, x)
    _ = assert_equal(xtx, 1 + 4 + 9)


fn test_vec_vecT() raises:
    print("# test_vec_vecT")
    let x = mc.arange(4)

    let xxt = mc.vec_vecT(x, x)

    _ = assert_equal(xxt.shape()[0], x.shape()[0])
    _ = assert_equal(xxt.shape()[1], x.shape()[0])

    for i in range(xxt.shape()[0]):
        for j in range(xxt.shape()[1]):
            _ = assert_equal(xxt[i, j], x[i] * x[j])


fn test_vecT_mat_vec() raises:
    print("# test_vecT_mat_vec")
    let x = mc.arange(2)
    var A = Tensor[x.dtype](2, 1)
    A[Index(0, 0)] = 0
    A[Index(1, 0)] = 1
    let y = mc.arange(1)

    let result = mc.vecT_mat_vec(x, A, y)

    _ = assert_equal(result, 0)


fn test_forward_substition_solve() raises:
    print("# test_forward_substitution_solve")
    var L = Tensor[DType.float64](2, 2)
    L[Index(0, 0)] = 1.0
    L[Index(1, 0)] = 2.0
    L[Index(1, 1)] = 3.0

    var b = Tensor[DType.float64](L.shape()[0])
    b[0] = -1.0
    b[1] = -2.0

    let x = mc.forward_substitution_solve(L, b)
    _ = assert_equal(x.shape()[0], L.shape()[1])

    let y = mc.mat_vec(L, x)
    _ = assert_equal(y.shape()[0], b.shape()[0])

    for i in range(y.shape()[0]):
        _ = assert_almost_equal(y[i], b[i])


fn test_forward_substitution_solve_permuted() raises:
    print("# test_forward_substitution_solve_permuted")
    var L = Tensor[DType.float64](2, 2)
    L[Index(0, 0)] = 1.0
    L[Index(1, 0)] = 2.0
    L[Index(1, 1)] = 3.0

    var b = Tensor[DType.float64](L.shape()[0])
    b[0] = -1.0
    b[1] = -2.0

    var row_order = Tensor[DType.int64](2)
    row_order[0] = 1
    row_order[1] = 0

    let x = mc.forward_substitution_solve_permuted(L, b, row_order)
    _ = assert_equal(x.shape()[0], L.shape()[1])

    let y = mc.mat_vec(L, x)
    _ = assert_equal(y.shape()[0], b.shape()[0])

    for i in range(y.shape()[0]):
        _ = assert_almost_equal(y[row_order[i].to_int()], b[i])


fn test_forward_substitution_solve_transpose() raises:
    print("# test_forward_substitution_solve_transpose")
    var U = Tensor[DType.float64](2, 2)
    U[Index(0, 0)] = 1.0
    U[Index(0, 1)] = 2.0
    U[Index(1, 0)] = 3.0

    var b = Tensor[DType.float64](U.shape()[0])
    b[0] = -1.0
    b[1] = -2.0


fn main() raises:
    _ = test_arange()
    _ = test_eye()
    _ = test_zeros()
    _ = test_zeros2()
    _ = test_zeros_like()
    _ = test_ones()
    _ = test_ones2()
    _ = test_ones_like()
    _ = test_swap()
    _ = test_add()
    _ = test_add2()
    _ = test_subtract()
    _ = test_multiply()
    _ = test_multiply2()
    _ = test_mat_mat()
    _ = test_matT_mat()
    _ = test_mat_matT()
    _ = test_mat_vec()
    _ = test_matT_vec()
    _ = test_matT_vec2()
    _ = test_vecT_vec()
    _ = test_vec_vecT()
    _ = test_vecT_mat_vec()
    _ = test_forward_substition_solve()
    _ = test_forward_substitution_solve_permuted()