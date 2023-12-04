from utils.index import Index
import testing

# ------------------------- Numerical Derivatives ------------------------- #
fn nder[
    n_out: Int, n_in: Int
](
    f: fn (eps: SIMD[DType.float64, n_in]) capturing -> SIMD[DType.float64, n_out],
    eps: Float64 = 1e-6,
) -> Tensor[DType.float64]:
    let x = SIMD[DType.float64, n_in](0)
    let fx = f(x)

    var d = Tensor[DType.float64](n_out, n_in)
    for i in range(n_in):
        var eps_vec = SIMD[DType.float64, n_in](0)
        eps_vec[i] = eps
        let diff = (f(eps_vec) - fx) / eps

        for j in range(n_out):
            d[Index(j, i)] = diff[j]

    return d

# ------------------------- Testing ------------------------- #


fn assert_true(cond: Bool, message: String) raises:
    if not testing.assert_true(cond, message):
        raise Error(message)


fn assert_almost_equal[
    type: DType, size: Int
](
    lhs: SIMD[type, size],
    rhs: SIMD[type, size],
    atol: SIMD[type, 1] = 0,
    rtol: SIMD[type, 1] = 0,
) raises:
    if atol != 0 or rtol != 0:
        if not testing.assert_almost_equal[type, size](lhs, rhs, atol, rtol):
            raise Error()
    else:
        if not testing.assert_almost_equal[type, size](lhs, rhs):
            raise Error()


fn assert_almost_equal_tensor[
    type: DType, n: Int
](lhs: Tensor[type], rhs: Tensor[type]) raises:
    if not testing.assert_true(lhs.shape() == rhs.shape()):
        raise Error("Not the same shape")

    var lhs_simd = lhs.data().simd_load[n](0)
    var rhs_simd = rhs.data().simd_load[n](0)

    for i in range(lhs.shape().num_elements(), n):
        lhs_simd[i] = 0
        rhs_simd[i] = 0

    assert_almost_equal[type, n](lhs_simd, rhs_simd)
