from utils.index import Index
from python import Python
import testing

# def simd2np(simd: SIMD) -> PythonObject:
#     np = Python.import_module("numpy")
#     n = simd.__len__()
#     out = np.zeros(n)
#     print(out)
#     for i in range(n):
#         out[i] = simd[i]
#     return out


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


# ------------------------- Conversions ------------------------- #
fn np2simd[
    n: Int, type: DType = DType.float64
](a: PythonObject) raises -> SIMD[type, n]:
    var o = SIMD[type, n]()
    for i in range(a.__len__()):
        o[i] = a[i].to_float64().cast[type]()
    return o


fn np2tensor[type: DType](a: PythonObject) raises -> Tensor[type]:
    let n: Int = a.size.__index__()
    var o = Tensor[type](n)
    for i in range(n):
        o[i] = a[i].to_float64().cast[type]()
    return o


fn np2tensor2d[
    type: DType
](a: PythonObject, owned n: Int = 0, owned m: Int = 0) raises -> Tensor[type]:
    if n == 0:
        n = a.shape[0].__index__()
    if m == 0:
        m = a.shape[1].__index__()
    var o = Tensor[type](n, m)
    for i in range(n):
        for j in range(m):
            o[Index(i, j)] = a[i][j].to_float64().cast[type]()
    return o


fn np2tensor2d_i64(npin: PythonObject) -> Tensor[DType.int64]:
    try:
        let np = Python.import_module("numpy")

        let rows = npin.shape[0].__index__()
        let cols = npin.shape[1].__index__()
        let image = Tensor[DType.int64](rows, cols)

        let in_pointer = Pointer(
            __mlir_op.`pop.index_to_pointer`[
                _type = __mlir_type[`!kgen.pointer<scalar<si64>>`]
            ](
                SIMD[DType.index, 1](
                    npin.__array_interface__["data"][0].__index__()
                ).value
            )
        )
        let out_pointer = Pointer(
            __mlir_op.`pop.index_to_pointer`[
                _type = __mlir_type[`!kgen.pointer<scalar<si64>>`]
            ](SIMD[DType.index, 1](image.data().__as_index()).value)
        )
        for row in range(rows):
            for col in range(cols):
                let index = row * cols + col
                out_pointer.store(index, in_pointer[index])

        return image
    except:
        print("Failed in new")
        return Tensor[DType.int64]()


fn np2tensor2d_f64(npin: PythonObject) -> Tensor[DType.float64]:
    try:
        let np = Python.import_module("numpy")

        let rows = npin.shape[0].__index__()
        let cols = npin.shape[1].__index__()
        let image = Tensor[DType.float64](rows, cols)

        let in_pointer = Pointer(
            __mlir_op.`pop.index_to_pointer`[
                _type = __mlir_type[`!kgen.pointer<scalar<f64>>`]
            ](
                SIMD[DType.index, 1](
                    npin.__array_interface__["data"][0].__index__()
                ).value
            )
        )
        let out_pointer = Pointer(
            __mlir_op.`pop.index_to_pointer`[
                _type = __mlir_type[`!kgen.pointer<scalar<f64>>`]
            ](SIMD[DType.index, 1](image.data().__as_index()).value)
        )
        for row in range(rows):
            for col in range(cols):
                let index = row * cols + col
                out_pointer.store(index, in_pointer[index])

        return image
    except:
        print("Failed in new")
        return Tensor[DType.float64]()


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
