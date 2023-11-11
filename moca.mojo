from tensor import Tensor, TensorShape, TensorSpec
from utils.index import Index
from algorithm import vectorize_unroll
import math


fn elementwise_add[
    dtype: DType
](lhs: Tensor[dtype], rhs: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](lhs.shape())

    @parameter
    fn _add[nelts: Int](index: Int):
        result[index] = lhs[index] + rhs[index]

    vectorize_unroll[simdwidthof[dtype](), 2, _add](lhs.num_elements())

    return result


fn elementwise_subtract[
    dtype: DType
](lhs: Tensor[dtype], rhs: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](lhs.shape())

    @parameter
    fn _sub[nelts: Int](index: Int):
        result[index] = lhs[index] - rhs[index]

    vectorize_unroll[simdwidthof[dtype](), 2, _sub](lhs.num_elements())

    return result


fn elementwise_multiply[
    dtype: DType
](lhs: Tensor[dtype], rhs: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](lhs.shape())

    @parameter
    fn _mul[nelts: Int](index: Int):
        result[index] = lhs[index] * rhs[index]

    vectorize_unroll[simdwidthof[dtype](), 2, _mul](lhs.num_elements())

    return result


fn elementwise_divide[
    dtype: DType
](lhs: Tensor[dtype], rhs: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](lhs.shape())

    @parameter
    fn _div[nelts: Int](index: Int):
        result[index] = lhs[index] * rhs[index]

    vectorize_unroll[simdwidthof[dtype](), 2, _div](lhs.num_elements())

    return result


fn matrix_matrix_multiply[
    type: DType
](lhs: Tensor[type], rhs: Tensor[type]) -> Tensor[type]:
    let m = lhs.shape()[0]
    let n = rhs.shape()[1]
    let p = rhs.shape()[0]
    var result = Tensor[type](m, n)
    for i in range(m):
        for j in range(n):
            let i_j = Index(i, j)
            for k in range(p):
                result[i_j] += lhs[i, k] * rhs[k, j]

    return result


fn matrix_vector_multiply[
    type: DType
](mat: Tensor[type], vec: Tensor[type]) -> Tensor[type]:
    let m = mat.shape()[0]
    let n = vec.shape()[0]
    var result = Tensor[type](m)
    for i in range(m):
        for j in range(n):
            result[i] += mat[i, j] * vec[j]

    return result


fn forward_substitution_solve[
    type: DType
](L: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let n = L.shape()[0]
    var x = b
    for i in range(n):
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]

    return x


fn back_substitution_solve[
    type: DType
](U: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let n = U.shape()[0]
    var x = b
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    return x


fn llt_decompose[type: DType](A: Tensor[type]) -> Tensor[type]:
    let n = A.shape()[0]
    var L = Tensor[type](A.shape())
    for i in range(n):
        for j in range(i):
            let i_j = Index(i, j)
            L[i_j] = A[i_j]
            for k in range(j):
                L[i_j] -= L[i, k] * L[j, k]
            L[i_j] /= L[j, j]

        let i_i = Index(i, i)
        L[i_i] = A[i_i]
        for k in range(i):
            L[i_i] -= L[i, k] * L[i, k]
        L[i_i] = math.sqrt(L[i_i])

    return L


fn uut_decompose[type: DType](A: Tensor[type]) -> Tensor[type]:
    let n = A.shape()[0]
    var U = Tensor[type](A.shape())
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            let i_j = Index(i, j)
            U[i_j] = A[i_j]
            for k in range(n - 1, j, -1):
                U[i_j] -= U[i, k] * U[j, k]
            U[i_j] /= U[j, j]

        let i_i = Index(i, i)
        U[i_i] = A[i_i]
        for k in range(i + 1, n):
            U[i_i] -= U[i, k] * U[i, k]
        U[i_i] = math.sqrt(U[i, i])

    return U


@value
struct QR[type: DType]:
    var Q: Tensor[type]
    var R: Tensor[type]


fn qr_reduced_decompose[type: DType](A: Tensor[type]) -> QR[type]:
    let m = A.shape()[0]
    let n = A.shape()[1]
    var qr = QR[type](Tensor[type](m, n), Tensor[type](n, n))

    return qr


fn qr_full_decompose[type: DType](A: Tensor[type]) -> QR[type]:
    let m = A.shape()[0]
    let n = A.shape()[1]
    var qr = QR[type](Tensor[type](m, m), A)
    for i in range(m):
        qr.Q[Index(i, i)] = 1.0

    for k in range(n):
        # Compute u
        let u_dim = n - k
        var u = Tensor[type](u_dim)
        var v: SIMD[type, 1] = 0
        u[0] = qr.R[k, k]
        for i in range(1, u_dim):
            u[i] = qr.R[i + k, k]
            v += u[i] * u[i]
        if u[0] >= 0:
            u[0] += math.sqrt(v + u[0] * u[0])
        else:
            u[0] -= math.sqrt(v + u[0] * u[0])
        let u_norm = math.sqrt(v + u[0] * u[0])
        for i in range(u_dim):
            u[i] /= u_norm

        # Modify R
        for j in range(k, n):
            var zr: SIMD[type, 1] = 0
            for i in range(u_dim):
                zr += u[i] * qr.R[k + i, j]
            for i in range(k, m):
                let i_j = Index(i, j)
                qr.R[i_j] -= 2 * u[i - k] * zr

        # Modify Q
        for j in range(k, m):
            var zq: SIMD[type, 1] = 0
            for i in range(u_dim):
                zq += u[i] * qr.Q[k + i, j]
            for i in range(k, m):
                let i_j = Index(i, j)
                qr.Q[i_j] -= 2 * u[i - k] * zq

    return qr


fn qr_decompose[type: DType](A: Tensor[type], mode: StringRef = "reduced") -> QR[type]:
    if mode == "reduced":
        return qr_reduced_decompose(A)
    elif mode == "full":
        return qr_full_decompose(A)
    else:
        return qr_reduced_decompose(A)


alias Vector2d = SIMD[DType.float64, 2]
alias Vector3d = SIMD[DType.float64, 4]
alias Vector4d = SIMD[DType.float64, 4]
