from tensor import Tensor, TensorShape, TensorSpec
from utils.index import Index
from algorithm import vectorize_unroll
import math
import random
from memory import memset_zero, memset


# ------------------------- EASTON IMPLENTATIONS - UNTESTED ------------------------- #
fn rand_rows[type: DType](t: Tensor[type], num_rows: Int) -> Tensor[type]:
    let idx = Tensor[DType.uint64](10)
    random.randint(idx.data(), num_rows, 0, t.dim(0))
    return index[type](t, idx)


fn index[type: DType](t: Tensor[type], rows: Tensor[DType.uint64]) -> Tensor[type]:
    var out = Tensor[type](rows.num_elements(), t.dim(1))
    for i in range(rows.num_elements()):
        for j in range(out.dim(1)):
            out[Index(i, j)] = t[Index(rows[i].to_int(), j)]

    return out


fn get_row[type: DType, n: Int](t: Tensor[type], row: Int) -> SIMD[type, n]:
    return t.simd_load[n](row * t.dim(1))


fn get_row[type: DType](t: Tensor[type], row: Int) -> Tensor[type]:
    var out = Tensor[type](t.dim(1))
    for i in range(t.dim(1)):
        out[i] = t[Index(row, i)]
    return out


fn set_row[
    type: DType, n: Int
](inout t: Tensor[DType.float64], row: Int, insert: SIMD[DType.float64, n]):
    t.simd_store(t.dim(1) * row, insert)


fn argmax(t: Tensor) -> Int:
    var m = t[0]
    var idx = 0
    for i in range(1, t.num_elements()):
        if t[i] > m:
            m = t[i]
            idx = i

    return idx


# ------------------------- MOCA ------------------------- #


fn squared_norm[type: DType](x: Tensor[type]) -> SIMD[type, 1]:
    var result = SIMD[type, 1](0)
    let ptr = x.data()

    for i in range(x.num_elements()):
        result += ptr.load(i) ** 2

    return result


fn norm[type: DType](x: Tensor[type]) -> SIMD[type, 1]:
    return math.sqrt(squared_norm(x))


fn arange[type: DType = DType.int64](n: Int) -> Tensor[type]:
    var result = Tensor[type](n)
    for i in range(n):
        result[i] = i
    return result


fn arange[type: DType, n: Int]() -> SIMD[type, n]:
    var result = SIMD[type, n]()
    for i in range(n):
        result[i] = i
    return result


fn eye[type: DType = DType.float64](n: Int) -> Tensor[type]:
    var result = Tensor[type](n, n)

    memset_zero(result.data(), n * n)
    for i in range(n):
        result[Index(i, i)] = 1

    return result


fn zeros[type: DType = DType.float64](*dims: Int) -> Tensor[type]:
    let result = Tensor[type](dims)
    memset_zero(result.data(), result.num_elements())
    return result


fn zeros_like[type: DType](A: Tensor[type]) -> Tensor[type]:
    let Z = Tensor[type](A.shape())
    memset_zero(Z.data(), Z.num_elements())
    return Z


fn ones[type: DType = DType.float64](*dims: Int) -> Tensor[type]:
    let result = Tensor[type](dims)
    let ptr = result.data()
    for i in range(result.num_elements()):
        ptr.simd_store[1](i, 1)
    return result


fn ones_like[type: DType](A: Tensor[type]) -> Tensor[type]:
    let O = Tensor[type](A.shape())
    let ptr = O.data()
    for i in range(O.num_elements()):
        ptr.simd_store[1](i, 1)
    return O


fn swap[type: DType](inout A: Tensor[type], i: StaticIntTuple, j: StaticIntTuple):
    let tmp = A[i]
    A[i] = A[j]
    A[j] = tmp


fn add[dtype: DType](lhs: Tensor[dtype], rhs: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](lhs.shape())

    @parameter
    fn _add[nelts: Int](index: Int):
        result[index] = lhs[index] + rhs[index]

    vectorize_unroll[simdwidthof[dtype](), 2, _add](lhs.num_elements())

    return result


fn add[type: DType](lhs: SIMD[type, 1], rhs: Tensor[type]) -> Tensor[type]:
    let result = rhs
    let ptr = result.data()
    for i in range(result.num_elements()):
        ptr.store(i, lhs + ptr.load(i))
    return result


fn subtract[dtype: DType](lhs: Tensor[dtype], rhs: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](lhs.shape())

    @parameter
    fn _sub[nelts: Int](index: Int):
        result[index] = lhs[index] - rhs[index]

    vectorize_unroll[simdwidthof[dtype](), 2, _sub](lhs.num_elements())

    return result


fn multiply[dtype: DType](lhs: Tensor[dtype], rhs: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](lhs.shape())

    @parameter
    fn _mul[nelts: Int](index: Int):
        result[index] = lhs[index] * rhs[index]

    vectorize_unroll[simdwidthof[dtype](), 2, _mul](lhs.num_elements())

    return result


fn multiply[type: DType](lhs: SIMD[type, 1], rhs: Tensor[type]) -> Tensor[type]:
    var result = rhs
    let result_data = result.data()
    for i in range(result.num_elements()):
        result.simd_store(i, lhs * result.simd_load[1](i))
    return result


fn divide[dtype: DType](lhs: Tensor[dtype], rhs: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](lhs.shape())

    @parameter
    fn _div[nelts: Int](index: Int):
        result[index] = lhs[index] * rhs[index]

    vectorize_unroll[simdwidthof[dtype](), 2, _div](lhs.num_elements())

    return result


fn divide[type: DType](lhs: Tensor[type], rhs: SIMD[type, 1]) -> Tensor[type]:
    var result = lhs
    for i in range(lhs.shape()[0]):
        result[i] /= rhs

    return result


fn mat_mat[type: DType](lhs: Tensor[type], rhs: Tensor[type]) -> Tensor[type]:
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


fn matT_mat[type: DType](lhs: Tensor[type], rhs: Tensor[type]) -> Tensor[type]:
    let m = lhs.shape()[1]
    let n = rhs.shape()[1]
    let r = lhs.shape()[0]
    var result = Tensor[type](m, n)
    for i in range(m):
        for j in range(n):
            let i_j = Index(i, j)
            result[i_j] = 0
            for k in range(r):
                result[i_j] += lhs[k, i] * rhs[k, j]
    return result


fn mat_matT[type: DType](lhs: Tensor[type], rhs: Tensor[type]) -> Tensor[type]:
    let m = lhs.shape()[0]
    let n = rhs.shape()[0]
    let r = lhs.shape()[1]
    var result = Tensor[type](m, n)
    for i in range(m):
        for j in range(n):
            let i_j = Index(i, j)
            for k in range(r):
                result[i_j] += lhs[i, k] * rhs[j, k]

    return result


fn matT_matT[type: DType](lhs: Tensor[type], rhs: Tensor[type]) -> Tensor[type]:
    let m = lhs.shape()[1]
    let n = rhs.shape()[0]
    let r = lhs.shape()[0]
    var result = Tensor[type](m, n)
    for i in range(m):
        for j in range(n):
            let i_j = Index(i, j)
            result[i_j] = 0
            for k in range(r):
                result[i_j] += lhs[k, i] * rhs[j, k]
    return result


fn mat_vec[type: DType](mat: Tensor[type], vec: Tensor[type]) -> Tensor[type]:
    let m = mat.shape()[0]
    let n = vec.shape()[0]
    var result = Tensor[type](m)
    for i in range(m):
        for j in range(n):
            result[i] += mat[i, j] * vec[j]

    return result

fn mat_vec[
    type: DType, dim: Int
](mat: Tensor[type], vec: SIMD[type, dim]) -> SIMD[type, dim]:
    let m = mat.shape()[0]
    let n = mat.shape()[1]
    var result = SIMD[type, dim](0)
    for i in range(m):
        for j in range(n):
            result[i] += mat[i, j] * vec[j]

    return result


fn matT_vec[type: DType](mat: Tensor[type], vec: Tensor[type]) -> Tensor[type]:
    let m = mat.shape()[1]
    let n = mat.shape()[0]
    var result = Tensor[type](m)
    for j in range(n):
        for i in range(m):
            result[i] += mat[j, i] * vec[j]

    return result


fn matT_vec[type: DType, n: Int](mat: Tensor[type], vec: SIMD[type, n]) -> Tensor[type]:
    let m = mat.shape()[1]
    var result = Tensor[type](m)
    memset_zero(result.data(), m)
    for j in range(n):
        for i in range(m):
            result[i] += mat[j, i] * vec[j]

    return result


fn vecT_vec[type: DType](x: Tensor[type], y: Tensor[type]) -> SIMD[type, 1]:
    let n = x.shape()[0]
    var result = SIMD[type, 1](0)
    for i in range(n):
        result += x[i] * y[i]

    return result


fn vec_vecT[type: DType](x: Tensor[type], y: Tensor[type]) -> Tensor[type]:
    let n = x.shape()[0]
    var result = Tensor[type](n, n)
    for i in range(n):
        for j in range(n):
            result[Index(i, j)] = x[i] * x[j]

    return result


fn vecT_mat_vec[
    type: DType
](x: Tensor[type], A: Tensor[type], y: Tensor[type]) -> SIMD[type, 1]:
    let m = x.shape()[0]
    let n = A.shape()[1]

    var result = SIMD[type, 1](0)

    for i in range(m):
        var elem = SIMD[type, 1](0)
        for j in range(n):
            elem += A[i, j] * y[j]
        result += x[i] * elem

    return result

fn vecT_mat_vec[
    type: DType, dim: Int
](x: SIMD[type, dim], A: Tensor[type], y: SIMD[type, dim]) -> SIMD[type, 1]:
    let m = A.shape()[0]
    let n = A.shape()[1]

    var result = SIMD[type, 1](0)

    for i in range(m):
        var elem = SIMD[type, 1](0)
        for j in range(n):
            elem += A[i, j] * y[j]
        result += x[i] * elem

    return result


fn inv3[type:DType](matrix: Tensor[type]) -> Tensor[type]:
    var result = Tensor[type](3, 3)
    let det = matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]) -
              matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0]) +
              matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])

    if det == 0:
        # Matrix is not invertible
        return result

    let inv_det = 1.0 / det

    result[Index(0, 0)] = (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]) * inv_det
    result[Index(0, 1)] = (matrix[0, 2] * matrix[2, 1] - matrix[0, 1] * matrix[2, 2]) * inv_det
    result[Index(0, 2)] = (matrix[0, 1] * matrix[1, 2] - matrix[0, 2] * matrix[1, 1]) * inv_det
    result[Index(1, 0)] = (matrix[1, 2] * matrix[2, 0] - matrix[1, 0] * matrix[2, 2]) * inv_det
    result[Index(1, 1)] = (matrix[0, 0] * matrix[2, 2] - matrix[0, 2] * matrix[2, 0]) * inv_det
    result[Index(1, 2)] = (matrix[0, 2] * matrix[1, 0] - matrix[0, 0] * matrix[1, 2]) * inv_det
    result[Index(2, 0)] = (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0]) * inv_det
    result[Index(2, 1)] = (matrix[0, 1] * matrix[2, 0] - matrix[0, 0] * matrix[2, 1]) * inv_det
    result[Index(2, 2)] = (matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]) * inv_det

    return result

fn solve_from_top_left[type: DType](A: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let n = A.shape()[0]
    var x = Tensor[type](n)
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i]

    return x


fn solveT_from_top_left[type: DType](At: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let n = At.shape()[0]
    var x = Tensor[type](n)
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= At[j, i] * x[j]
        x[i] /= At[i, i]

    return x


fn solve_from_top_right[type: DType](A: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let n = A.shape()[0]
    var x = Tensor[type](n)
    for i in range(n):
        let index = n - i - 1
        x[index] = b[i]
        for j in range(n - i, n):
            x[index] -= A[i, j] * x[j]
        x[index] /= A[i, index]

    return x


fn solve_from_bottom_left[
    type: DType
](A: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let n = A.shape()[0]
    var x = Tensor[type](n)
    for i in range(n):
        let diff = n - 1 - i
        x[i] = b[diff]
        for j in range(i):
            x[i] -= A[diff, j] * x[j]
        x[i] /= A[diff, i]

    return x


fn solve_from_bottom_right[
    type: DType
](A: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let n = A.shape()[0]
    var x = Tensor[type](n)
    for i in range(n):
        let diff = n - 1 - i
        x[diff] = b[diff]
        for j in range(n - i, n):
            x[diff] -= A[diff, j] * x[j]
        x[diff] /= A[diff, diff]

    return x


fn solveT_from_bottom_right[
    type: DType
](At: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let n = At.shape()[0]
    var x = Tensor[type](n)
    for i in range(n):
        let diff = n - 1 - i
        x[diff] = b[diff]
        for j in range(n - i, n):
            x[diff] -= At[j, diff] * x[j]
        x[diff] /= At[diff, diff]

    return x


fn forward_substitution_solve_permuted[
    type: DType
](L: Tensor[type], b: Tensor[type], row_order: Tensor[DType.int64]) -> Tensor[type]:
    let m = L.shape()[0]
    let n = L.shape()[1]
    let diff = m - n
    var x = Tensor[type](n)
    for i in range(n):
        let i_diff = i + diff
        x[i] = b[row_order[i_diff].to_int()]
        for j in range(i):
            x[i] -= L[i_diff, j] * x[j]
        x[i] /= L[i_diff, i]

    return x


fn copy_row[type: DType](A: Tensor[type], row: Int, start_col: Int = 0) -> Tensor[type]:
    let x = Tensor[type](A.shape()[1] - start_col)
    var ptr = x.data()
    for j in range(start_col, A.shape()[1]):
        ptr.simd_store(A[row, j])
        ptr += 1
    return x


fn copy_col[type: DType](A: Tensor[type], col: Int, start_row: Int = 0) -> Tensor[type]:
    let x = Tensor[type](A.shape()[0] - start_row)
    var ptr = x.data()
    for i in range(start_row, A.shape()[0]):
        ptr.simd_store(A[i, col])
        ptr += 1
    return x


fn transpose[type: DType](A: Tensor[type]) -> Tensor[type]:
    var At = Tensor[A.dtype](A.shape()[1], A.shape()[0])
    for i in range(At.shape()[0]):
        for j in range(At.shape()[1]):
            At[Index(i, j)] = A[j, i]

    return At


@value
struct LU[type: DType]:
    var row_order: Tensor[DType.int64]
    var L: Tensor[type]
    var U: Tensor[type]

    fn solve(self, b: Tensor[type]) -> Tensor[type]:
        let y = forward_substitution_solve_permuted(self.L, b, self.row_order)
        let x = solve_from_bottom_left(self.U, y)
        return x


fn lu_factor[type: DType](A: Tensor[type]) -> LU[type]:
    let n = A.shape()[0]
    var lu = LU[type](arange(n), eye[type](n), A)

    for k in range(n):
        var i = k
        var max_value = math.abs(lu.U[i, k])
        for j in range(k, n):
            let value = math.abs(lu.U[i, k])
            if value > max_value:
                i = j
                max_value = value
        for j in range(k, n):
            swap(lu.U, Index(k, j), Index(i, j))
        for j in range(k):
            swap(lu.L, Index(k, j), Index(i, j))
        swap(lu.row_order, Index(k), Index(i))
        for j in range(k + 1, n):
            lu.L[Index(j, k)] = lu.U[Index(j, k)] / lu.U[k, k]
        for j in range(k + 1, n):
            for s in range(k, n):
                lu.U[Index(j, s)] -= lu.L[Index(j, k)] * lu.U[Index(s, k)]

    return lu


@value
struct LLT[type: DType]:
    var L: Tensor[type]

    fn solve(self, b: Tensor[type]) -> Tensor[type]:
        let y = solve_from_top_left(self.L, b)
        return solveT_from_bottom_right(self.L, y)


fn llt_factor[type: DType](A: Tensor[type]) -> LLT[type]:
    let n = A.shape()[0]
    var llt = LLT(zeros_like(A))
    for i in range(n):
        for j in range(i):
            let i_j = Index(i, j)
            llt.L[i_j] = A[i_j]
            for k in range(j):
                llt.L[i_j] -= llt.L[i, k] * llt.L[j, k]
            llt.L[i_j] /= llt.L[j, j]

        let i_i = Index(i, i)
        llt.L[i_i] = A[i_i]
        for k in range(i):
            llt.L[i_i] -= llt.L[i, k] * llt.L[i, k]
        llt.L[i_i] = math.sqrt(llt.L[i_i])

    return llt


@value
struct UUT[type: DType]:
    var U: Tensor[type]

    fn solve(self, b: Tensor[type]) -> Tensor[type]:
        let y = solve_from_bottom_right(self.U, b)
        return solveT_from_top_left(self.U, y)


fn uut_factor[type: DType](A: Tensor[type]) -> UUT[type]:
    let n = A.shape()[0]
    var uut = UUT(zeros_like(A))
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            let i_j = Index(i, j)
            uut.U[i_j] = A[i_j]
            for k in range(n - 1, j, -1):
                uut.U[i_j] -= uut.U[i, k] * uut.U[j, k]
            uut.U[i_j] /= uut.U[j, j]

        let i_i = Index(i, i)
        uut.U[i_i] = A[i_i]
        for k in range(i + 1, n):
            uut.U[i_i] -= uut.U[i, k] * uut.U[i, k]
        uut.U[i_i] = math.sqrt(uut.U[i, i])

    return uut


@value
struct EigenPair[type: DType]:
    var val: SIMD[type, 1]
    var vec: Tensor[type]


fn power_method[
    type: DType
](
    A: Tensor[type],
    eigen0: EigenPair[type],
    max_iters: Int = 100,
    absTol: SIMD[type, 1] = 1e-12,
    relTol: SIMD[type, 1] = 1e-12,
) -> EigenPair[type]:
    var eigen = eigen0
    for i in range(max_iters):
        let y = mat_vec(A, eigen.vec)
        let old_eigen = eigen
        eigen.vec = divide(y, norm(y))
        eigen.val = vecT_mat_vec(eigen.vec, A, eigen.vec)

        let absoluteError = squared_norm(
            subtract(multiply(eigen.val, eigen.vec), mat_vec(A, eigen.vec))
        )
        let relativeDiff = squared_norm(subtract(eigen.vec, old_eigen.vec)) + (
            eigen.val - old_eigen.val
        ) ** 2
        if absoluteError < absTol or relativeDiff < relTol:
            break

    return eigen


fn shifted_lu_power_method[
    type: DType
](
    A: Tensor[type],
    target_eigval: SIMD[type, 1],
    eigen0: EigenPair[type],
    max_iters: Int = 100,
    absTol: SIMD[type, 1] = 1e-12,
    relTol: SIMD[type, 1] = 1e-12,
) -> EigenPair[type]:
    let lu = lu_factor(subtract(A, multiply(target_eigval, eye[A.dtype](A.shape()[0]))))
    var eigen = eigen0
    for i in range(max_iters):
        let y = lu.solve(eigen.vec)
        let old_eigen = eigen
        eigen.vec = divide(y, norm(y))
        eigen.val = vecT_mat_vec(eigen.vec, A, eigen.vec)

        let absoluteError = squared_norm(
            subtract(multiply(eigen.val, eigen.vec), mat_vec(A, eigen.vec))
        )
        let relativeDiff = squared_norm(subtract(eigen.vec, old_eigen.vec)) + (
            eigen.val - old_eigen.val
        ) ** 2
        if absoluteError < absTol or relativeDiff < relTol:
            break

    return eigen


@value
struct QR[type: DType]:
    var row_order: Tensor[DType.int64]
    var Q: Tensor[type]
    var R: Tensor[type]

    fn solve(self, b: Tensor[type]) -> Tensor[type]:
        let y = matT_vec(self.Q, b)
        return solve_from_bottom_left(self.R, y)


fn qr_factor[type: DType](A: Tensor[type]) -> QR[type]:
    let m = A.shape()[0]
    let n = A.shape()[1]
    var qr = QR[type](arange(m), eye[type](m), A)

    for k in range(n):
        # Compute u
        var u = copy_col(qr.R, k, k)
        u[0] += math.copysign(norm(u), u[0])
        u = divide(u, norm(u))
        let u_dim = u.shape()[0]

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

    qr.Q = transpose(qr.Q)

    return qr


fn householder_redirect[type: DType](inout u: Tensor[type]):
    u[0] += math.copysign(norm(u), u[0])
    u = divide(u, norm(u))


@value
struct Hessenberg[type: DType]:
    """A = QHQT."""

    var H: Tensor[type]
    var Q: Tensor[type]


fn hessenberg_factor[type: DType](A: Tensor[type]) -> Hessenberg[type]:
    let m = A.shape()[0]
    let n = A.shape()[1]
    var hessenberg = Hessenberg(A, eye[type](m))
    for k in range(n - 2):
        let kp1 = k + 1
        var u = copy_col(hessenberg.H, k, start_row=kp1)
        householder_redirect(u)

        for j in range(k, hessenberg.H.shape()[1]):
            var v: SIMD[type, 1] = 0
            for i in range(u.shape()[0]):
                v += u[i] * hessenberg.H[i + kp1, j]
            v *= 2
            for i in range(u.shape()[0]):
                hessenberg.H[Index(i + kp1, j)] -= u[i] * v

        for i in range(m):
            var v: SIMD[type, 1] = 0
            for j in range(u.shape()[0]):
                v += hessenberg.H[i, j + kp1] * u[j]
            v *= 2
            for j in range(u.shape()[0]):
                hessenberg.H[Index(i, j + kp1)] -= v * u[j]

        for j in range(m):
            var v: SIMD[type, 1] = 0
            for i in range(u.shape()[0]):
                v += u[i] * hessenberg.Q[kp1 + i]
            v *= 2
            for i in range(u.shape()[0]):
                hessenberg.Q[Index(i + kp1, j)] -= v * u[i]

    hessenberg.Q = transpose(hessenberg.Q)

    return hessenberg


fn qr_iteration[type: DType](A0: Tensor[type], max_iters: Int = 100):
    var A = A0
    for i in range(max_iters):
        let qr = qr_factor(A)
        A = mat_vec(qr.R, qr.Q)


fn qr_from_hessenberg[type: DType](H: Tensor[type]) -> QR[type]:
    let m = H.shape()[0]
    let n = H.shape()[1]
    var qr = QR(arange(H.shape()[0]), eye[type](m), H)
    for j in range(math.min(n, m)):
        let i = j + 1
        let im1 = i - 1
        let a = qr.R[im1, j]
        let b = qr.R[i, j]
        var G = Tensor[type](2, 2)
        let ab_norm = math.sqrt(a**2 + b**2)
        G[Index(0, 0)] = a / ab_norm
        G[Index(0, 1)] = b / ab_norm
        G[Index(1, 0)] = -G[0, 1]
        G[Index(1, 1)] = G[0, 0]
        for k in range(i - 1, i + 2):
            for r in range(j, qr.R.shape()[1]):
                qr.R[Index(k, r)] = G[k, 0] * qr.R[0, r] + G[k, 1] * qr.R[1, r]
        for k in range(i - 1, i + 2):
            for r in range(i + 2):
                qr.Q[Index(k, r)] = G[k, 0] * qr.Q[0, r] + G[k, 1] * qr.Q[1, r]

    qr.Q = transpose(qr.Q)

    return qr


fn solve_homogeneous_equation[
    type: DType
](A: Tensor[type], x0: Tensor[type]) -> Tensor[type]:
    let m = A.shape()[0]
    let n = A.shape()[1]
    let np1 = n + 1
    var count = 0
    let max_count = 12
    var x = x0
    var lambd: SIMD[type, 1] = 0.0
    let AtA = matT_mat(A, A)

    while count < max_count:
        let lambd2 = 2 * lambd
        var grad_L = Tensor[type](np1)
        let grad_1 = mat_vec(AtA, x)
        for i in range(n):
            grad_L[i] = grad_1[i] + lambd2 * x[i]
        grad_L[n] = vecT_vec(x, x) - 1.0

        var D2L = Tensor[type](np1, np1)
        for i in range(n):
            for j in range(n):
                let i_j = Index(i, j)
                D2L[i_j] = AtA[i_j]
                if i == j:
                    D2L[i_j] += lambd2

            let i_n = Index(i, n)
            D2L[i_n] = 2 * x[i]
            let n_i = Index(n, i)
            D2L[n_i] = D2L[i_n]
        D2L[Index(n, n)] = 1000

        print(AtA)
        print(grad_L)
        print(D2L)

        let step = llt_factor(D2L).solve(grad_L)
        print(step)

        for i in range(x.shape()[0]):
            x[i] -= step[i]

        lambd -= step[x.shape()[0]]

        print(x)
        print(lambd)

        count += 1

    return x


fn diag[type: DType](v: Tensor[type]) -> Tensor[type]:
    let n = v.shape()[0]
    if v.shape().num_elements() == 1:
        var result = Tensor[type](n, n)
        memset_zero(result.data(), n * n)
        for i in range(n):
            result[Index(i, i)] = v[i]
        return result
    var result = Tensor[type](n)
    for i in range(n):
        result[i] = v[i, i]
    return result


# @value
# struct SVD[type: DType]:
#     var U: Tensor[type]
#     var s: Tensor[type]
#     var V: Tensor[type]


# fn svd_factor[type: DType](A: Tensor[type]) -> SVD[type]:
#     let m = A.shape()[0]
#     let n = A.shape()[1]
#     let result: SVD[type]
#     result.U = eye[type](m)
#     result.V = eye[type](n)

#     var B = A

#     for _ in range(100):
#         let qr = qr_full_decompose(A)
#         B = matrix_matrix(qr.R, qr.Q)
#         result.U = matrix_matrix(result.U, qr.Q)
#         result.V = matrix_matrix_transpose(result.V, qr.Q)

#         if squared_norm(subtract(A, diag(diag(A)))) < 1e-9:
#             break

#     result.s = diag(A)

#     return result


# fn dlt[type: DType](x: Tensor[type], x_prime: Tensor[type]) -> Tensor[type]:
#     let num_correspondences = x.shape()[0]

#     var A = Tensor[type](2 * num_correspondences, 9)
#     for i in range(num_correspondences):
#         let x_prime_x = multiply(x_prime[i, 0], x)
#         let y_prime_x = multiply(x_prime[i, 1], x)
#         let w_prime_x = multiply(x_prime[i, 2], x)

#         let i2 = 2 * i
#         A[Index(i2, 0)] = 0
#         A[Index(i2, 1)] = 0
#         A[Index(i2, 2)] = 0
#         A[Index(i2, 3)] = -w_prime_x[0]
#         A[Index(i2, 4)] = -w_prime_x[1]
#         A[Index(i2, 5)] = -w_prime_x[2]
#         A[Index(i2, 6)] = y_prime_x[0]
#         A[Index(i2, 7)] = y_prime_x[1]
#         A[Index(i2, 8)] = y_prime_x[2]

#         let i2p1 = i2 + 1
#         A[Index(i2p1, 0)] = w_prime_x[0]
#         A[Index(i2p1, 1)] = w_prime_x[1]
#         A[Index(i2p1, 2)] = w_prime_x[2]
#         A[Index(i2p1, 3)] = 0
#         A[Index(i2p1, 4)] = 0
#         A[Index(i2p1, 5)] = 0
#         A[Index(i2p1, 6)] = -x_prime_x[0]
#         A[Index(i2p1, 7)] = -x_prime_x[1]
#         A[Index(i2p1, 8)] = -x_prime_x[2]

# let svd = compute_svd(A)


alias Vector2d = SIMD[DType.float64, 2]
alias Vector3d = SIMD[DType.float64, 4]
alias Vector4d = SIMD[DType.float64, 4]
alias Vector5d = SIMD[DType.float64, 8]
alias Vector6d = SIMD[DType.float64, 8]
alias Vector7d = SIMD[DType.float64, 8]
alias Vector8d = SIMD[DType.float64, 8]
alias Vector9d = SIMD[DType.float64, 16]
alias Vector10d = SIMD[DType.float64, 16]
alias Vector11d = SIMD[DType.float64, 16]
alias Vector12d = SIMD[DType.float64, 16]
alias Vector13d = SIMD[DType.float64, 16]
alias Vector14d = SIMD[DType.float64, 16]
alias Vector15d = SIMD[DType.float64, 16]
alias Vector16d = SIMD[DType.float64, 16]
