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


fn arange[type: DType = DType.int64](n: Int) -> Tensor[type]:
    var result = Tensor[type](n)
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
    var result = Tensor[type](dims)
    memset_zero(result.data(), result.num_elements())
    return result


fn zeros_like[type: DType](A: Tensor[type]) -> Tensor[type]:
    var Z = Tensor[type](A.shape())
    memset_zero(Z.data(), Z.num_elements())
    return Z


fn ones[type: DType = DType.float64](*dims: Int) -> Tensor[type]:
    var result = Tensor[type](dims)
    let ptr = result.data()
    for i in range(result.num_elements()):
        ptr.simd_store[1](i, 1)
    return result


fn ones_like[type: DType](A: Tensor[type]) -> Tensor[type]:
    var O = Tensor[type](A.shape())
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
    var result = rhs
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


fn mat_vec[type: DType](mat: Tensor[type], vec: Tensor[type]) -> Tensor[type]:
    let m = mat.shape()[0]
    let n = vec.shape()[0]
    var result = Tensor[type](m)
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


fn sub_solve_top_left[type: DType](A: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let n = A.shape()[0]
    var x = Tensor[type](n)
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i]

    return x


fn sub_solve_top_right[type: DType](A: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let n = A.shape()[0]
    var x = Tensor[type](n)
    for i in range(n):
        let index = n - i - 1
        x[index] = b[i]
        for j in range(index + 1, n):
            x[index] -= A[i, j] * x[j]

        x[index] /= A[i, index]


fn forward_substitution_solve[
    type: DType
](L: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let m = L.shape()[0]
    let n = L.shape()[1]
    let diff = m - n
    var x = Tensor[type](n)
    for i in range(n):
        let i_diff = i + diff
        x[i] = b[i_diff]
        for j in range(i):
            x[i] -= L[i_diff, j] * x[j]
        x[i] /= L[i_diff, i]

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


fn forward_substitution_solve_transpose[
    type: DType
](U: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let m = b.shape()[0]
    let n = U.shape()[0]
    let diff = m - n
    var x = Tensor[type](n)
    for i in range(n):
        let i_diff = i + diff
        x[i] = b[i_diff]
        for j in range(i):
            x[i] -= U[j, i_diff] * x[j]
        x[i] /= U[i_diff, i_diff]

    return x


fn back_substitution_solve[
    type: DType
](U: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let m = U.shape()[0]
    let n = U.shape()[1]
    let diff = m - n
    var x = Tensor[type](n)  # b
    for i in range(n - 1, -1, -1):
        let i_diff = i - diff
        x[i] = b[i_diff]
        for j in range(i + 1, n):
            x[i] -= U[i_diff, j] * x[j]
        x[i] /= U[i_diff, i_diff]
    return x


fn back_substitution_solve_transpose[
    type: DType
](L: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let m = b.shape()[0]
    let n = L.shape()[0]
    let diff = m - n
    var x = Tensor[type](n)
    for i in range(n - 1, -1, -1):
        let i_diff = i - diff
        x[i] = b[i_diff]
        for j in range(i + 1, n):
            x[i] -= L[j, i_diff] * x[j]
        x[i] /= L[i_diff, i_diff]
    return x


@value
struct LU[type: DType]:
    var row_order: Tensor[DType.int64]
    var L: Tensor[type]
    var U: Tensor[type]

    fn solve(self, b: Tensor[type]) -> Tensor[type]:
        let y = forward_substitution_solve_permuted(self.L, b, self.row_order)
        let x = back_substitution_solve(self.U, y)
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
        let y = forward_substitution_solve(self.L, b)
        return back_substitution_solve_transpose(self.L, y)


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
        let y = back_substitution_solve(self.U, b)
        return forward_substitution_solve_transpose(self.U, y)


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
struct QR[type: DType]:
    var Q: Tensor[type]
    var R: Tensor[type]

    fn solve(self, b: Tensor[type]) -> Tensor[type]:
        let y = matT_vec(self.Q, b)
        return back_substitution_solve(self.R, y)


fn qr_factor[type: DType](A: Tensor[type]) -> QR[type]:
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
    if v.shape().num_elements() == 1:
        let n = v.shape()[0]
        var result = Tensor[type](n, n)
        memset_zero(result.data(), n * n)
        for i in range(n):
            result[Index(i, i)] = v[i]
        return result
    elif v.shape().num_elements() == 2:
        let n = v.shape()[0]
        var result = Tensor[type](n)
        for i in range(n):
            result[i] = v[i, i]
            return result


fn squared_norm[type: DType](x: Tensor[type]) -> SIMD[type, 1]:
    var result = SIMD[type, 1](0)
    let ptr = x.data()

    for i in range(x.num_elements()):
        result += ptr.load(i) ** 2

    return result


@value
struct SVD[type: DType]:
    var U: Tensor[type]
    var s: Tensor[type]
    var V: Tensor[type]


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


fn dlt[type: DType](x: Tensor[type], x_prime: Tensor[type]) -> Tensor[type]:
    let num_correspondences = x.shape()[0]

    var A = Tensor[type](2 * num_correspondences, 9)
    for i in range(num_correspondences):
        let x_prime_x = multiply(x_prime[i, 0], x)
        let y_prime_x = multiply(x_prime[i, 1], x)
        let w_prime_x = multiply(x_prime[i, 2], x)

        let i2 = 2 * i
        A[Index(i2, 0)] = 0
        A[Index(i2, 1)] = 0
        A[Index(i2, 2)] = 0
        A[Index(i2, 3)] = -w_prime_x[0]
        A[Index(i2, 4)] = -w_prime_x[1]
        A[Index(i2, 5)] = -w_prime_x[2]
        A[Index(i2, 6)] = y_prime_x[0]
        A[Index(i2, 7)] = y_prime_x[1]
        A[Index(i2, 8)] = y_prime_x[2]

        let i2p1 = i2 + 1
        A[Index(i2p1, 0)] = w_prime_x[0]
        A[Index(i2p1, 1)] = w_prime_x[1]
        A[Index(i2p1, 2)] = w_prime_x[2]
        A[Index(i2p1, 3)] = 0
        A[Index(i2p1, 4)] = 0
        A[Index(i2p1, 5)] = 0
        A[Index(i2p1, 6)] = -x_prime_x[0]
        A[Index(i2p1, 7)] = -x_prime_x[1]
        A[Index(i2p1, 8)] = -x_prime_x[2]

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
