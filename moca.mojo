from tensor import Tensor, TensorShape, TensorSpec
from utils.index import Index
from algorithm import vectorize_unroll

fn add[dtype: DType](lhs: Tensor[dtype], rhs: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](lhs.shape())

    @parameter
    fn _add[nelts: Int](index: Int):
        result[index] = lhs[index] + rhs[index]

    vectorize_unroll[simdwidthof[dtype](), 2, _add](lhs.num_elements())

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


fn divide[dtype: DType](lhs: Tensor[dtype], rhs: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](lhs.shape())

    @parameter
    fn _div[nelts: Int](index: Int):
        result[index] = lhs[index] * rhs[index]

    vectorize_unroll[simdwidthof[dtype](), 2, _div](lhs.num_elements())

    return result


fn dot[dtype: DType](lhs: Tensor[dtype], rhs: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](lhs.shape()[0], rhs.shape()[1])
    for i in range(lhs.shape()[0]):
        for j in range(rhs.shape()[1]):
            let index = Index(i, j)
            result[index] = 0
            for k in range(lhs.shape()[1]):
                result[index] += lhs[i, k] * rhs[k, j]

    return result


fn forward_substitution_solve[type: DType](L: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let n = L.shape()[0]
    var x = b
    for i in range(n):
        for j in range(i):
            x[i] -= L[i,j]*x[j]
        x[i] /= L[i,i]

    return x

fn back_substitution_solve[type: DType](U: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    let n = U.shape()[0]
    var x = b
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            x[i] -= U[i,j]*x[j]
        x[i] /= U[i,i]
    return x

alias Vector2d = SIMD[DType.float64, 2]
alias Vector3d = SIMD[DType.float64, 4]
alias Vector4d = SIMD[DType.float64, 4]
