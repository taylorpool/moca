from tensor import Tensor, TensorShape, TensorSpec
from utils.index import Index
from algorithm import vectorize_unroll


struct Vector[T: DType, size: Int]:
    var data: SIMD[T, size]

    fn __init__(inout self):
        self.data = SIMD[T,size]()

    fn __init__(inout self, data: SIMD[T, size]):
        self.data = data

    fn __getitem__(self, index: Int) -> SIMD[T, 1]:
        return self.data[index]

    fn __setitem__(inout self, index: Int, val: SIMD[T, 1]):
        self.data.__setitem__(index, val)

    fn __add__(self, rhs: Vector[T,size]) -> Vector[T, size]:
        return self.data + rhs.data

struct Matrix[T: DType, rows: Int, cols: Int]:
    var data: SIMD[T,rows*cols]

    fn __init__(inout self):
        self.data = SIMD[T,rows*cols]()

    fn __getitem__(self, row: Int, col: Int) -> SIMD[T,1]:
        return self.data[row*cols+col]

    fn __setitem__(inout self, row: Int, col: Int, val: SIMD[T,1]):
        self.data[row*cols+col] = val 

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


fn mul[dtype: DType](lhs: Tensor[dtype], rhs: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](lhs.shape())

    @parameter
    fn _mul[nelts: Int](index: Int):
        result[index] = lhs[index] * rhs[index]

    vectorize_unroll[simdwidthof[dtype](), 2, _mul](lhs.num_elements())

    return result


fn div[dtype: DType](lhs: Tensor[dtype], rhs: Tensor[dtype]) -> Tensor[dtype]:
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


fn print_matrix[dtype: DType](x: Tensor[dtype]):
    for i in range(x.shape()[0]):
        for j in range(x.shape()[1]):
            print_no_newline(x[i, j], " ")
        print()


def main():
    var x = Tensor[DType.float64](2, 2)
    x[Index(0, 0)] = 1.0
    x[Index(0, 1)] = 0.0
    x[Index(1, 0)] = 0.0
    x[Index(1, 1)] = 1.0
    print_matrix(x)
    let y = x
    print_matrix(y)
    let z = add(x, y)
    print("z")
    print_matrix(z)
    let zz = subtract(x, y)
    print("zz")
    print_matrix(zz)

    let aa = mul(x, y)
    print("mul")
    print_matrix(aa)

    let bb = div(x, y)
    print("div")
    print_matrix(bb)

    let cc = dot(x, y)
    print("dot")
    print_matrix(cc)

    var v = Vector[DType.float64, 3]()
    v[0] = 0
    v[1] = 1
    v[2] = 2
    print(v[0])
    print(v[1])

    var u = Vector[DType.float64, 3]()
    u[0] = 0
    u[1] = 1
    u[2] = 2
    let w = v + u
    print(w[0])
    print(w[1])
    print(w[2])


    var A = Matrix[DType.float64, 3, 3]()
    A[0,0] = 1
    A[0,1] = 0
    A[0,2] = 0
    A[1,0] = 0
    A[1,1] = 1
    A[1,2] = 0
    A[2,0] = 0
    A[2,1] = 0
    A[2,2] = 1
    print(A[2,2])
