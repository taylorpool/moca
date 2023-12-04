from utils.index import Index
from memory import memcpy
from python import Python


# ------------------------- Conversions ------------------------- #


fn pyfloat[type: DType](i: PythonObject) -> SIMD[type, 1]:
    return i.to_float64().cast[type]()


fn pyint(i: PythonObject) -> Int:
    return i.__index__()


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
        np.set_printoptions(5, None, 1000, 1000, True)

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

        if npin.dtype != np.int64:
            raise Error("np2tensor2d_f64: input is not float64")

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

        if npin.dtype != np.float64:
            raise Error("np2tensor2d_f64: input is not float64")

        return image
    except:
        print("Failed in new")
        return Tensor[DType.float64]()


fn np2tensor1d_f64(npin: PythonObject) -> Tensor[DType.float64]:
    try:
        let np = Python.import_module("numpy")

        let num = npin.shape[0].__index__()
        let image = Tensor[DType.float64](num)

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
        for row in range(num):
            out_pointer.store(row, in_pointer[row])

        if npin.dtype != np.float64:
            raise Error("np2tensor2d_f64: input is not float64")

        return image
    except:
        print("Failed in new")
        return Tensor[DType.float64]()


fn tensor2np(A: Tensor[DType.float64]) -> PythonObject:
    try:
        let np = Python.import_module("numpy")

        let shape = Python.evaluate("[]")
        for i in range(A.rank()):
            _ = shape.append(A.dim(i).__index__())

        let A_np = np.empty(shape, np.float64)

        let in_pointer = Pointer(
            __mlir_op.`pop.index_to_pointer`[
                _type = __mlir_type[`!kgen.pointer<scalar<f64>>`]
            ](SIMD[DType.index, 1](A.data().__as_index()).value)
        )

        let out_pointer = Pointer(
            __mlir_op.`pop.index_to_pointer`[
                _type = __mlir_type[`!kgen.pointer<scalar<f64>>`]
            ](
                SIMD[DType.index, 1](
                    A_np.__array_interface__["data"][0].__index__()
                ).value
            )
        )

        memcpy(out_pointer, in_pointer, A.num_elements())

        if A_np.dtype != np.float64:
            raise Error("tensor2np: input is not float64")

        return A_np
    except:
        print("Failed in new")
        return PythonObject()


@value
struct SVDResult[type: DType = DType.float64]:
    var u: Tensor[type]
    var s: Tensor[type]
    var vh: Tensor[type]


fn svd(A: Tensor[DType.float64]) -> SVDResult:
    try:
        let np = Python.import_module("numpy")
        let A_np = tensor2np(A)
        let result_np = np.linalg.svd(A_np)
        let result = SVDResult(
            np2tensor2d_f64(result_np.U),
            np2tensor1d_f64(result_np.S),
            np2tensor2d_f64(result_np.Vh),
        )
        return result
    except:
        print("Failed in SVD")
        let t = Tensor[DType.float64](10, 10)
        return SVDResult(t, t, t)
