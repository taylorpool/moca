from python import Python
from utils.index import Index


fn np2tensor[type: DType]( a:PythonObject) raises -> Tensor[type]:
    let n : Int = a.size.__index__()
    var o = Tensor[type](n)
    for i in range(n):
        o[i] = a[i].to_float64().cast[type]()
    return o

# TODO: Speed this up, it's SLOW. If we can read directly from binary BLOB to tensor, we can avoid numpy altogether
fn np2tensor2d[type: DType]( a:PythonObject, owned n: Int = 0, owned m: Int = 0) raises -> Tensor[type]:
    if n == 0:
        n = a.shape[0].__index__()
    if m == 0:
        m = a.shape[1].__index__()
    var o = Tensor[type](n, m)
    for i in range(n):
        for j in range(m):
            o[Index(i,j)] = a[i][j].to_float64().cast[type]()
    return o