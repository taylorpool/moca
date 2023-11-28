struct Dict[type_key: AnyType, type_value: DType, hash: fn (type_key) -> Int, n: Int]:
    var keys: DynamicVector[type_key]
    var values: Tensor[type_value]
    var exists: Tensor[DType.bool]

    fn __init__(inout self):
        self.keys = DynamicVector[type_key](n)
        self.values = Tensor[type_value](n)
        self.exists = Tensor[DType.bool](n)

    fn __getitem__(inout self, key: type_key) -> SIMD[type_value, 1]:
        let e = hash(key)
        if self.exists[e]:
            return self.values[e]
        else:
            return SIMD[type_value, 1]()

    fn __setitem__(inout self, key: type_key, value: SIMD[type_value, 1]):
        self.add(key, value)

    fn add(inout self, key: type_key, value: SIMD[type_value, 1]):
        let e = hash(key)
        if self.exists[e]:
            return
        else:
            self.keys.push_back(key)
            self.values[e] = value
            self.exists[e] = True

    fn contains(inout self, x: type_key) -> Bool:
        let e = hash(x)
        return self.exists[e]

    fn pop(inout self, x: type_key) -> SIMD[type_value, 1]:
        let e = self[x]
        self.remove(x)
        return e

    fn remove(inout self, x: type_key):
        let e = hash(x)
        if not self.exists[e]:
            return

        self.exists[e] = False

        # Find where in keys it's stored
        var idx: Int = 0
        for i in range(self.keys.__len__()):
            if hash(self.keys[i]) == e:
                idx = i
                break

        self.keys[idx] = self.keys[self.keys.__len__() - 1]
        let blank = self.keys.pop_back()

    fn size(inout self) -> Int:
        return self.keys.__len__()

    fn __copyinit__(inout self, existing: Self):
        self.keys = existing.keys
        self.values = existing.values
        self.exists = existing.exists


# ------------------------- Basic dictionary ------------------------- #
fn identity_hash(x: Int) -> Int:
    return x


alias IntIntDict = Dict[Int, DType.int64, identity_hash, 1024]


# ------------------------- An dict for camera pairs -> idx ------------------------- #
fn hash_pair[n: Int](x: Tuple[Int, Int]) -> Int:
    let id1 = x.get[0, Int]()
    let id2 = x.get[1, Int]()
    if id1 > id2:
        return n * id2 + id1
    else:
        return n * id1 + id2


alias TupleIntDict = Dict[Tuple[Int, Int], DType.int64, hash_pair[512], 512 * 512]
