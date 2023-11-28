struct Set[type: AnyType, hash: fn (type) -> Int]:
    var n: Int
    var elements: DynamicVector[type]
    var exists: Tensor[DType.bool]

    fn __init__(inout self, n: Int):
        self.n = n
        self.elements = DynamicVector[type](n)
        self.exists = Tensor[DType.bool](n)

    fn add(inout self, x: type):
        let e = hash(x)
        if self.exists[e]:
            return
        else:
            self.elements.push_back(x)
            self.exists[e] = True

    fn contains(inout self, x: type) -> Bool:
        let e = hash(x)
        return self.exists[e]

    fn size(self) -> Int:
        return self.elements.__len__()

    fn remove(inout self, x: type):
        let e = hash(x)
        if not self.exists[e]:
            return

        self.exists[e] = False

        # Find where in keys it's stored
        var idx: Int = 0
        for i in range(self.elements.__len__()):
            if hash(self.elements[i]) == e:
                idx = i
                break

        self.elements[idx] = self.elements[self.elements.__len__() - 1]
        let blank = self.elements.pop_back()

    fn __str__(self: Self) -> String:
        var s: String = "{"
        for i in range(self.elements.__len__()):
            s += String(hash(self.elements[i]))
            if i != self.elements.__len__() - 1:
                s += ", "
        s += "}"
        return s

    fn __repr__(self: Self) -> String:
        return self.__str__()


# ------------------------- An integer set ------------------------- #
fn id(x: Int) -> Int:
    return x


alias IntSet = Set[Int, id]
