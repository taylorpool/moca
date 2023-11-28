from src.dict import IntIntDict
from src.util import assert_true


def test_basic() -> NoneType:
    print("# dict basic")
    var x = IntIntDict()
    x[1] = 2
    x.add(3, 4)

    assert_true(x[1] == 2, "Dictionary missing element")
    assert_true(x[3] == 4, "Dictionary missing element")
    assert_true(x.contains(1), "Dictionary contains element")
    assert_true(x.contains(3), "Dictionary contains element")

    assert_true(~x.contains(2), "Dictionary does not contain element")
    assert_true(~x.contains(4), "Dictionary does not contain element")


def test_remove() -> NoneType:
    print("# dict remove")
    var x = IntIntDict()
    x[1] = 2
    x[3] = 4
    x.remove(1)
    assert_true(x.size() == 1, "Dictionary size is not 1")
    assert_true(x.keys[0] == 3, "Dictionary size is not 1")


fn main() raises:
    test_basic()
    test_remove()
