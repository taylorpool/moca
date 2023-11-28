from src.set import IntSet
from src.util import assert_true


def test_basic() -> NoneType:
    print("# set basic")
    var x = IntSet(300)
    x.add(1)
    x.add(3)

    assert_true(x.contains(1), "Set contains element")
    assert_true(x.contains(3), "Set contains element")
    assert_true(~x.contains(2), "Set does not contain element")
    assert_true(~x.contains(4), "Set does not contain element")


def test_remove() -> NoneType:
    print("# set remove")
    var x = IntSet(300)
    x.add(1)
    x.add(3)

    x.remove(1)

    assert_true(x.size() == 1, "Set size is not 1")
    assert_true(x.elements[0] == 3, "Set size is not 1")


fn main() raises:
    test_basic()
    test_remove()
