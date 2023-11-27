import src.moca as mc


@value
@register_passable("trivial")
struct Landmark:
    var val: mc.Vector3d

    @always_inline
    @staticmethod
    fn identity() -> Self:
        return Self {val: mc.Vector3d(0, 0, 0, 0)}

    fn __invert__(self) -> Self:
        return Self(-self.val)

    fn inv(self) -> Self:
        return ~self

    @always_inline
    fn __add__(self, other: mc.Vector3d) -> Self:
        """Not really addition, just syntatic sugar for "retract" operation.
        Adds in an vector from the tangent space.
        """
        return Self {val: self.val + other}
