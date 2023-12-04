from src.dict import TupleIntDict
from src.set import IntSet
import src.moca as mc
import src.util as util
from utils.index import Index


@value
struct NextPair:
    var id1: Int
    var id2: Int
    var factor_idx: Tensor[DType.int64]


fn truncate(t: Tensor[DType.int64]) -> Tensor[DType.int64]:
    """Helper to remove -1 at end of all index tensors."""
    var idx = t.dim(0)
    for i in range(t.dim(0)):
        if t[i] == -1:
            idx = i
            break

    let out = Tensor[DType.int64](idx)

    let in_pointer = Pointer(
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<si64>>`]
        ](SIMD[DType.index, 1](t.data().__as_index()).value)
    )
    let out_pointer = Pointer(
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<si64>>`]
        ](SIMD[DType.index, 1](out.data().__as_index()).value)
    )
    for i in range(idx):
        out_pointer.store(i, in_pointer[i])

    return out


struct SceneGraph:
    # Maps from (img1, img2) to the pair index
    var id_pairs: TupleIntDict
    # Each row corresponds to a match, with the entries being the factor index #.
    var indices_pair: Tensor[DType.int64]
    var lms_per_pair: Tensor[DType.uint64]
    # Each row corresponds to a pose, with the entries being the factor index #
    # var indices_pose: Tensor[DType.int64]
    var lms_per_pose: Tensor[DType.uint64]
    # Each row corresponds to a lm, with the entries being the factor index #
    # var indices_lm: Tensor[DType.int64]

    fn __init__(inout self):
        self.id_pairs = TupleIntDict()
        self.indices_pair = Tensor[DType.int64](0)
        self.lms_per_pair = Tensor[DType.uint64](0)
        self.lms_per_pose = Tensor[DType.uint64](0)

    fn setup(
        inout self,
        id_pairs: TupleIntDict,
        indices_pair: Tensor[DType.int64],
        lms_per_pose: Tensor[DType.uint64],
    ) raises:
        # Track all the pairs
        self.id_pairs = id_pairs
        self.indices_pair = indices_pair
        self.lms_per_pose = lms_per_pose

        # Track how many landmarks each pair has
        self.lms_per_pair = Tensor[DType.uint64](self.indices_pair.dim(0))
        for i in range(self.indices_pair.dim(0)):
            for j in range(self.indices_pair.dim(1)):
                if self.indices_pair[i, j] == -1:
                    break
                self.lms_per_pair[i] += 1

    fn get_first_pair(inout self) -> NextPair:
        # Use the image with the most landmarks as the first image
        let id_pose = mc.argmax(self.lms_per_pose).__int__()

        # Find it's pair with the most factors and go from there
        var pair_max = 0
        var pair_id = 0
        var pair = (0, 0)
        for i in range(self.lms_per_pose.dim(0)):
            if self.id_pairs.contains((id_pose, i)):
                let id_pair = self.id_pairs[(id_pose, i)].__int__()
                let num_fc = self.lms_per_pair[id_pair].__int__()
                if num_fc > pair_max:
                    pair_max = num_fc
                    pair_id = id_pair
                    pair = (id_pose, i)

        self.id_pairs.remove(pair)

        return NextPair(
            pair.get[0, Int](),
            pair.get[1, Int](),
            truncate(mc.get_row(self.indices_pair, pair_id)),
        )

    # fn get_ready_pair(inout self, inout active_poses: IntSet) -> NextPair:
    #     # Check if there's any free pairs we can add with both poses active
    #     # It's possible this'll lead to no new factors, but that's fine
    #     for i in range(self.id_pairs.size()):
    #         var k = self.id_pairs.keys[i]
    #         if active_poses.contains(k.get[0, Int]()) and active_poses.contains(
    #             k.get[1, Int]()
    #         ):
    #             var id_pair = self.id_pairs.pop(k).__int__()
    #             return NextPair(
    #                 k.get[0, Int](),
    #                 k.get[1, Int](),
    #                 truncate(mc.get_row(self.indices_pair, id_pair)),
    #             )

    #     return NextPair(0, 0, Tensor[DType.int64](0))

    fn get_next_pair(inout self, inout active_poses: IntSet) -> NextPair:
        # TODO: This just hoping b/c one image is inited, the other will
        # have enough landmarks in it to start. No guarantee though! May need to verify more closely

        # A better option is whenever a landmakr is added, add it to the count for each pose
        # From that we can choose the next pose to add and the next pair to add from that

        # Get the next image pair with at least one inited pose and the most number of factors
        var pair_max = 0
        var pair_id = 0
        var pair = (0, 0)
        for i in range(self.id_pairs.size()):
            let k = self.id_pairs.keys[i]
            if active_poses.contains(k.get[0, Int]()) or active_poses.contains(
                k.get[1, Int]()
            ):
                let id_pair = self.id_pairs[k].__int__()
                let num_fc = self.lms_per_pair[id_pair].__int__()
                if num_fc > pair_max:
                    pair_max = num_fc
                    pair_id = id_pair
                    pair = k

        self.id_pairs.remove(pair)

        return NextPair(
            pair.get[0, Int](),
            pair.get[1, Int](),
            truncate(mc.get_row(self.indices_pair, pair_id)),
        )
