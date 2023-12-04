from src.sfm import SfM
from pathlib import Path
from python import Python
from src.variables import Landmark


fn main() raises:
    var x = DynamicVector[Landmark](5)
    x.push_back(Landmark(SIMD[DType.float64, 4](1.0, 2.0, 3.0, 4.0)))
    var y = x[0]
    y.val[0] = 5.0
    print(x[0].val, y.val)
    let dir = Path("trex_small")
    var sfm = SfM(dir)

    sfm.frontend()

    sfm.register_first_pair()
    sfm.optimize()

    # print(sfm.state.poses.active_to_id.__str__())
    # print()

    # for i in range(5):
    #     sfm.optimize()
    #     sfm.register()
    #     print(sfm.state.poses.active_to_id.__str__())
    #     print(sfm.state.cameras.active_to_id.__str__())
    #     print()

    var np = Python.import_module("numpy")
    var data = Python.evaluate("[]")
    for i in range(sfm.state.landmarks.size()):
        let lm = sfm.state.landmarks[i]
        _ = data.append([lm.val[0], lm.val[1], lm.val[2]])

    _ = np.save("data.npy", data)
