from src.sfm import SfM
from pathlib import Path
from python import Python


fn main() raises:
    let dir = Path("trex8")
    var sfm = SfM(dir)

    try:
        sfm.frontend()
    except e:
        print("Failed to run SfM frontend")
        print(e)

    sfm.register_first_pair()

    print(sfm.state.poses.active_to_id.__str__())
    print()

    for i in range(5):
        sfm.optimize()
        sfm.register()
        print(sfm.state.poses.active_to_id.__str__())
        print(sfm.state.cameras.active_to_id.__str__())
        print()
