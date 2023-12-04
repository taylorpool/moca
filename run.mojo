from src.sfm import SfM
from pathlib import Path
from python import Python


fn main() raises:
    let dir = Path("trex")
    var sfm = SfM(dir)

    try:
        sfm.frontend()
    except e:
        print("Failed to run SfM frontend")
        print(e)

    sfm.register_first_pair()

    # for i in range(num_pairs):
    #     sfm.optimize()
    #     sfm.register()
    
