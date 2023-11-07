from pathlib import Path
from python import Python

# TODO LIST
# - Make db struct to interface with colmap database
# - Make camera/image/keypoint/match objects as needed for optimizaiton purpose

fn run_colmap_frontend(dir_images : Path, dir_db : Path):
    let colmap : PythonObject
    let os : PythonObject
    try:
        colmap = Python.import_module("pycolmap")
        os = Python.import_module("os")
    except e:
        print("Failed to import python objects, can't run frontend")
        print(e)
        return

    try:
        var temp = os.makedirs(dir_db.path, 511, True)
        let db_path = dir_db / "database.db"
        if os.path.exists(db_path.path):
            print("Skipping COLMAP, db already exists")
            return

        temp = colmap.extract_features(db_path.path, dir_images.path)
        temp = colmap.match_exhaustive(db_path.path)

    except e:
        print("Failed at running colamp")
        print(e)


struct SfM:
    var db : PythonObject

    fn __init__(inout self, db : Path):
        try:
            var COLMAPDatabase = Python.import_module("colmap_db.COLMAPDatabase")
            self.db = COLMAPDatabase.connect(db.path)
            print("Connected!")
        except e:
            print(e)

    fn register(inout self):
        pass

    fn optimize(inout self):
        pass


fn main():
    let dir_in = "trex"
    let dir_db = "data"
    var sfm = SfM(Path(dir_db) / "database.db")
    # run_colmap_frontend(dir_in, dir_db)