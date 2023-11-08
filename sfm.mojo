from pathlib import Path
from python import Python
from utils.index import Index

# TODO LIST
# - Finish DB loading

fn np2tensor[type: DType]( a:PythonObject) raises -> Tensor[type]:
    let n : Int = a.size.__index__()
    var o = Tensor[type](n)
    for i in range(n):
        o[i] = a[i].to_float64().cast[type]()
    return o

struct Camera:
    # fx fy px py
    # See: https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h#L206
    var cal : Tensor[DType.float64]

    fn __init__(inout self, cal : Tensor[DType.float64]):
        self.cal = cal

struct Image:
    var id_cam : Int32
    var kp : Tensor[DType.float32]

    fn __init__(inout self, inout id_cam : Int32, borrowed kp : Tensor[DType.float32]):
        self.id_cam = id_cam
        self.kp = kp

struct MatchPair:
    var id1 : Int32
    var id2 : Int32
    var matches : Tensor[DType.uint32]    


struct SfM:
    var dir_images : Path
    var matches : DynamicVector[MatchPair]
    var cameras : DynamicVector[Camera]
    var images : DynamicVector[Image]

    fn __init__(inout self, dir_images : Path):
        self.dir_images = dir_images
        self.matches = DynamicVector[MatchPair]()
        self.cameras = DynamicVector[Camera]()
        self.images = DynamicVector[Image]()

    fn frontend(inout self, force : Bool = False) raises:
        # Run COLMAP
        let os = Python.import_module("os")
        let db_path = self.dir_images / "database.db"
        if os.path.exists(db_path.path):
            if force:
                let temp = os.remove(db_path.path)
            else:
                print("COLMAP database already exists, loading it...")
        else:
            self._colmap_run(db_path)

        # Load COLMAP from path
        self._colmap_load(db_path)


    fn _colmap_run(inout self, db_path : Path) raises:
        """Runs Colmap & saves result at the db_path"""
        let colmap = Python.import_module("pycolmap")
        var temp = colmap.extract_features(db_path.path, self.dir_images.path)
        temp = colmap.match_exhaustive(db_path.path)


    fn _colmap_load(inout self, db_path : Path) raises:
        """Loads all colmap results from file"""
        let sqlite3 = Python.import_module("sqlite3")
        let np = Python.import_module("numpy")
        let cur = sqlite3.connect(db_path.path).cursor()

        # Get all cameras
        var result_cameras = cur.execute("SELECT camera_id, params FROM cameras").fetchall()
        let num_cameras : Int = result_cameras.__len__().__index__()
        self.cameras = DynamicVector[Camera](num_cameras)
        for r in result_cameras:
            let cam_id : Int32 = r[0].__index__()
            let params = np2tensor[DType.float64](np.frombuffer(r[1], np.float64))
            let cam = Camera(params)
            # self.cameras.push_back(cam)


        # Get all images

        # Get all matches


    fn register(inout self):
        pass

    fn optimize(inout self):
        pass


fn main() raises:
    let dir_in = "trex"
    let dir_db = "data"
    var sfm = SfM(Path(dir_in))
    sfm.frontend()
    # run_colmap_frontend(dir_in, dir_db)