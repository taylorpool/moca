from pathlib import Path
from python import Python
from tensor import TensorShape

from mytypes import PinholeCamera, Image, MatchPair
from myutils import np2tensor, np2tensor2d

# TODO LIST
# - Figure out how to store Image / Matches in SfM
# - SE(3) class for optimization
# - Fix loading binary -> numpy -> Tensor


struct SfM:
    var dir_images: Path
    var cameras: DynamicVector[PinholeCamera]
    var images: Pointer[Image]
    var matches: Pointer[MatchPair]

    fn __init__(inout self, dir_images: Path):
        self.dir_images = dir_images
        self.cameras = DynamicVector[PinholeCamera](0)
        self.images = Pointer[Image]()
        self.matches = Pointer[MatchPair]()

    fn frontend(inout self, force: Bool = False) raises:
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

    fn _colmap_run(inout self, db_path: Path) raises:
        """Runs Colmap & saves result at the db_path"""
        let colmap = Python.import_module("pycolmap")
        var temp = colmap.extract_features(db_path.path, self.dir_images.path)
        temp = colmap.match_exhaustive(db_path.path)

    fn _colmap_load(inout self, db_path: Path) raises:
        """Loads all colmap results from file"""
        let sqlite3 = Python.import_module("sqlite3")
        let np = Python.import_module("numpy")
        let cur = sqlite3.connect(db_path.path).cursor()

        # ------------------------- Get all cameras ------------------------- #
        var result_cameras = cur.execute(
            "SELECT camera_id, params FROM cameras"
        ).fetchall()
        let num_cameras: Int = result_cameras.__len__().__index__()
        self.cameras = DynamicVector[PinholeCamera](num_cameras)
        # pushback a dummy one since camera indexing technically starts from one
        self.cameras.push_back(PinholeCamera(0, 0, 0, 0))
        for r in result_cameras:
            let cam_id: Int = r[0].__index__()
            let params = np2tensor[DType.float64](np.frombuffer(r[1], np.float64))
            let cam = PinholeCamera(params[0], params[1], params[2], params[3])
            self.cameras.push_back(cam)

        # ------------------------- Get all images ------------------------- #
        let result_images_ids = cur.execute(
            "SELECT image_id, camera_id FROM images"
        ).fetchall()
        let result_image_kps = cur.execute(
            "SELECT image_id, rows, cols, data FROM keypoints"
        ).fetchall()
        let num_images: Int = result_images_ids.__len__().__index__()
        # TODO: Add in dummy one since images indexing starts from one
        # self.images = Pointer[Image].alloc(num_images)

        for i in range(num_images):
            debug_assert(
                result_images_ids[i][0].__index__()
                == result_image_kps[i][0].__index__(),
                "Image ID / Keypoints aren't lined up in DB",
            )
            let image_id = result_images_ids[i][0].__index__()
            let cam_id = result_images_ids[i][1].__index__()
            let rows = result_image_kps[i][1].__index__()
            let cols = result_image_kps[i][2].__index__()
            let kp_np = np.frombuffer(result_image_kps[i][3], np.float32).reshape(
                (rows, cols)
            )
            let kp = np2tensor2d[DType.float32](kp_np, m=2)

        # ------------------------- Get all matches ------------------------- #
        var result_matches = cur.execute(
            "SELECT pair_id, rows, cols, data FROM two_view_geometries"
        ).fetchall()
        let num_pairs = result_matches.__len__().__index__()
        # self.matches = Pointer[Match].alloc(num_pairs)

        for r in result_matches:
            let pair_id = r[0].__index__()
            let img2_id = pair_id % 2147483647
            let img1_id = (pair_id - img2_id) / 2147483647

            let row = r[1].__index__()
            let col = r[2].__index__()
            let match_np = np.frombuffer(r[3], np.uint32).reshape((row, col))
            let matches = np2tensor2d[DType.uint32](match_np)

    fn register(inout self):
        pass

    fn optimize(inout self):
        pass


fn main() raises:
    let dir_in = "trex"
    var sfm = SfM(Path(dir_in))
    sfm.frontend()

    # var test = Pointer[Test].alloc(10)
    # # test.alloc(10)
    # test.store(1, Test(0))
    # print(test[1].fx)
