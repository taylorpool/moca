from dataclasses import dataclass
import sqlite3
import numpy as np
from tqdm import tqdm


@dataclass
class ProjectionFactor:
    id_pose: int
    id_cam: int
    id_lm: int
    id_kp: int  # solely for bookkeeping purposes
    u: float
    v: float


@dataclass
class PinholeCamera:
    id: int
    fx: float
    fy: float
    px: float
    py: float


@dataclass
class Image:
    id: int
    id_cam: int
    kp: np.ndarray


@dataclass
class Match:
    id1: int
    id2: int
    matches: np.ndarray


@dataclass
class SfMData:
    num_poses: int
    num_lm: int
    cameras: list[PinholeCamera]
    factors: list[ProjectionFactor]


# Helper to make sure things are straight when running
def verify_lookup(factors, seen, lm_lookup):
    for id_lm, id_factors in lm_lookup.items():
        for id in id_factors:
            print(id)
            f = factors[id]
            assert (
                seen[(f.id_pose, f.id_kp)] == id_lm
            ), f"Somethings off! {seen[(f.id_pose, f.id_kp)]} != {id_lm}"


def colmapdb2sfm(
    path: str,
) -> tuple[list[PinholeCamera], int, int, list[ProjectionFactor]]:
    cur = sqlite3.connect(path).cursor()

    cameras = []
    images = []
    matches = []

    # ------------------------- Get all cameras ------------------------- #
    result_cameras = cur.execute("SELECT camera_id, params FROM cameras").fetchall()
    for id, params in result_cameras:
        cal = np.frombuffer(params, np.float64)
        cameras.append(PinholeCamera(id - 1, *cal))

    # ------------------------- Load all keypoints ------------------------- #
    result_image_kps = cur.execute(
        "SELECT image_id, rows, cols, data FROM keypoints"
    ).fetchall()
    result_images_ids = cur.execute("SELECT image_id, camera_id FROM images").fetchall()

    for i in range(len(result_image_kps)):
        id, rows, cols, kp_buffer = result_image_kps[i]
        id_img, id_cam = result_images_ids[i]
        assert id == id_img

        kp = np.frombuffer(kp_buffer, np.float32).reshape((rows, cols))[:, :2]
        images.append(Image(id - 1, id_cam - 1, kp))

    # ------------------------- Get all matches ------------------------- #
    result_matches = cur.execute(
        "SELECT pair_id, rows, cols, data FROM two_view_geometries"
    ).fetchall()

    for r in result_matches:
        pair_id, rows, cols, match_buffer = r
        id_img2 = (pair_id % 2147483647) - 1
        id_img1 = int((pair_id - id_img2) / 2147483647) - 1

        m = np.frombuffer(match_buffer, np.uint32).reshape((rows, cols))
        matches.append(Match(id_img1, id_img2, m))

    # ------------------------- Convert to projection factors ------------------------- #
    seen = dict()  # holds tuples of (img_id, kp) -> lm_id
    lm_lookup = dict()  # holds lm_id -> [factor_idx, ...]
    factors = []
    idx_lm_count = 0
    idx_lm_removed = []
    for match in tqdm(matches, leave=False):
        for idx_kp1, idx_kp2 in match.matches:
            tuple1 = (match.id1, idx_kp1)
            tuple2 = (match.id2, idx_kp2)
            seen_in_1 = tuple1 in seen
            seen_in_2 = tuple2 in seen

            # if both seen, do some shuffling
            if seen_in_1 and seen_in_2:
                idx_lm1 = seen[tuple1]
                idx_lm2 = seen[tuple2]
                # Relabel if they're not the same
                if idx_lm1 != idx_lm2:
                    idxs_factors = lm_lookup.pop(idx_lm2)
                    lm_lookup[idx_lm1].extend(idxs_factors)
                    for i in idxs_factors:
                        seen[(factors[i].id_pose, factors[i].id_kp)] = idx_lm1
                        factors[i].id_lm = idx_lm1

                    idx_lm_removed.append(idx_lm2)

            # If only seen in one before
            elif seen_in_1 or seen_in_2:
                if seen_in_1:
                    idx_lm = seen[tuple1]
                else:
                    idx_lm = seen[tuple2]

            # If never seen before
            else:
                if len(idx_lm_removed) != 0:
                    idx_lm = idx_lm_removed.pop(0)
                else:
                    idx_lm = idx_lm_count
                    idx_lm_count += 1
                lm_lookup[idx_lm] = []

            # Add in factors
            if not seen_in_1:
                img = images[match.id1]
                lm_lookup[idx_lm].append(len(factors))
                factors.append(
                    ProjectionFactor(
                        img.id, img.id_cam, idx_lm, idx_kp1, *img.kp[idx_kp1]
                    )
                )
                seen[(match.id1, idx_kp1)] = idx_lm
            if not seen_in_2:
                img = images[match.id2]
                lm_lookup[idx_lm].append(len(factors))
                factors.append(
                    ProjectionFactor(
                        img.id, img.id_cam, idx_lm, idx_kp2, *img.kp[idx_kp2]
                    )
                )
                seen[(match.id2, idx_kp2)] = idx_lm

            # verify_lookup(factors, seen, lm_lookup)

    # Reorganize a bit so there's no holes!
    for idx_lm in idx_lm_removed:
        idx_to_rm = idx_lm_count - 1

        idxs_factors = lm_lookup.pop(idx_to_rm)
        lm_lookup[idx_lm] = idxs_factors
        for i in idxs_factors:
            seen[(factors[i].id_pose, factors[i].id_kp)] = idx_lm
            factors[i].id_lm = idx_lm

        idx_lm_count -= 1

    # verify_lookup(factors, seen, lm_lookup)

    return SfMData(len(images), idx_lm_count, cameras, factors)


if __name__ == "__main__":
    colmapdb2sfm("data/database.db")
