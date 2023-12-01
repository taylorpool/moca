from dataclasses import dataclass
import sqlite3
import numpy as np
from tqdm import tqdm
import pycolmap
from pathlib import Path


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
    pair_indices: np.ndarray
    pairs: np.ndarray


# Helper to make sure things are straight when running
def verify_lookup(factors, seen, lm_lookup):
    for id_lm, id_factors in lm_lookup.items():
        for id in id_factors:
            f = factors[id]
            assert f.id_lm == id_lm, f"Somethings off! {f.id_lm} lm in {id_lm} lookup"
            assert (
                seen[(f.id_pose, f.id_kp)] in id_factors
            ), f"Somethings off! {seen[(f.id_pose, f.id_kp)]} not in {id_factors}, has {factors[seen[(f.id_pose, f.id_kp)]]} lm_id"


def pad_to_dense(M, val=0):
    """Appends the minimal required amount of zeroes at the end of each
    array in the jagged array `M`, such that `M` looses its jagedness.
    https://stackoverflow.com/questions/37676539/numpy-padding-matrix-of-different-row-size
    """

    maxlen = max(len(r) for r in M)

    Z = np.full((len(M), maxlen), val)
    for enu, row in enumerate(M):
        Z[enu, : len(row)] = row
    return Z


def frontend(dir_img_str: str, force=False) -> SfMData:
    # ------------------------- Run frontend ------------------------- #
    dir_img = Path(dir_img_str)
    file_db = dir_img / "database.db"
    if force and file_db.exists():
        file_db.unlink()
    if file_db.exists():
        print("COLMAP database already exists, loading it...")
    else:
        pycolmap.extract_features(file_db, dir_img)
        pycolmap.match_exhaustive(file_db)

    cur = sqlite3.connect(file_db).cursor()

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
    seen: dict[
        tuple[int, int], int
    ] = dict()  # holds tuples of (img_id, kp) -> factor_idx
    lm_lookup: dict[int, list[int]] = dict()  # holds lm_id -> [factor_idx, ...]
    pair_factor_list = []
    pairs = []
    factors: list[ProjectionFactor] = []

    idx_lm_count = 0
    idx_lm_removed: list[int] = []
    for match in tqdm(matches, leave=False):
        factor_idx_list = []
        pairs.append((match.id1, match.id2))
        for idx_kp1, idx_kp2 in match.matches:
            tuple1 = (match.id1, idx_kp1)
            tuple2 = (match.id2, idx_kp2)
            seen_in_1 = tuple1 in seen
            seen_in_2 = tuple2 in seen

            # if both seen, do some shuffling
            if seen_in_1 and seen_in_2:
                idx_lm2 = factors[seen[tuple2]].id_lm
                idx_lm1 = factors[seen[tuple1]].id_lm
                # Relabel if they're not the same
                if idx_lm1 != idx_lm2:
                    idxs_factors = lm_lookup.pop(idx_lm2)
                    lm_lookup[idx_lm1].extend(idxs_factors)
                    for i in idxs_factors:
                        factors[i].id_lm = idx_lm1

                    idx_lm_removed.append(idx_lm2)

            # If only seen in one before
            elif seen_in_1 or seen_in_2:
                if seen_in_1:
                    idx_lm = factors[seen[tuple1]].id_lm
                else:
                    idx_lm = factors[seen[tuple2]].id_lm

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
                seen[tuple1] = len(factors)
                factors.append(
                    ProjectionFactor(
                        img.id,
                        img.id_cam,
                        idx_lm,
                        idx_kp1,
                        *img.kp[idx_kp1],
                    )
                )

            if not seen_in_2:
                img = images[match.id2]
                lm_lookup[idx_lm].append(len(factors))
                seen[tuple2] = len(factors)
                factors.append(
                    ProjectionFactor(
                        img.id,
                        img.id_cam,
                        idx_lm,
                        idx_kp2,
                        *img.kp[idx_kp2],
                    )
                )

            factor_idx_list.append(seen[tuple1])
            factor_idx_list.append(seen[tuple2])

        pair_factor_list.append(factor_idx_list)

    # Reorganize a bit so there's no holes!
    for idx_lm in idx_lm_removed:
        idx_to_rm = idx_lm_count - 1

        idxs_factors = lm_lookup.pop(idx_to_rm)
        lm_lookup[idx_lm] = idxs_factors
        for i in idxs_factors:
            factors[i].id_lm = idx_lm

        idx_lm_count -= 1

    verify_lookup(factors, seen, lm_lookup)

    return SfMData(
        len(images),
        idx_lm_count,
        cameras,
        factors,
        pad_to_dense(pair_factor_list, -1),
        np.array(pairs),
    )


if __name__ == "__main__":
    frontend("trex")
