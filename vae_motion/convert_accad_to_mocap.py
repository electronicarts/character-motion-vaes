#!/usr/bin/env python3
import argparse, io, tarfile, numpy as np, json, os

def aa_to_rot6(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (...,3) -> rot6 (...,6), memory-friendly."""
    aa = np.asarray(aa, dtype=np.float32)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True) + 1e-8
    axis = aa / angle
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    cos_angle = np.cos(angle)[..., 0]
    sin_angle = np.sin(angle)[..., 0]

    # First column of R
    c1_0 = 1 + (cos_angle - 1) * x * x
    c1_1 = (cos_angle - 1) * x * y - sin_angle * z
    c1_2 = (cos_angle - 1) * x * z + sin_angle * y

    # Second column of R
    c2_0 = (cos_angle - 1) * x * y + sin_angle * z
    c2_1 = 1 + (cos_angle - 1) * y * y
    c2_2 = (cos_angle - 1) * y * z - sin_angle * x

    col1 = np.stack([c1_0, c1_1, c1_2], axis=-1)
    col2 = np.stack([c2_0, c2_1, c2_2], axis=-1)
    return np.concatenate([col1, col2], axis=-1)  # (..., 6)

def parse_keep_joints(arg: str, J_total: int) -> np.ndarray:
    """
    Parse '--keep-joints' like '0,1,2,...' or 'range:0-43'.
    Default: first 44 joints (0..43).
    """
    if arg is None:
        return np.arange(44, dtype=np.int64)  # default 44
    arg = arg.strip()
    if arg.startswith("range:"):
        lo, hi = arg.split("range:")[1].split("-")
        idx = np.arange(int(lo), int(hi) + 1, dtype=np.int64)
    else:
        idx = np.array([int(x) for x in arg.split(",") if x.strip() != ""], dtype=np.int64)
    if len(idx) != 44:
        raise ValueError(f"--keep-joints must select exactly 44 joints, got {len(idx)}")
    if idx.min() < 0 or idx.max() >= J_total:
        raise ValueError(f"Joint index out of range [0, {J_total-1}] in {idx}")
    return idx

def build_features_267(trans: np.ndarray, poses: np.ndarray, keep_idx: np.ndarray) -> np.ndarray:
    """
    Make 267-D features: [root trans(3), 44 joints * rot6(264)] = 267.
    No delta terms.
    poses: (T, 156) for SMPL-H (52*3).
    keep_idx: (44,) joint indices to keep (0-based within 52).
    """
    T = poses.shape[0]
    joints = poses.reshape(T, -1, 3)                  # (T, 52, 3) for SMPL-H
    # batchify for memory
    B = 1000
    rot6_sel_parts = []
    for s in range(0, T, B):
        e = min(s + B, T)
        # convert to rot6 for the whole frame-chunk
        r6 = aa_to_rot6(joints[s:e].reshape(-1, 3)).reshape(e - s, -1, 6)  # (chunk, 52, 6)
        r6_sel = r6[:, keep_idx, :].reshape(e - s, -1)                     # (chunk, 44*6)
        rot6_sel_parts.append(r6_sel.astype(np.float32))
    rot6_sel = np.concatenate(rot6_sel_parts, axis=0)                      # (T, 264)

    # concat: trans(3) + rot6(264) = 267
    X = np.concatenate([trans.astype(np.float32), rot6_sel], axis=1)
    assert X.shape[1] == 267, f"Expected 267-D, got {X.shape[1]}"
    return X

def convert(accad_tar_bz2: str, out_npz: str, keep_spec: str = None, min_len: int = 10) -> dict:
    seq_features, meta = [], []
    with tarfile.open(accad_tar_bz2, "r:bz2") as tf:
        names = [m.name for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".npz")]
        # pre-read first file to infer J_total
        if not names:
            raise RuntimeError("No .npz found inside archive.")
        # Infer total joints (should be 52 for SMPL-H in AMASS)
        head = np.load(io.BytesIO(tf.extractfile(names[0]).read()), allow_pickle=True)
        J_total = head["poses"].shape[1] // 3
        keep_idx = parse_keep_joints(keep_spec, J_total)

        for n in names:
            try:
                data = tf.extractfile(n).read()
                npz = np.load(io.BytesIO(data), allow_pickle=True)
            except Exception as e:
                print("Skip (read error):", n, e); continue
            if not all(k in npz for k in ["poses", "trans", "mocap_framerate"]):
                continue
            X = build_features_267(npz["trans"], npz["poses"], keep_idx)
            if X.shape[0] < min_len:
                continue
            seq_features.append(X)
            meta.append({
                "name": n,
                "framerate": float(npz["mocap_framerate"]),
                "length": int(X.shape[0])
            })
    if not seq_features:
        raise RuntimeError("No valid sequences found.")
    lengths = np.array([s.shape[0] for s in seq_features], dtype=np.int32)
    offsets = np.cumsum(np.concatenate([[0], lengths[:-1]])).astype(np.int32)
    Xcat = np.concatenate(seq_features, axis=0).astype(np.float32)
    dim = np.array([Xcat.shape[1]], dtype=np.int32)
    np.savez(out_npz, X=Xcat, lengths=lengths, offsets=offsets, dim=dim, meta=np.array(meta, dtype=object))
    return {"clips": len(seq_features), "total_frames": int(Xcat.shape[0]), "dim": int(dim[0])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accad", required=True, help="Path to ACCAD.tar.bz2")
    ap.add_argument("--out", default="environments/mocap.npz", help="Output npz path")
    ap.add_argument("--keep-joints", default=None,
                    help="Comma list (e.g. '0,1,...,43') or 'range:0-43'. Must be exactly 44 indices.")
    ap.add_argument("--min-len", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    stats = convert(args.accad, args.out, args.keep_joints, args.min_len)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
