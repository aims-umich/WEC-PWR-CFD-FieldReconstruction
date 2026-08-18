
import numpy as np

DEFAULT_GEOMETRY_MASK = np.array([
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
], dtype=np.uint8)


def periodic_missing_mask(
    geometry_mask: np.ndarray,
    frac_missing: float,
    pattern: str = "checkerboard",
    stride=None,
    axis: str = "row",
    seed: int | None = 42,
) -> np.ndarray:
    """Create a periodic missing-mask (True = missing) constrained to the geometry."""
    gm = np.asarray(geometry_mask).astype(bool)
    h, w = gm.shape
    n_valid = int(gm.sum())
    if n_valid == 0:
        return np.zeros_like(gm, dtype=bool)

    f = float(frac_missing)
    if f > 1.0:
        f /= 100.0
    f = np.clip(f, 0.0, 1.0)

    target = int(round(f * n_valid))
    if target == 0:
        return np.zeros_like(gm, dtype=bool)
    if target == n_valid:
        return gm.copy()

    rng = np.random.default_rng(seed)
    rr, cc = np.indices((h, w))

    def choose_phase_grid(sy, sx):
        best = None
        best_diff = 10**9
        for py in range(sy):
            for px in range(sx):
                cand = ((rr - py) % sy == 0) & ((cc - px) % sx == 0)
                cnt = int((cand & gm).sum())
                diff = abs(cnt - target)
                if diff < best_diff:
                    best_diff = diff
                    best = (py, px, cand)
        return best

    if pattern == "grid":
        if stride is None:
            s = max(1, int(round((1.0 / max(f, 1e-8)) ** 0.5)))
            sy, sx = s, s
        else:
            try:
                sy, sx = stride
            except TypeError:
                sy, sx = int(stride), int(stride)
        _py, _px, cand = choose_phase_grid(max(1, sy), max(1, sx))
        base = cand & gm
    elif pattern == "checkerboard":
        cand0 = ((rr + cc) % 2 == 0) & gm
        cand1 = ((rr + cc) % 2 == 1) & gm
        base = cand0 if abs(int(cand0.sum()) - target) <= abs(int(cand1.sum()) - target) else cand1
    elif pattern == "stripes":
        if stride is None:
            stride = max(1, int(round(1.0 / max(f, 1e-8))))
        s = int(stride)
        best = None
        best_diff = 10**9
        if axis not in ("row", "col"):
            raise ValueError("axis must be 'row' or 'col' for stripes")
        if axis == "row":
            for ph in range(s):
                cand = ((rr - ph) % s == 0) & gm
                diff = abs(int(cand.sum()) - target)
                if diff < best_diff:
                    best, best_diff = cand, diff
        else:
            for ph in range(s):
                cand = ((cc - ph) % s == 0) & gm
                diff = abs(int(cand.sum()) - target)
                if diff < best_diff:
                    best, best_diff = cand, diff
        base = best
    else:
        raise ValueError("pattern must be 'grid', 'checkerboard', or 'stripes'")

    base_idx = np.flatnonzero(base)
    if len(base_idx) > target:
        keep = rng.choice(base_idx, size=target, replace=False)
        out = np.zeros_like(gm, dtype=bool)
        out.flat[keep] = True
        return out
    if len(base_idx) < target:
        out = base.copy()
        need = target - len(base_idx)
        pool = np.flatnonzero(gm & ~out)
        if need > 0 and len(pool) > 0:
            add = rng.choice(pool, size=min(need, len(pool)), replace=False)
            out.flat[add] = True
        return out
    return base


def build_observed_and_missing_masks(
    geometry_mask: np.ndarray,
    frac_missing: float,
    pattern: str,
    stride=None,
    axis: str = "row",
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gm = np.asarray(geometry_mask).astype(bool)
    missing_mask = periodic_missing_mask(gm, frac_missing, pattern=pattern, stride=stride, axis=axis, seed=seed)
    missing_mask &= gm
    observed_mask = gm & ~missing_mask
    return gm, missing_mask, observed_mask
