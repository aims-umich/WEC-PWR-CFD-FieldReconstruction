"""
SPOD for masked 2D scalar fields (time × H × W), tailored to CatawbaCFD plane data.

- Input data layout matches your HDF5: /data -> (T, H, W), /time -> (T,)
- Mask: boolean (H, W); True = valid physics, False = padding
- Windowing: Welch blocks (Hann/Hamming), overlap in [0, 0.95)
- Weighting: optional cell weights (e.g., rho*ΔA); defaults to ones on mask
- Solver: snapshot method (K×K Hermitian) with np.linalg.eigh per frequency bin

New options to avoid "all-noise except mode 1" when global (rank‑1) content dominates:
- remove_domain_mean: subtract the spatial mean over the mask at each time → removes uniform field component (common in mass-flow planes)
- normalize: 'none' | 'relative' | 'zscore'  
  * 'relative': (q - μ)/ (|μ| + eps) using the time-mean μ per cell → flattens radial profile  
  * 'zscore':  (q - μ)/ (σ + eps) per cell → equalizes variance across the section
- highpass_hz: optional FFT-based high-pass (zeros bins f < cutoff before SPOD)  
- fmin_hz helper for picking peaks above DC when visualizing

Returns frequency-resolved eigenvalues and W-orthonormal modes on the original grid
(with NaNs on padding).
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Tuple

# ---------------------------- Utilities ------------------------------------

def _flatten_masked(stack: np.ndarray, mask: np.ndarray, dtype=np.float64) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (T,H,W) stack -> (M,T) matrix using boolean mask.
    Returns (X, midx) where midx are the flat indices of valid cells.
    """
    assert stack.ndim == 3, "stack must be (T,H,W)"
    T, H, W = stack.shape
    assert mask.shape == (H, W), "mask must be (H,W)"
    midx = mask.astype(bool).ravel()
    M = int(midx.sum())
    X = np.empty((M, T), dtype=dtype)
    for k in range(T):
        X[:, k] = stack[k].ravel()[midx]
    return X, midx


def _unflatten_to_grid(vec: np.ndarray, midx: np.ndarray, H: int, W: int, fill=np.nan, dtype=np.complex128) -> np.ndarray:
    """Place a length-M vector back onto an (H,W) grid; padding cells -> fill (default NaN)."""
    out = np.full((H, W), fill, dtype=dtype)
    out.ravel()[midx] = vec
    return out


def _make_window(nfft: int, kind: str = "hann", dtype=np.float64) -> np.ndarray:
    kind = kind.lower()
    if kind == "hann":
        return np.hanning(nfft).astype(dtype)
    if kind == "hamming":
        return np.hamming(nfft).astype(dtype)
    raise ValueError(f"Unsupported window: {kind}")


# ------------------------- Optional pre-processing --------------------------

def _preprocess_scalar(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    remove_domain_mean: bool = False,
    normalize: str = "none",   # 'none' | 'relative' | 'zscore'
    highpass_hz: Optional[float] = None,
    dt: Optional[float] = None,
    eps: float = 1e-12,
    dtype = np.float64,
) -> np.ndarray:
    """Apply optional de-trending/normalization to improve mode visibility.

    - remove_domain_mean: subtract spatial average over the mask at each time (kills rank‑1 global field).
    - normalize='relative': (q-μ)/(abs(μ)+eps) per cell using time-mean μ (flattens radial profile).
    - normalize='zscore'  : (q-μ)/(σ+eps) per cell using time-std σ (equalizes variance).
    - highpass_hz: FFT-based high-pass per cell (zeros rFFT bins below cutoff), requires dt.
    """
    assert data.ndim == 3
    T, H, W = data.shape
    Q = data.astype(dtype, copy=True)

    # Remove global (uniform) component at each time
    if remove_domain_mean:
        m = mask.astype(bool)
        area = m.sum()
        # spatial mean over the mask for each time
        gmean = np.where(area>0, np.nanmean(Q[:, m], axis=1), 0.0)
        Q[:, m] -= gmean[:, None]

    # Per-cell mean/std for normalization
    mu = np.mean(Q, axis=0)
    if normalize == 'relative':
        denom = np.abs(mu) + eps
        Q = (Q - mu) / denom
    elif normalize == 'zscore':
        sig = np.std(Q, axis=0) + eps
        Q = (Q - mu) / sig

    # Optional high-pass
    if highpass_hz is not None and highpass_hz > 0.0:
        if dt is None:
            raise ValueError("highpass_hz requires dt")
        n = T
        freqs = np.fft.rfftfreq(n, d=dt)
        keep = freqs >= float(highpass_hz)
        # Process per masked pixel to save work
        idx = mask.ravel()
        for j in np.where(idx)[0]:
            ts = Q.reshape(T, -1)[:, j]
            Xf = np.fft.rfft(ts)
            Xf[~keep] = 0.0
            Q.reshape(T, -1)[:, j] = np.fft.irfft(Xf, n=n)

    return Q


# ---------------------------- Core SPOD -------------------------------------

def spod_scalar_masked(
    data: np.ndarray,
    mask: np.ndarray,
    dt: Optional[float] = None,
    time: Optional[np.ndarray] = None,
    *,
    nfft: int = 1024,
    overlap: float = 0.5,
    nmodes: int = 6,
    window: str = "hann",
    weights: Optional[np.ndarray] = None,
    demean: bool = True,
    # New knobs
    remove_domain_mean: bool = False,
    normalize: str = "none",
    highpass_hz: Optional[float] = None,
    dtype = np.float64,
) -> Dict[str, np.ndarray]:
    """Spectral Proper Orthogonal Decomposition for a masked scalar field.

    Parameters
    ----------
    data : (T,H,W) array
        Time-resolved scalar field (e.g., mass flow) with time along axis 0.
    mask : (H,W) bool
        True for valid cells; False for zero-padding. Used both to extract and plot.
    dt : float, optional
        Sampling interval. If None, inferred from `time` via median diff.
    time : (T,), optional
        Time vector; used to infer `dt` if `dt` is None.
    nfft : int, default 1024
        Window length (samples) for Welch blocks.
    overlap : float in [0, 0.95), default 0.5
        Fractional overlap between adjacent windows.
    nmodes : int, default 6
        Number of leading modes to return per frequency bin.
    window : {"hann", "hamming"}
        Temporal window type.
    weights : (H,W) array, optional
        Non-negative spatial weights (e.g., ρ·ΔA). If None, uses ones on the mask.
    demean : bool, default True
        Subtract time-mean at each spatial DOF before SPOD.
    remove_domain_mean : bool, default False
        Subtract spatial mean over mask at each time step (kills rank‑1 global component).
    normalize : str, default 'none'
        'relative' or 'zscore' normalization per cell (see module docstring) applied **before** demeaning.
    highpass_hz : float, optional
        If set, apply FFT high-pass at `highpass_hz` prior to SPOD (after normalization).
    dtype : numpy dtype, default float64
        Working real dtype for time-domain arrays and weights.

    Returns
    -------
    result : dict with keys
        freqs : (F,) real-valued frequency bins (Hz)
        evals : (F, nmodes) leading SPOD eigenvalues (spectral energy)
        modes : (F, H, W, nmodes) complex modes on grid (NaN on padding)
        K     : number of Welch blocks
        hop   : hop size (samples)
        nfft, overlap, win_norm, dt, mask : analysis metadata
    """
    assert data.ndim == 3, "data must be (T,H,W)"
    T, H, W = data.shape
    assert mask.shape == (H, W), "mask must be (H,W)"
    if dt is None:
        if time is None:
            raise ValueError("Provide either dt or time vector to infer dt.")
        time = np.asarray(time)
        if time.ndim != 1 or time.size != T:
            raise ValueError("time must be 1D and match data length T.")
        dt = float(np.median(np.diff(time)))
    if not (0.0 <= overlap < 0.95):
        raise ValueError("overlap must be in [0, 0.95)")

    # Optional pre-processing to expose coherent structure
    Q = _preprocess_scalar(
        data, mask,
        remove_domain_mean=remove_domain_mean,
        normalize=normalize,
        highpass_hz=highpass_hz,
        dt=dt,
        dtype=dtype,
    )

    # Flatten on mask -> (M,T)
    X, midx = _flatten_masked(Q.astype(dtype, copy=False), mask, dtype=dtype)
    M = X.shape[0]

    # Demean (after optional normalization) to kill DC leakage
    if demean:
        X -= X.mean(axis=1, keepdims=True)

    # Spatial weights (energy inner product). Use ones if none provided.
    if weights is None:
        w_cell = np.ones(M, dtype=dtype)
    else:
        if weights.shape != (H, W):
            raise ValueError("weights must be (H,W)")
        w_cell = np.asarray(weights, dtype=dtype).ravel()[midx]
        if np.any(w_cell < 0):
            raise ValueError("weights must be non-negative")
    Wsqrt = np.sqrt(w_cell)

    # Welch blocking
    win = _make_window(nfft, window, dtype=dtype)
    hop = int(nfft * (1.0 - overlap))
    if hop <= 0 or nfft > T:
        raise ValueError(f"Invalid nfft/overlap for T={T}: hop={hop}")
    K = 1 + (T - nfft) // hop
    F = nfft // 2 + 1
    freqs = np.fft.rfftfreq(nfft, d=dt)
    win_norm = float(np.sum(win**2))

    Xhat_blocks = np.empty((K, M, F), dtype=np.complex128)
    bi = 0
    for t0 in range(0, T - nfft + 1, hop):
        seg = X[:, t0:t0 + nfft] * win[None, :]
        Xhat_blocks[bi] = np.fft.rfft(seg, axis=1)
        bi += 1

    # Frequency-by-frequency eigenproblems
    evals = np.zeros((F, nmodes), dtype=dtype)
    modes = np.full((F, H, W, nmodes), np.nan, dtype=np.complex128)
    inv_sqrt = 1.0 / np.sqrt(win_norm)

    for fi in range(F):
        Yf = (Xhat_blocks[:, :, fi].T * Wsqrt[:, None]) * inv_sqrt   # (M,K)
        C = (Yf.conj().T @ Yf) / K                                   # (K×K)
        lam, Alpha = np.linalg.eigh(C)
        idx = np.argsort(lam)[::-1]
        lam = lam[idx]; Alpha = Alpha[:, idx]
        r = min(nmodes, lam.size)
        for m in range(r):
            if lam[m] <= 0: continue
            psi = (Yf @ Alpha[:, m]) / np.sqrt(K * lam[m])           # weighted mode
            phi = psi / Wsqrt                                        # physical mode
            evals[fi, m] = lam[m]
            modes[fi, :, :, m] = _unflatten_to_grid(phi, midx, H, W, fill=np.nan)

    return dict(freqs=freqs, evals=evals, modes=modes,
                K=K, hop=hop, nfft=nfft, overlap=overlap,
                win_norm=win_norm, dt=dt, mask=mask)


# ---------------------------- Plot helpers ----------------------------------

def plot_spod_spectrum(evals: np.ndarray, freqs: np.ndarray, which: int = 0, ax=None):
    """Plot the leading (or m-th) SPOD eigenvalue vs frequency."""
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    ax.plot(freqs, evals[:, which])
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(f"λ{which+1}")
    ax.set_title("SPOD spectrum")
    return ax


def pick_peak_index(evals: np.ndarray, freqs: np.ndarray, fmin_hz: float = 0.0, which: int = 0) -> int:
    """Return the index of the largest λ_{which+1} for freqs >= fmin_hz."""
    mask = freqs >= float(fmin_hz)
    if not np.any(mask):
        return int(np.argmax(evals[:, which]))
    i0 = np.argmax(evals[mask, which])
    return int(np.where(mask)[0][0] + i0)


def plot_modes_grid(modes_f: np.ndarray, fval: float, title_prefix: str = "SPOD modes", rows: int = 2, cols: int = 3):
    """Plot a rows×cols grid of the first modes at a given frequency bin.
    modes_f: (H,W,nmodes) complex slice from result['modes'][fi]
    """
    import matplotlib.pyplot as plt
    H, W, nm = modes_f.shape
    nm_show = min(rows * cols, nm)
    fig = plt.figure(figsize=(12, 6))
    for i in range(nm_show):
        ax = fig.add_subplot(rows, cols, i + 1)
        im = ax.imshow(np.real(modes_f[:, :, i]), origin='lower')
        ax.set_title(f"Mode {i+1}")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{title_prefix} — first {nm_show} modes @ {fval:.2f} Hz", y=0.98)
    fig.tight_layout()
    return fig


# ---------------------------- Example usage ---------------------------------
if __name__ == "__main__":
    """Minimal example using the project HDF5 layout.

    Recommended settings for mass-flow planes:
    - remove_domain_mean=True
    - normalize='relative' (or 'zscore')
    - fmin_hz ~ 1–5 Hz to avoid DC peak when plotting
    - longer nfft (e.g., 2048) and 75% overlap for smoother spectra
    """
    import h5py, matplotlib.pyplot as plt
    path = "/mnt/data/plane_09_lowest.h5"
    with h5py.File(path, "r") as f:
        data = np.array(f["/data"][...], dtype=np.float64)   # (T,H,W)
        time = np.array(f["/time"][...], dtype=np.float64)   # (T,)

    # Infer a mask from time-mean magnitude (or supply your own project mask)
    mu = np.mean(np.abs(data), axis=0)
    mask = mu > (1e-6 * np.max(mu) + 1e-12)

    res = spod_scalar_masked(
        data, mask, time=time,
        nfft=2048, overlap=0.75, nmodes=6, window="hann",
        remove_domain_mean=True, normalize='relative', highpass_hz=None,
    )

    freqs, evals = res["freqs"], res["evals"]
    plt.figure(); plot_spod_spectrum(evals, freqs, which=0); plt.tight_layout(); plt.show()

    fi = pick_peak_index(evals, freqs, fmin_hz=2.0, which=0)
    fpk = float(freqs[fi])
    print("Chosen frequency (>=2 Hz):", fpk, "Hz")

    modes_f = res["modes"][fi]   # (H,W,nmodes)
    plot_modes_grid(modes_f, fpk, title_prefix="Scalar SPOD"); plt.show()
