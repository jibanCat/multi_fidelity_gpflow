# sb28_smf_loader.py
"""
SB28 Stellar Mass Function (SMF) multi-fidelity data loader — 28D params, three fidelities, two+ snapshots

Key points
----------
- 28D parameter set (auto-detects all numeric columns; no hard-coded 6D).
- All transformations are top-level functions: bin width, phi→counts, Anscombe, standardization, uncertainties.
- Class exposes normalized X and Y for training: X128/X256/X512 and Y128/Y256/Y512.
- Stacks multiple snapshots (e.g., 73 and 90) along features: [bins@73 | bins@90].
- Units: phi in (h^3 Mpc^-3 dex^-1), L in (Mpc/h).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .gpemulator_singlebin import _map_params_to_unit_cube as input_normalize

# ------------------------- Top-level transforms -------------------------
CountModel = Literal["poisson", "jeffreys"]


def infer_bin_width(centers: np.ndarray, edges: Optional[np.ndarray] = None) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if edges is not None:
        edges = np.asarray(edges, dtype=float)
        w = np.diff(edges)
        if np.any(w <= 0):
            raise ValueError("Non-positive bin widths from edges.")
        return w.astype(np.float32)
    diffs = np.diff(centers)
    med = np.median(diffs)
    # If nearly uniform, use the median spacing; else use mid-edge widths
    if np.allclose(diffs, med, rtol=0, atol=1e-3):
        return np.full_like(centers, med, dtype=np.float32)
    left = np.r_[centers[0] - diffs[0] / 2, (centers[:-1] + centers[1:]) / 2]
    right = np.r_[(centers[:-1] + centers[1:]) / 2, centers[-1] + diffs[-1] / 2]
    return (right - left).astype(np.float32)


def to_counts(phi: np.ndarray, dlog10M: np.ndarray, Lbox: float) -> np.ndarray:
    """phi→counts for stacked snapshots.
    phi: (N, B*S), dlog10M: (B,), S snapshots → we tile widths to (B*S,).
    """
    phi = np.asarray(phi, dtype=float)
    B = int(dlog10M.shape[0])
    S = phi.shape[1] // B
    d = np.tile(dlog10M, S)  # (B*S,)
    V = float(Lbox) ** 3
    return phi * (V * d)


def anscombe(n: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(n + 3.0 / 8.0)


def inv_anscombe_mean_var(mA: np.ndarray, vA: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_hat = (mA * 0.5) ** 2 - 3.0 / 8.0
    var_n = (mA * 0.5) ** 2 * vA
    return n_hat, var_n


def counts_uncertainty(
    phi: np.ndarray,
    dlog10M: np.ndarray,
    Lbox: float,
    count_model: CountModel = "jeffreys",
    frac_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (n_counts, sigma_phi_model, sigma_phi_total) for stacked snapshots."""
    n = to_counts(phi, dlog10M, Lbox)
    if count_model == "poisson":
        var_n = np.clip(n, 0.0, None)
    elif count_model == "jeffreys":
        var_n = np.clip(n + 0.5, 0.0, None)  # nonzero variance at n=0
    else:
        raise ValueError("count_model must be 'poisson' or 'jeffreys'")
    B = dlog10M.shape[0]
    S = phi.shape[1] // B
    d = np.tile(dlog10M, S)
    V = float(Lbox) ** 3
    var_phi = var_n / (V * d) ** 2
    sigma_phi = np.sqrt(var_phi)
    sigma_total = np.sqrt(sigma_phi**2 + (frac_floor * np.abs(phi)) ** 2)
    return n, sigma_phi, sigma_total


# ------------------------- Standardization helpers -------------------------
def fit_standardizer(Y: np.ndarray) -> Dict[str, np.ndarray]:
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return {"mu": mu, "sd": sd}


def apply_standardizer(Y: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (Y - stats["mu"]) / stats["sd"]


# ------------------------- IO helpers -------------------------
@dataclass
class SB28Paths:
    basedir: str
    params_n128: str = "CosmoAstroSeed_IllustrisTNG_L25n128_SB28_MF.txt"
    params_n256: str = "CosmoAstroSeed_IllustrisTNG_L25n256_SB28.txt"
    params_n512: str = "CosmoAstroSeed_IllustrisTNG_L25n512_SB28_MF.txt"
    smf_low_pattern: str = "SB28_low_smf_13bins_{snap}.npy"  # n128
    smf_mid_pattern: str = "SB28_smf_13bins_{snap}.npy"       # n256
    smf_high_pattern: str = "SB28_high_smf_13bins_{snap}.npy" # n512

    # info file for parameter limits
    params_info: str = "Info_IllustrisTNG_L25n256_28params.txt"


# ------------------------- Main DataLoader -------------------------
class SMFDataLoaderSB28:
    """SB28 3-fidelity loader with normalized X/Y ready for training.

    Parameters
    ----------
    paths : SB28Paths
    snapshots : list[int]
    log10M_centers : (B,) array
    bin_edges : optional (B+1,) array
    Lbox_mpc_over_h : float, default 25.0
    y_transform : {'phi','counts','anscombe'}, default 'anscombe'
        Which target space to model before standardizing.
    standardize_X : bool, default True
    standardize_Y : bool, default True
    param_subset : Optional[list[str]]
        If provided, select these columns from parameter tables (must exist across files).
        If None, **all numeric columns** are used (28D typical for SB28 tables).
    """

    def __init__(
        self,
        paths: SB28Paths,
        snapshots: List[int],
        log10M_centers: np.ndarray,
        bin_edges: Optional[np.ndarray] = None,
        Lbox_mpc_over_h: float = 25.0,
        y_transform: Literal["phi", "counts", "anscombe"] = "anscombe",
        standardize_X: bool = True,
        standardize_Y: bool = True,
        param_subset: Optional[List[str]] = None,
    ) -> None:
        self.paths = paths
        self.snapshots = list(snapshots)
        self.log10M = np.asarray(log10M_centers, dtype=np.float32)
        self.dlog10M = infer_bin_width(self.log10M, bin_edges)
        self.B = len(self.log10M)
        self.S = len(self.snapshots)
        self.Lbox = float(Lbox_mpc_over_h)
        self.y_transform = y_transform
        self.standardize_X = standardize_X
        self.standardize_Y = standardize_Y
        self.param_subset = param_subset

        # Load parameter tables
        self.df128 = self._read_param_table(paths.params_n128)
        self.df256 = self._read_param_table(paths.params_n256)
        self.df512 = self._read_param_table(paths.params_n512)

        # Select columns (28D default = all numeric)
        self.X128_raw = self._select_param_columns(self.df128)
        self.X256_raw = self._select_param_columns(self.df256)
        self.X512_raw = self._select_param_columns(self.df512)


        # Standardize X if requested (fit on each fidelity separately by default)
        if self.standardize_X:
            self.X128_stats = fit_standardizer(self.X128_raw)
            self.X256_stats = fit_standardizer(self.X256_raw)
            self.X512_stats = fit_standardizer(self.X512_raw)
            self.X128 = apply_standardizer(self.X128_raw, self.X128_stats)
            self.X256 = apply_standardizer(self.X256_raw, self.X256_stats)
            self.X512 = apply_standardizer(self.X512_raw, self.X512_stats)
        else:
            self.X128 = self.X128_raw
            self.X256 = self.X256_raw
            self.X512 = self.X512_raw
            self.X128_stats = self.X256_stats = self.X512_stats = {
                "mu": np.zeros(self.X128_raw.shape[1]),
                "sd": np.ones(self.X128_raw.shape[1]),
            }

        # Load SMFs and stack snapshots
        self.PHI128 = self._load_stack_smfs(paths.smf_low_pattern)
        self.PHI256 = self._load_stack_smfs(paths.smf_mid_pattern)
        self.PHI512 = self._load_stack_smfs(paths.smf_high_pattern)

        # Build target space (phi / counts / Anscombe) and standardize per-fidelity
        self.Y128_raw = self._build_targets(self.PHI128)
        self.Y256_raw = self._build_targets(self.PHI256)
        self.Y512_raw = self._build_targets(self.PHI512)


        if self.standardize_Y:
            self.Y128_stats = fit_standardizer(self.Y128_raw)
            self.Y256_stats = fit_standardizer(self.Y256_raw)
            self.Y512_stats = fit_standardizer(self.Y512_raw)
            self.Y128 = apply_standardizer(self.Y128_raw, self.Y128_stats)
            self.Y256 = apply_standardizer(self.Y256_raw, self.Y256_stats)
            self.Y512 = apply_standardizer(self.Y512_raw, self.Y512_stats)
        else:
            self.Y128 = self.Y128_raw
            self.Y256 = self.Y256_raw
            self.Y512 = self.Y512_raw
            D = self.Y128.shape[1]
            self.Y128_stats = self.Y256_stats = self.Y512_stats = {"mu": np.zeros(D), "sd": np.ones(D)}

    # ------------------------- Internals -------------------------
    def _read_param_table(self, filename: str) -> pd.DataFrame:
        df = pd.read_table(f"{self.paths.basedir}{filename}", sep=r"\s+", header=0, index_col=0)
        # Ensure consistent dtypes
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        return df

    def _select_param_columns(self, df: pd.DataFrame) -> np.ndarray:
        if self.param_subset is not None:
            missing = [c for c in self.param_subset if c not in df.columns]
            if missing:
                raise KeyError(f"Missing parameter columns: {missing}")
            cols = self.param_subset
        else:
            cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.param_names_ = cols  # keep last-used cols (28D typical)
        X = df[cols].to_numpy(dtype=float)
        return X

    def _load_stack_smfs(self, pattern: str) -> np.ndarray:
        arrays = []
        for snap in self.snapshots:
            path = f"{self.paths.basedir}{pattern.format(snap=snap)}"
            arr = np.load(path, allow_pickle=True)
            if arr.ndim != 2 or arr.shape[1] != self.B:
                raise ValueError(f"Unexpected SMF shape in {path}: {arr.shape}, expected (N, {self.B})")
            arrays.append(arr)
        return np.concatenate(arrays, axis=1)  # (N, B*S)

    def _build_targets(self, phi_stack: np.ndarray) -> np.ndarray:
        if self.y_transform == "phi":
            return phi_stack.astype(float)
        elif self.y_transform == "counts":
            return to_counts(phi_stack, self.dlog10M, self.Lbox)
        elif self.y_transform == "anscombe":
            n = to_counts(phi_stack, self.dlog10M, self.Lbox)
            return anscombe(n)
        else:
            raise ValueError("y_transform must be 'phi', 'counts', or 'anscombe'")

    # ------------------------- Public helpers -------------------------
    def get_training(self, fidelity: Literal["n128", "n256", "n512"]) -> Tuple[np.ndarray, np.ndarray]:
        X = {"n128": self.X128, "n256": self.X256, "n512": self.X512}[fidelity]
        Y = {"n128": self.Y128, "n256": self.Y256, "n512": self.Y512}[fidelity]
        return X, Y

    def get_training_stats(self, fidelity: Literal["n128", "n256", "n512"]) -> Dict[str, Dict[str, np.ndarray]]:
        Xstats = {"n128": self.X128_stats, "n256": self.X256_stats, "n512": self.X512_stats}[fidelity]
        Ystats = {"n128": self.Y128_stats, "n256": self.Y256_stats, "n512": self.Y512_stats}[fidelity]
        return {"X": Xstats, "Y": Ystats}

    # ------------------------- Plotting -------------------------
    def plot_smf_triplet(self, sim_idx: int, ax: Optional[plt.Axes] = None, labels: Optional[List[str]] = None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        labels = labels or ["n128", "n256", "n512"]
        colors = ["C0", "C1", "C3"]
        for (phi, lab, col) in [
            (self.PHI128, labels[0], colors[0]),
            (self.PHI256, labels[1], colors[1]),
            (self.PHI512, labels[2], colors[2]),
        ]:
            for s_i, snap in enumerate(self.snapshots):
                start, end = s_i * self.B, (s_i + 1) * self.B
                ax.plot(self.log10M, phi[sim_idx, start:end], marker="o", label=f"{lab} (snap {snap})", color=col, alpha=0.9 - 0.15 * s_i)
        ax.set_xlabel(r"$\log_{10} M_\star\,[M_\odot]$")
        ax.set_ylabel(r"$\phi\;[h^3\,\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(ncol=2)
        return ax

    def plot_uncertainty_bands(
        self,
        fidelity: Literal["n128", "n256", "n512"],
        sim_idx: int,
        count_model: CountModel = "jeffreys",
        frac_floor: float = 0.0,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        phi_map = {"n128": self.PHI128, "n256": self.PHI256, "n512": self.PHI512}
        phi = phi_map[fidelity]
        _, sigma_phi, _ = counts_uncertainty(phi, self.dlog10M, self.Lbox, count_model, frac_floor)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        for s_i, snap in enumerate(self.snapshots):
            start, end = s_i * self.B, (s_i + 1) * self.B
            m = phi[sim_idx, start:end]
            s = sigma_phi[sim_idx, start:end]
            ax.plot(self.log10M, m, label=f"{fidelity} (snap {snap})")
            ax.fill_between(self.log10M, m - s, m + s, alpha=0.25)
        ax.set_xlabel(r"$\log_{10} M_\star$")
        ax.set_ylabel(r"$\phi$")
        ax.legend()
        ax.grid(True, ls=":", alpha=0.4)
        return ax


# ------------------------- Minimal usage example -------------------------
if __name__ == "__main__":
    basedir = "../../data/SB28/ming-feng/"
    paths = SB28Paths(basedir=basedir)
    log10M = np.array(
        [8.15, 8.45, 8.75, 9.05, 9.35, 9.65, 9.95, 10.25, 10.55, 10.85, 11.15, 11.45, 11.75],
        dtype=np.float32,
    )
    snapshots = [73, 90]

    # Build loader with Anscombe targets + z-scoring for X and Y
    loader = SMFDataLoaderSB28(
        paths, snapshots, log10M_centers=log10M, y_transform="anscombe", standardize_X=True, standardize_Y=True
    )

    # Normalized training matrices
    X_n512, Y_n512 = loader.get_training("n512")
    stats_n512 = loader.get_training_stats("n512")

    # Quick plots
    loader.plot_smf_triplet(sim_idx=0)
    plt.show()
    loader.plot_uncertainty_bands("n512", sim_idx=0, count_model="jeffreys", frac_floor=0.01)
    plt.show()