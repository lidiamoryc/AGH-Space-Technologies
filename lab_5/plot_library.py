#!/usr/bin/env python3
"""
Spectral Library Viewer
-----------------------
Load one or more spectral library CSV files (produced by viewer.py)
and display them together: mean spectrum + ±1 std shaded envelope.

Usage:
    # Single file:
    python plot_library.py water.csv

    # Multiple files (all classes at once):
    python plot_library.py water.csv forest.csv urban.csv

    # No arguments — opens a file picker dialog:
    python plot_library.py

Requires: numpy, matplotlib
"""

import sys
import csv
from pathlib import Path
from tkinter import filedialog, Tk

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Colour palette — one per class name (fallback cycles if unknown) ──────────
CLASS_COLORS = {
    "water":       "#2196F3",   # blue
    "urban":       "#9E9E9E",   # grey
    "green areas": "#8BC34A",   # light green
    "forest":      "#2E7D32",   # dark green
    "field":       "#FFC107",   # amber
}
FALLBACK_COLORS = ["#E91E63", "#9C27B0", "#FF5722", "#00BCD4", "#FF9800"]


# ── CSV reader ────────────────────────────────────────────────────────────────

def load_library_csv(path: Path) -> dict:
    """
    Read a spectral library CSV.
    Returns dict with keys:
        wavelength  np.ndarray (nbands,)
        mean        np.ndarray (nbands,)
        std         np.ndarray (nbands,)
        min         np.ndarray (nbands,)
        max         np.ndarray (nbands,)
        n_pixels    int
        class_name  str
        path        Path
    """
    wavelength, mean, std, mn, mx = [], [], [], [], []
    class_name = "unknown"
    n_pixels = 0
    has_wavelength = True

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wl_key = "wavelength_nm" if "wavelength_nm" in row else "band"
            has_wavelength = wl_key == "wavelength_nm"
            wavelength.append(float(row[wl_key]))
            mean.append(float(row["mean"]) if row["mean"] else np.nan)
            std.append(float(row["std"])  if row["std"]  else np.nan)
            mn.append(float(row["min"])   if row["min"]  else np.nan)
            mx.append(float(row["max"])   if row["max"]  else np.nan)
            if row.get("n_pixels"):
                n_pixels = int(row["n_pixels"])
            if row.get("class_name"):
                class_name = row["class_name"].strip()

    return {
        "wavelength":    np.array(wavelength),
        "mean":          np.array(mean),
        "std":           np.array(std),
        "min":           np.array(mn),
        "max":           np.array(mx),
        "n_pixels":      n_pixels,
        "class_name":    class_name,
        "has_wavelength": has_wavelength,
        "path":          path,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_library(entries: list[dict]):
    """
    Plot all library entries on a single axes.
    Each entry gets:
      - solid line  : mean spectrum
      - shaded band : mean ± 1 std
      - dashed lines: min / max (optional, toggle via SHOW_MINMAX)
    """
    SHOW_MINMAX = len(entries) == 1   # show min/max only when a single class

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    fallback_idx = 0

    for entry in entries:
        cname = entry["class_name"]
        color = CLASS_COLORS.get(cname)
        if color is None:
            color = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1

        x    = entry["wavelength"]
        mu   = entry["mean"]
        sig  = entry["std"]
        lo   = mu - sig
        hi   = mu + sig
        label = f"{cname}  (n={entry['n_pixels']:,})"

        # ±1 std envelope
        ax.fill_between(x, lo, hi, color=color, alpha=0.20, linewidth=0)

        # mean line
        ax.plot(x, mu, color=color, linewidth=1.8, label=label)

        # min/max dashed (only for single-entry view)
        if SHOW_MINMAX:
            ax.plot(x, entry["min"], color=color, linewidth=0.8,
                    linestyle="--", alpha=0.5, label=f"{cname} min")
            ax.plot(x, entry["max"], color=color, linewidth=0.8,
                    linestyle=":",  alpha=0.5, label=f"{cname} max")

    # ── axes styling ──────────────────────────────────────────────────────────
    ax.set_xlabel("Wavelength (nm)" if entries[0]["has_wavelength"] else "Band index",
                  color="#c9d1d9", fontsize=12)
    ax.set_ylabel("Reflectance", color="#c9d1d9", fontsize=12)
    ax.set_title("Spectral Library — Mean Signatures  (shading = ±1 std)",
                 color="#e6edf3", fontsize=14, pad=14)

    ax.tick_params(colors="#8b949e", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", color="#21262d", linewidth=0.8)
    ax.grid(which="minor", color="#161b22", linewidth=0.4)

    # visible light region annotation (only when wavelength axis is real)
    if entries[0]["has_wavelength"]:
        vis_bands = [
            (380, 450, "#7B68EE", "violet"),
            (450, 495, "#4169E1", "blue"),
            (495, 570, "#3CB371", "green"),
            (570, 620, "#FFD700", "yellow"),
            (620, 750, "#DC143C", "red"),
        ]
        ymin, ymax = ax.get_ylim()
        for wl0, wl1, c, _ in vis_bands:
            ax.axvspan(wl0, wl1, ymin=0, ymax=0.035,
                       color=c, alpha=0.55, linewidth=0)
        ax.text(565, ymin, "← VIS →", color="#8b949e",
                fontsize=7, va="bottom", ha="center")

    legend = ax.legend(
        facecolor="#21262d", edgecolor="#30363d",
        labelcolor="#c9d1d9", fontsize=10,
        loc="upper right", framealpha=0.9,
    )

    fig.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def pick_files() -> list[Path]:
    """Open a file-picker dialog and return selected paths."""
    root = Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(
        title="Select spectral library CSV files",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    return [Path(p) for p in paths]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_paths = [Path(p) for p in sys.argv[1:]]
    else:
        csv_paths = pick_files()

    if not csv_paths:
        print("No files selected. Exiting.")
        sys.exit(0)

    entries = []
    for p in csv_paths:
        if not p.exists():
            print(f"Warning: file not found — {p}")
            continue
        try:
            entries.append(load_library_csv(p))
            print(f"Loaded: {p.name}  |  class='{entries[-1]['class_name']}'  "
                  f"|  {entries[-1]['n_pixels']:,} pixels  "
                  f"|  {len(entries[-1]['wavelength'])} bands")
        except Exception as e:
            print(f"Error loading {p.name}: {e}")

    if not entries:
        print("No valid files loaded.")
        sys.exit(1)

    plot_library(entries)