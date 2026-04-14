#!/usr/bin/env python3
"""
Water Quality Analysis — Task 3
---------------------------------
1. False-color composites from hyperspectral data cube
2. Water quality indices: Chl-a, DOC, Turbidity
3. Sentinel-2 data download (closest acquisition date)  ← PLACEHOLDER section
4. Same indices from Sentinel-2 + comparison

Usage:
    python water_quality.py path/to/file.hdr

Requirements:
    pip install spectral numpy matplotlib rasterio pystac-client requests
    (sentinelhub or planetary-computer optional for S2 download)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import spectral.io.envi as envi
except ImportError:
    sys.exit("Missing: pip install spectral")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent / "data" / "images"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Sentinel-2 placeholder path — replace with real downloaded file
S2_PLACEHOLDER = Path(__file__).parent / "data" / "sentinel2_placeholder.tif"

# Cloud cover threshold for S2 search
S2_MAX_CLOUD = 20

# ─────────────────────────────────────────────────────────────────────────────
# ENVI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_hdr(hdr_path: Path):
    """Open ENVI image (memory-mapped)."""
    return envi.open(str(hdr_path))


def get_wavelengths(img) -> np.ndarray:
    wl = img.metadata.get("wavelength")
    if not wl:
        raise ValueError("No wavelength info in HDR. Cannot compute spectral indices.")
    return np.array([float(w) for w in wl])


def get_acquisition_date(img) -> str | None:
    """Try to read acquisition date from ENVI metadata."""
    for key in ("acquisition date", "acquisition time", "date"):
        val = img.metadata.get(key)
        if val:
            return str(val).strip()
    return None


def get_ignore_value(img) -> float | None:
    raw = img.metadata.get("data ignore value")
    if raw:
        try:
            return float(str(raw).strip())
        except ValueError:
            pass
    return None


def get_extent(img) -> dict | None:
    """Extract spatial extent from ENVI map info if available."""
    map_info = img.metadata.get("map info")
    if not map_info:
        return None
    try:
        parts = [p.strip() for p in map_info]
        # map info: {proj, ref_col, ref_row, x_start, y_start, x_ps, y_ps, ...}
        x0 = float(parts[3])
        y0 = float(parts[4])
        ps_x = float(parts[5])
        ps_y = float(parts[6])
        x1 = x0 + img.ncols * ps_x
        y1 = y0 - img.nrows * ps_y
        return {"x_min": x0, "x_max": x1, "y_min": y1, "y_max": y0,
                "ps_x": ps_x, "ps_y": ps_y}
    except Exception:
        return None


def wavelength_to_band(wavelengths: np.ndarray, target_nm: float, tolerance: float = 10.0) -> int:
    """Return 0-based band index closest to target wavelength."""
    diffs = np.abs(wavelengths - target_nm)
    idx = int(np.argmin(diffs))
    if diffs[idx] > tolerance:
        raise ValueError(
            f"No band within {tolerance} nm of {target_nm} nm. "
            f"Closest: {wavelengths[idx]:.1f} nm (diff={diffs[idx]:.1f} nm)."
        )
    return idx


def read_band(img, band_idx: int, ignore_value: float | None) -> np.ndarray:
    """Read a single band as float32, masking no-data values as NaN."""
    data = img.read_bands([band_idx]).squeeze().astype(np.float32)
    if ignore_value is not None:
        data[data >= ignore_value] = np.nan
    data[data < 0] = np.nan
    return data


def read_rgb_stretch(img, r: int, g: int, b: int, ignore_value: float | None) -> np.ndarray:
    """Read 3 bands and apply 2–98% percentile stretch for display."""
    rgb = img.read_bands([r, g, b]).astype(np.float32)
    if ignore_value is not None:
        rgb[rgb >= ignore_value] = np.nan
    rgb[rgb < 0] = np.nan
    for c in range(3):
        ch = rgb[:, :, c]
        p2, p98 = np.nanpercentile(ch, [2, 98])
        rgb[:, :, c] = np.clip((ch - p2) / max(p98 - p2, 1e-6), 0, 1)
    return np.nan_to_num(rgb, nan=0.0)


def get_default_rgb_bands(img) -> tuple[int, int, int]:
    db = img.metadata.get("default bands")
    if db and len(db) >= 3:
        return tuple(int(float(v)) - 1 for v in db[:3])
    return (30, 20, 10)

# ─────────────────────────────────────────────────────────────────────────────
# 1. FALSE-COLOR COMPOSITES
# ─────────────────────────────────────────────────────────────────────────────

def make_false_color_composites(img, wavelengths: np.ndarray, ignore_value: float | None):
    """
    Generate and save 4 false-color composites:
      - Natural color (RGB)
      - NIR–Red–Green (vegetation contrast)
      - Red-edge–NIR–Blue (water/vegetation boundary)
      - SWIR–NIR–Red (sediment / water turbidity contrast)
    """
    print("\n[1] False-color composites")

    def safe_band(nm, tol=15):
        try:
            return wavelength_to_band(wavelengths, nm, tol)
        except ValueError as e:
            print(f"    WARNING: {e}")
            return None

    # Wavelength targets (nm)
    B_nm,  G_nm,  R_nm  = 470,  550,  660
    RE_nm, NIR_nm        = 720,  800
    SWIR_nm              = 1600  # may not be in airborne VIS/NIR sensor

    composites = []

    # Natural color
    r, g, b = safe_band(R_nm), safe_band(G_nm), safe_band(B_nm)
    if None not in (r, g, b):
        composites.append(("Natural color (R-G-B)", r, g, b))

    # NIR–Red–Green
    nir = safe_band(NIR_nm)
    if None not in (nir, r, g):
        composites.append(("NIR – Red – Green", nir, r, g))

    # Red-edge–NIR–Blue
    re = safe_band(RE_nm)
    if None not in (re, nir, b):
        composites.append(("Red-edge – NIR – Blue", re, nir, b))

    # SWIR–NIR–Red (only if sensor covers SWIR)
    swir = safe_band(SWIR_nm, tol=30)
    if None not in (swir, nir, r):
        composites.append(("SWIR – NIR – Red", swir, nir, r))
    else:
        # Fallback: use default bands
        dr, dg, db = get_default_rgb_bands(img)
        composites.append((f"Default bands ({dr},{dg},{db})", dr, dg, db))

    n = len(composites)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (title, ri, gi, bi) in zip(axes, composites):
        rgb = read_rgb_stretch(img, ri, gi, bi, ignore_value)
        ax.imshow(rgb, aspect="auto", interpolation="bilinear")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle("False-Color Composites — Hyperspectral", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "false_color_composites.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")
    return composites

# ─────────────────────────────────────────────────────────────────────────────
# 2. WATER QUALITY INDICES — HYPERSPECTRAL
# ─────────────────────────────────────────────────────────────────────────────

def compute_water_quality_indices(img, wavelengths: np.ndarray, ignore_value: float | None) -> dict:
    """
    Compute Chl-a, DOC and Turbidity indices from hyperspectral bands.

    Formulas used
    -------------
    Chl-a     : (R700 - R670) / (R700 + R670)   — red-edge ratio
                 (sensitive to chlorophyll absorption at 670 nm)
    DOC       : R440 / R490                       — blue ratio
                 (CDOM / dissolved organic carbon absorption)
    Turbidity : R860 / R550                       — NIR/green ratio
                 (high NIR reflectance → scattering by suspended sediment)
    """
    print("\n[2] Computing water quality indices (hyperspectral)")

    def band(nm, tol=10):
        return wavelength_to_band(wavelengths, nm, tol)

    def rb(nm, tol=10):
        return read_band(img, band(nm, tol), ignore_value)

    indices = {}

    # ── Chl-a ──
    try:
        R670 = rb(670)
        R700 = rb(700)
        chla = (R700 - R670) / (R700 + R670 + 1e-9)
        indices["Chl-a"] = chla
        print(f"    Chl-a   : min={np.nanmin(chla):.3f}  max={np.nanmax(chla):.3f}  "
              f"mean={np.nanmean(chla):.3f}")
    except ValueError as e:
        print(f"    Chl-a SKIPPED: {e}")

    # ── DOC ──
    try:
        R440 = rb(440)
        R490 = rb(490)
        doc  = R440 / (R490 + 1e-9)
        indices["DOC"] = doc
        print(f"    DOC     : min={np.nanmin(doc):.3f}  max={np.nanmax(doc):.3f}  "
              f"mean={np.nanmean(doc):.3f}")
    except ValueError as e:
        print(f"    DOC SKIPPED: {e}")

    # ── Turbidity ──
    try:
        R860 = rb(860, tol=20)
        R550 = rb(550)
        turb = R860 / (R550 + 1e-9)
        indices["Turbidity"] = turb
        print(f"    Turbidity: min={np.nanmin(turb):.3f}  max={np.nanmax(turb):.3f}  "
              f"mean={np.nanmean(turb):.3f}")
    except ValueError as e:
        print(f"    Turbidity SKIPPED: {e}")

    if not indices:
        print("    No indices could be computed — check wavelength range of your sensor.")
        return indices

    # ── Plot ──
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    cmaps = {"Chl-a": "YlGn", "DOC": "YlOrBr", "Turbidity": "Blues"}
    for ax, (name, data) in zip(axes, indices.items()):
        vmin, vmax = np.nanpercentile(data, [2, 98])
        im = ax.imshow(data, cmap=cmaps.get(name, "viridis"),
                       vmin=vmin, vmax=vmax, aspect="auto", interpolation="bilinear")
        ax.set_title(name, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Water Quality Indices — Hyperspectral", fontsize=13)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "indices_hyperspectral.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")

    return indices


# ─────────────────────────────────────────────────────────────────────────────
# 3. SENTINEL-2 DOWNLOAD  ← PLACEHOLDER
# ─────────────────────────────────────────────────────────────────────────────

def search_and_download_sentinel2(img, extent: dict | None) -> Path | None:
    """
    Search Copernicus Data Space for a Sentinel-2 scene acquired as close
    as possible to the airborne acquisition date, then download it.

    ── HOW TO FILL THIS IN ──────────────────────────────────────────────────
    1. Register at https://dataspace.copernicus.eu (free).
    2. Set env vars:  CDSE_USER / CDSE_PASSWORD
    3. pip install sentinelhub  OR  pip install odc-stac pystac-client

    Example using pystac-client + direct S3 download:

        from pystac_client import Client
        import os, requests

        catalog = Client.open(
            "https://catalogue.dataspace.copernicus.eu/stac",
            headers={"Authorization": f"Bearer {token}"}
        )
        items = catalog.search(
            collections=["SENTINEL-2"],
            bbox=[extent["x_min"], extent["y_min"],
                  extent["x_max"], extent["y_max"]],
            datetime=f"{date_minus_30d}/{date_plus_30d}",
            query={"eo:cloud_cover": {"lt": S2_MAX_CLOUD}}
        ).item_collection()

        # Sort by date proximity to airborne acquisition
        closest = min(items, key=lambda i: abs(
            (i.datetime - acq_datetime).days
        ))
        print(f"Best S2 scene: {closest.id}  date: {closest.datetime.date()}")

        # Download B03 (Green), B04 (Red), B05 (Red-edge), B8A (NIR)
        # ... download logic here ...

    ─────────────────────────────────────────────────────────────────────────
    """
    print("\n[3] Sentinel-2 download — PLACEHOLDER")
    acq_date = get_acquisition_date(img)
    print(f"    Airborne acquisition date from HDR: {acq_date or 'NOT FOUND in metadata'}")
    print("    ⚠  S2 download not implemented yet.")
    print(f"    → Place your downloaded S2 GeoTIFF at: {S2_PLACEHOLDER}")
    print("    → It should contain bands: B03 (Green), B04 (Red),")
    print("      B05 (Red-edge 705nm), B8A (NIR 865nm), stacked in that order.")

    if S2_PLACEHOLDER.exists():
        print(f"    Found placeholder file: {S2_PLACEHOLDER}")
        return S2_PLACEHOLDER

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. WATER QUALITY INDICES — SENTINEL-2
# ─────────────────────────────────────────────────────────────────────────────

def compute_s2_indices(s2_path: Path) -> dict:
    """
    Compute the same water quality indices from a Sentinel-2 GeoTIFF.
    Assumes stacked bands: B03=Green(560nm), B04=Red(665nm),
                           B05=RedEdge(705nm), B8A=NIR(865nm)

    Band mapping to match hyperspectral formulas:
      Chl-a     : (B05 - B04) / (B05 + B04)   — red-edge ratio
      DOC       : B03 / B04                    — green/red (CDOM proxy)
      Turbidity : B8A / B03                    — NIR/green ratio
    """
    print("\n[4] Computing water quality indices (Sentinel-2)")

    try:
        import rasterio
    except ImportError:
        print("    Missing: pip install rasterio  — skipping S2 indices.")
        return {}

    if not s2_path or not s2_path.exists():
        print("    No S2 file found — skipping.")
        print("    To enable: download a S2 scene and update S2_PLACEHOLDER path.")
        return {}

    with rasterio.open(s2_path) as src:
        count = src.count
        print(f"    S2 file: {s2_path.name}  |  bands={count}  "
              f"shape={src.height}×{src.width}")

        if count < 4:
            print(f"    Expected 4 bands (B03,B04,B05,B8A), got {count}. Skipping.")
            return {}

        B03 = src.read(1).astype(np.float32)  # Green  560 nm
        B04 = src.read(2).astype(np.float32)  # Red    665 nm
        B05 = src.read(3).astype(np.float32)  # RE     705 nm
        B8A = src.read(4).astype(np.float32)  # NIR    865 nm

    nodata = 0.0
    for arr in (B03, B04, B05, B8A):
        arr[arr <= nodata] = np.nan

    indices = {}

    chla_s2 = (B05 - B04) / (B05 + B04 + 1e-9)
    doc_s2  = B03 / (B04 + 1e-9)
    turb_s2 = B8A / (B03 + 1e-9)

    indices = {"Chl-a": chla_s2, "DOC": doc_s2, "Turbidity": turb_s2}

    for name, data in indices.items():
        print(f"    {name:12s}: min={np.nanmin(data):.3f}  max={np.nanmax(data):.3f}  "
              f"mean={np.nanmean(data):.3f}")

    # Plot
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    cmaps = {"Chl-a": "YlGn", "DOC": "YlOrBr", "Turbidity": "Blues"}
    for ax, (name, data) in zip(axes, indices.items()):
        vmin, vmax = np.nanpercentile(data, [2, 98])
        im = ax.imshow(data, cmap=cmaps.get(name, "viridis"),
                       vmin=vmin, vmax=vmax, aspect="auto", interpolation="bilinear")
        ax.set_title(name, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Water Quality Indices — Sentinel-2", fontsize=13)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "indices_sentinel2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")

    return indices


# ─────────────────────────────────────────────────────────────────────────────
# 5. COMPARISON PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def compare_indices(hs_indices: dict, s2_indices: dict):
    """
    Side-by-side maps + scatterplots comparing hyperspectral vs Sentinel-2 indices.
    Resamples hyperspectral to S2 resolution by block-averaging.
    """
    print("\n[5] Comparing indices")

    shared = [k for k in hs_indices if k in s2_indices]
    if not shared:
        print("    No shared indices to compare — skipping.")
        print("    (This section requires both hyperspectral and S2 indices.)")
        return

    cmaps = {"Chl-a": "YlGn", "DOC": "YlOrBr", "Turbidity": "Blues"}

    for name in shared:
        hs = hs_indices[name]   # (H_hs, W_hs)
        s2 = s2_indices[name]   # (H_s2, W_s2)

        # ── Resample hs → s2 resolution by simple block average ──
        scale_r = hs.shape[0] / s2.shape[0]
        scale_c = hs.shape[1] / s2.shape[1]

        hs_resampled = np.full_like(s2, np.nan)
        for r in range(s2.shape[0]):
            for c in range(s2.shape[1]):
                r0 = int(r * scale_r)
                r1 = int((r + 1) * scale_r)
                c0 = int(c * scale_c)
                c1 = int((c + 1) * scale_c)
                block = hs[r0:r1, c0:c1]
                if block.size > 0:
                    hs_resampled[r, c] = np.nanmean(block)

        # ── Figure: side-by-side maps + scatterplot ──
        fig = plt.figure(figsize=(18, 5))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

        cmap = cmaps.get(name, "viridis")
        vmin = np.nanpercentile(np.concatenate([hs_resampled.ravel(), s2.ravel()]), 2)
        vmax = np.nanpercentile(np.concatenate([hs_resampled.ravel(), s2.ravel()]), 98)
        norm = Normalize(vmin=vmin, vmax=vmax)

        ax1 = fig.add_subplot(gs[0])
        im1 = ax1.imshow(hs_resampled, cmap=cmap, norm=norm, aspect="auto")
        ax1.set_title(f"{name}\nHyperspectral (resampled)", fontsize=10)
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[1])
        im2 = ax2.imshow(s2, cmap=cmap, norm=norm, aspect="auto")
        ax2.set_title(f"{name}\nSentinel-2", fontsize=10)
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Scatterplot
        ax3 = fig.add_subplot(gs[2])
        x = hs_resampled.ravel()
        y = s2.ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if len(x) > 0:
            ax3.scatter(x, y, alpha=0.15, s=2, color="steelblue", rasterized=True)

            # 1:1 line
            lim_min = min(x.min(), y.min())
            lim_max = max(x.max(), y.max())
            ax3.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.2, label="1:1")

            # Pearson r
            corr = np.corrcoef(x, y)[0, 1]
            ax3.set_title(f"{name} — Scatter\nr = {corr:.3f}", fontsize=10)
            ax3.set_xlabel("Hyperspectral")
            ax3.set_ylabel("Sentinel-2")
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No overlapping valid pixels",
                     ha="center", va="center", transform=ax3.transAxes, color="gray")

        fig.suptitle(f"Comparison: {name}", fontsize=13)
        out_path = OUTPUT_DIR / f"comparison_{name.replace('-','').replace(' ','_')}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved → {out_path}")

    print(f"\n    NOTE: Resampling here is naive (block average).")
    print(f"    For a proper spatial comparison, co-register both datasets")
    print(f"    using rasterio.warp.reproject to a common CRS + resolution.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def find_hdr(data_dir: Path) -> Path | None:
    hdrs = sorted(data_dir.glob("*.hdr"))
    if len(hdrs) == 1:
        return hdrs[0]
    if len(hdrs) > 1:
        print("Multiple HDR files found:")
        for i, h in enumerate(hdrs):
            print(f"  [{i}] {h.name}")
        idx = int(input("Select index: "))
        return hdrs[idx]
    return None


def main():
    # ── Locate HDR ──
    if len(sys.argv) > 1:
        hdr_path = Path(sys.argv[1])
    else:
        hdr_path = find_hdr(DATA_DIR)

    if not hdr_path or not hdr_path.exists():
        sys.exit(f"HDR file not found. Usage: python water_quality.py path/to/file.hdr")

    print(f"Loading: {hdr_path.name}")
    img          = load_hdr(hdr_path)
    wavelengths  = get_wavelengths(img)
    ignore_value = get_ignore_value(img)
    extent       = get_extent(img)

    print(f"  Shape      : {img.nrows} lines × {img.ncols} samples × {img.nbands} bands")
    print(f"  Wavelengths: {wavelengths[0]:.1f} – {wavelengths[-1]:.1f} nm")
    print(f"  Ignore val : {ignore_value}")
    print(f"  Extent     : {extent}")

    # ── Run pipeline ──
    make_false_color_composites(img, wavelengths, ignore_value)
    hs_indices  = compute_water_quality_indices(img, wavelengths, ignore_value)
    s2_path     = search_and_download_sentinel2(img, extent)
    s2_indices  = compute_s2_indices(s2_path)
    compare_indices(hs_indices, s2_indices)

    print(f"\n✔ Done. All outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()