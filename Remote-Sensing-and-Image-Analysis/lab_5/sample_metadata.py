#!/usr/bin/env python3
"""
print_geo_meta.py
-----------------
Read geographic location metadata from ENVI .hdr files and print it.

Usage:
    # Single file:
    python print_geo_meta.py image.hdr

    # All .hdr files in a directory:
    python print_geo_meta.py "G:/Mój dysk/mggp_naloty/Obrazy lotnicze"

    # No arguments — opens a file picker:
    python print_geo_meta.py
"""

import sys
from pathlib import Path
from tkinter import filedialog, Tk

try:
    import spectral.io.envi as envi
except ImportError:
    sys.exit("Install spectral:  pip install spectral")


# ── ENVI map info parser ──────────────────────────────────────────────────────

def parse_map_info(meta: dict) -> dict | None:
    """
    Parse the ENVI 'map info' field.
    Format:
      {projection, ref_pixel_x, ref_pixel_y, easting, northing,
       pixel_size_x, pixel_size_y, datum, units, ...}
    All values are comma-separated strings inside the header list.
    """
    raw = meta.get("map info")
    if not raw:
        return None

    # spectral returns it as a list of strings already split on commas
    parts = [p.strip() for p in raw]
    if len(parts) < 7:
        return None

    try:
        return {
            "projection":   parts[0],
            "ref_pixel_x":  float(parts[1]),   # reference pixel column (1-based)
            "ref_pixel_y":  float(parts[2]),   # reference pixel row    (1-based)
            "easting":      float(parts[3]),   # X coordinate of reference pixel
            "northing":     float(parts[4]),   # Y coordinate of reference pixel
            "pixel_size_x": float(parts[5]),   # pixel width  in map units
            "pixel_size_y": float(parts[6]),   # pixel height in map units
            "datum":        parts[7] if len(parts) > 7 else "unknown",
            "units":        parts[8] if len(parts) > 8 else "unknown",
            "raw":          parts,
        }
    except ValueError:
        return {"raw": parts}


def parse_coordinate_system(meta: dict) -> str | None:
    """Return the coordinate system string (WKT or PROJ) if present."""
    return meta.get("coordinate system string") or meta.get("projection info")


def compute_corners(meta: dict, map_info: dict) -> dict | None:
    """
    Compute the four corner coordinates from map info + image dimensions.
    Works for both projected (UTM etc.) and geographic (lat/lon) CRS.
    """
    try:
        nrows = int(meta.get("lines", 0))
        ncols = int(meta.get("samples", 0))
        if nrows == 0 or ncols == 0:
            return None

        # Reference pixel (1-based) and its map coordinates
        ref_col = map_info["ref_pixel_x"] - 1   # convert to 0-based
        ref_row = map_info["ref_pixel_y"] - 1

        dx = map_info["pixel_size_x"]
        dy = map_info["pixel_size_y"]   # always positive in ENVI convention

        # Upper-left corner of pixel (0,0)
        ul_x = map_info["easting"]  - ref_col * dx
        ul_y = map_info["northing"] + ref_row * dy   # Y increases upward

        # Other corners
        ur_x = ul_x + (ncols - 1) * dx
        ur_y = ul_y

        ll_x = ul_x
        ll_y = ul_y - (nrows - 1) * dy

        lr_x = ur_x
        lr_y = ll_y

        center_x = ul_x + (ncols / 2) * dx
        center_y = ul_y - (nrows / 2) * dy

        return {
            "upper_left":  (ul_x, ul_y),
            "upper_right": (ur_x, ur_y),
            "lower_left":  (ll_x, ll_y),
            "lower_right": (lr_x, lr_y),
            "center":      (center_x, center_y),
        }
    except Exception:
        return None


# ── Printer ───────────────────────────────────────────────────────────────────

def print_hdr_geo(hdr_path: Path):
    print(f"\n{'═'*60}")
    print(f"  {hdr_path.name}")
    print(f"{'═'*60}")

    img  = envi.open(str(hdr_path))
    meta = img.metadata

    # Basic dimensions
    nrows  = meta.get("lines",   "?")
    ncols  = meta.get("samples", "?")
    nbands = meta.get("bands",   "?")
    print(f"  Dimensions   : {nrows} lines × {ncols} samples × {nbands} bands")

    # Map info
    map_info = parse_map_info(meta)
    if map_info:
        print(f"\n  Projection   : {map_info.get('projection', '?')}")
        print(f"  Datum        : {map_info.get('datum', '?')}")
        print(f"  Units        : {map_info.get('units', '?')}")
        print(f"  Pixel size   : {map_info.get('pixel_size_x', '?')} × "
              f"{map_info.get('pixel_size_y', '?')}  (X × Y)")
        print(f"  Reference px : col={map_info.get('ref_pixel_x')}, "
              f"row={map_info.get('ref_pixel_y')}")
        print(f"  Reference XY : easting={map_info.get('easting')}, "
              f"northing={map_info.get('northing')}")

        corners = compute_corners(meta, map_info)
        if corners:
            u = map_info.get("units", "")
            print(f"\n  ── Corner coordinates ({u}) ──")
            for name, (x, y) in corners.items():
                print(f"    {name:<14}: X={x:.3f}   Y={y:.3f}")
    else:
        print("  ⚠  No 'map info' field found in header.")
        print("     The image may not be georeferenced.")

    # Coordinate system string (WKT)
    crs = parse_coordinate_system(meta)
    if crs:
        # Print just the first line — WKT can be very long
        first_line = str(crs).split("\n")[0][:120]
        print(f"\n  CRS (first line): {first_line}")

    # Any other potentially geographic keys
    extra_keys = ["utm zone", "zone", "datum", "pixel size"]
    for key in extra_keys:
        val = meta.get(key)
        if val:
            print(f"  {key:<16}: {val}")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def collect_hdr_paths(args: list[str]) -> list[Path]:
    if not args:
        root = Tk()
        root.withdraw()
        chosen = filedialog.askopenfilenames(
            title="Select ENVI .hdr files",
            filetypes=[("ENVI header", "*.hdr"), ("All files", "*.*")],
        )
        root.destroy()
        return [Path(p) for p in chosen]

    paths = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.hdr")))
        elif p.suffix.lower() == ".hdr" and p.exists():
            paths.append(p)
        else:
            print(f"Skipping (not found or not .hdr): {arg}")
    return paths


if __name__ == "__main__":
    hdr_paths = collect_hdr_paths(sys.argv[1:])

    if not hdr_paths:
        print("No .hdr files found or selected.")
        sys.exit(0)

    for hdr in hdr_paths:
        try:
            print_hdr_geo(hdr)
        except Exception as e:
            print(f"  ERROR reading {hdr.name}: {e}\n")