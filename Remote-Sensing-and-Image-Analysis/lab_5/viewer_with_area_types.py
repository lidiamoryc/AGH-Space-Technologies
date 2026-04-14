#!/usr/bin/env python3
"""
Hyperspectral BSQ Viewer
------------------------
Browse ENVI/BSQ hyperspectral data cubes.
Click any pixel in the RGB preview to display its full spectral signature.
Export the selected spectrum to CSV.
Draw rectangles to extract spectral statistics for a land cover library.

Usage:
    python viewer.py [path/to/file.hdr]

If no path is given the tool searches data/images/ for .hdr files.
Requires Python 3.10+  (uses X | Y union type hints)
"""

import sys
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector                          # ── NEW

try:
    import spectral.io.envi as envi
except ImportError:
    sys.exit(
        "The 'spectral' library is missing.\n"
        "Install it with:  pip install spectral"
    )

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = Path("G:\Mój dysk\mggp_naloty\Obrazy lotnicze")
FALLBACK_RGB = (30, 20, 10)

# ── NEW: land cover classes ───────────────────────────────────────────────────
LAND_COVER_CLASSES = ["water", "urban", "green areas", "forest", "field"]

# ── ENVI header helpers ───────────────────────────────────────────────────────

def find_hdr_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.hdr"))


def parse_wavelengths(meta: dict) -> np.ndarray | None:
    wl = meta.get("wavelength")
    if wl:
        return np.array([float(w) for w in wl])
    return None


def get_rgb_bands(meta: dict) -> tuple[int, int, int]:
    db = meta.get("default bands")
    if db and len(db) >= 3:
        return tuple(int(float(v)) - 1 for v in db[:3])
    return FALLBACK_RGB


def get_ignore_value(meta: dict) -> float | None:
    raw = meta.get("data ignore value")
    if raw:
        try:
            return float(str(raw).strip())
        except ValueError:
            pass
    return None


# ── Image I/O ─────────────────────────────────────────────────────────────────

def load_image(hdr_path: Path):
    return envi.open(str(hdr_path))


def read_rgb(img, r: int, g: int, b: int, ignore_value: float | None, decimate: int = 10) -> np.ndarray:
    full = img.read_bands([r, g, b]).astype(np.float32)
    rgb = full[::decimate, ::decimate, :]

    if ignore_value is not None:
        rgb[rgb >= ignore_value] = np.nan
    rgb[rgb < 0] = np.nan

    for c in range(3):
        ch = rgb[:, :, c]
        p2, p98 = np.nanpercentile(ch, [2, 98])
        rgb[:, :, c] = np.clip((ch - p2) / max(p98 - p2, 1e-6), 0, 1)

    return np.nan_to_num(rgb, nan=0.0)


def read_spectrum(img, row: int, col: int, ignore_value: float | None) -> np.ndarray:
    patch = img.read_subimage([row], [col])   # shape: (1, 1, nbands)
    spec = patch[0, 0, :].astype(np.float64)
    if ignore_value is not None:
        spec[spec >= ignore_value] = np.nan
    spec[spec < 0] = np.nan
    return spec


# ── NEW: streaming statistics extraction ─────────────────────────────────────

def extract_roi_statistics(
    img,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    ignore_value: float | None,
    progress_callback=None,
) -> dict:
    """
    Stream through the rectangle row by row, accumulating statistics using
    Welford's online algorithm. RAM usage is O(ncols_in_rect × nbands) —
    one row at a time — regardless of rectangle height.

    Returns a dict with keys: n_pixels, mean, std, min, max
    All arrays have shape (nbands,).
    """
    nbands = img.nbands
    cols = list(range(col_start, col_end + 1))
    n_cols = len(cols)
    n_rows = row_end - row_start + 1

    # Welford accumulators — shape (nbands,)
    count = np.zeros(nbands, dtype=np.int64)   # valid (non-NaN) pixel count per band
    mean  = np.zeros(nbands, dtype=np.float64)
    M2    = np.zeros(nbands, dtype=np.float64)  # sum of squared deviations
    running_min = np.full(nbands, np.inf)
    running_max = np.full(nbands, -np.inf)

    for i, row in enumerate(range(row_start, row_end + 1)):
        # Read one full-width stripe: shape (1, n_cols, nbands)
        stripe = img.read_subimage([row], cols).astype(np.float64)
        stripe = stripe[0, :, :]   # shape (n_cols, nbands)

        # Mask bad values
        if ignore_value is not None:
            stripe[stripe >= ignore_value] = np.nan
        stripe[stripe < 0] = np.nan

        # Update Welford accumulators per band
        for b in range(nbands):
            col_vals = stripe[:, b]
            valid = col_vals[~np.isnan(col_vals)]
            if valid.size == 0:
                continue
            for x in valid:
                count[b] += 1
                delta = x - mean[b]
                mean[b] += delta / count[b]
                delta2 = x - mean[b]
                M2[b] += delta * delta2
            running_min[b] = min(running_min[b], valid.min())
            running_max[b] = max(running_max[b], valid.max())

        if progress_callback:
            progress_callback(i + 1, n_rows)

    # Finalise std — set to NaN where no valid pixels
    with np.errstate(invalid="ignore"):
        std = np.where(count > 1, np.sqrt(M2 / (count - 1)), 0.0)

    mean[count == 0] = np.nan
    std[count == 0] = np.nan
    running_min[running_min == np.inf] = np.nan
    running_max[running_max == -np.inf] = np.nan

    return {
        "n_pixels": int(count.max()),   # representative pixel count
        "mean": mean,
        "std": std,
        "min": running_min,
        "max": running_max,
    }


def save_library_csv(
    path: str,
    class_name: str,
    wavelengths: np.ndarray | None,
    stats: dict,
    nbands: int,
):
    """Write statistics CSV for one land cover extraction."""
    x = wavelengths if wavelengths is not None else np.arange(nbands)
    wl_header = "wavelength_nm" if wavelengths is not None else "band"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            wl_header, "mean", "std", "min", "max",
            "n_pixels", "class_name"
        ])
        for i, xi in enumerate(x):
            def fmt(v):
                return "" if np.isnan(v) else float(v)
            writer.writerow([
                float(xi),
                fmt(stats["mean"][i]),
                fmt(stats["std"][i]),
                fmt(stats["min"][i]),
                fmt(stats["max"][i]),
                stats["n_pixels"],
                class_name,
            ])


# ── Application ───────────────────────────────────────────────────────────────

class HyperspectralViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hyperspectral BSQ Viewer")
        self.root.geometry("1300x780")

        # state
        self.img = None
        self.decimate: int = 10
        self.wavelengths: np.ndarray | None = None
        self.ignore_value: float | None = None
        self.rgb_display: np.ndarray | None = None
        self.spectrum: np.ndarray | None = None
        self.pixel_pos: tuple[int, int] | None = None

        # ── NEW: library state ────────────────────────────────────────────────
        self.rect_selector = None                        # RectangleSelector widget
        self.roi_rect_fullres: tuple[int,int,int,int] | None = None  # (r0,r1,c0,c1)
        self.library_mode: bool = False                  # toggle draw vs click
        self._rect_patch = None                          # drawn rectangle artist

        self._build_ui()
        self._auto_load()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── top toolbar (existing) ────────────────────────────────────────────
        bar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(bar, text="Open file…", command=self._open_file).pack(
            side=tk.LEFT, padx=4, pady=3
        )
        tk.Button(bar, text="Export spectrum to CSV…", command=self._export_csv).pack(
            side=tk.LEFT, padx=4, pady=3
        )
        self.status_var = tk.StringVar(value="No file loaded.")
        tk.Label(bar, textvariable=self.status_var, anchor=tk.W, fg="#444").pack(
            side=tk.LEFT, padx=12
        )

        # ── NEW: library toolbar ──────────────────────────────────────────────
        lib_bar = tk.Frame(self.root, bd=1, relief=tk.RAISED, bg="#e8f0e8")
        lib_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(lib_bar, text="Spectral Library:", bg="#e8f0e8", font=("TkDefaultFont", 9, "bold")).pack(
            side=tk.LEFT, padx=6, pady=3
        )

        # class selector
        tk.Label(lib_bar, text="Class:", bg="#e8f0e8").pack(side=tk.LEFT, padx=(8, 2), pady=3)
        self.class_var = tk.StringVar(value=LAND_COVER_CLASSES[0])
        class_menu = tk.OptionMenu(lib_bar, self.class_var, *LAND_COVER_CLASSES)
        class_menu.config(width=12)
        class_menu.pack(side=tk.LEFT, padx=2, pady=3)

        # draw mode toggle
        self.draw_btn = tk.Button(
            lib_bar, text="✏ Draw Rectangle",
            command=self._toggle_draw_mode,
            bg="#b0d0b0", relief=tk.RAISED
        )
        self.draw_btn.pack(side=tk.LEFT, padx=6, pady=3)

        # extract button
        tk.Button(
            lib_bar, text="⬇ Extract & Save to Library…",
            command=self._extract_and_save,
            bg="#6a9e6a", fg="white"
        ).pack(side=tk.LEFT, padx=6, pady=3)

        # roi info label
        self.roi_var = tk.StringVar(value="No region selected.")
        tk.Label(lib_bar, textvariable=self.roi_var, bg="#e8f0e8", fg="#333").pack(
            side=tk.LEFT, padx=12
        )

        # matplotlib figure
        self.fig = Figure(figsize=(14, 6.5))
        self.ax_rgb = self.fig.add_subplot(1, 2, 1)
        self.ax_spec = self.fig.add_subplot(1, 2, 2)
        self.fig.tight_layout(pad=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        NavigationToolbar2Tk(self.canvas, self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # left-click to inspect a pixel (existing)
        self._click_cid = self.canvas.mpl_connect("button_press_event", self._on_click)

        # ── NEW: RectangleSelector (inactive until draw mode on) ──────────────
        self.rect_selector = RectangleSelector(
            self.ax_rgb,
            self._on_rect_selected,
            useblit=True,
            button=[1],
            minspanx=2, minspany=2,
            spancoords="pixels",
            interactive=True,
            props=dict(facecolor="#00aa00", edgecolor="#006600",
                       alpha=0.25, fill=True, linewidth=2),
        )
        self.rect_selector.set_active(False)   # off by default

    # ── NEW: draw mode toggle ─────────────────────────────────────────────────

    def _toggle_draw_mode(self):
        self.library_mode = not self.library_mode
        if self.library_mode:
            self.rect_selector.set_active(True)
            self.canvas.mpl_disconnect(self._click_cid)   # disable pixel click
            self.draw_btn.config(text="✖ Cancel Drawing", bg="#d08080")
            self.status_var.set("Draw mode ON — drag a rectangle on the image.")
        else:
            self.rect_selector.set_active(False)
            self._click_cid = self.canvas.mpl_connect(
                "button_press_event", self._on_click
            )
            self.draw_btn.config(text="✏ Draw Rectangle", bg="#b0d0b0")
            self.status_var.set("Draw mode OFF — click a pixel to inspect.")

    # ── NEW: rectangle selection callback ────────────────────────────────────

    def _on_rect_selected(self, eclick, erelease):
        """Called by RectangleSelector when user finishes drawing."""
        if self.img is None:
            return

        # Coords are in decimated display space
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])

        # Map to full-resolution pixel coordinates
        c0 = max(0, int(round(x0)) * self.decimate)
        c1 = min(self.img.ncols - 1, int(round(x1)) * self.decimate)
        r0 = max(0, int(round(y0)) * self.decimate)
        r1 = min(self.img.nrows - 1, int(round(y1)) * self.decimate)

        if c1 <= c0 or r1 <= r0:
            self.roi_var.set("Rectangle too small — try again.")
            return

        self.roi_rect_fullres = (r0, r1, c0, c1)
        n_pixels = (r1 - r0 + 1) * (c1 - c0 + 1)
        self.roi_var.set(
            f"ROI: rows {r0}–{r1}, cols {c0}–{c1}  "
            f"({r1-r0+1} × {c1-c0+1} = {n_pixels:,} px)"
        )
        self.status_var.set(
            f"Rectangle selected ({n_pixels:,} pixels). "
            "Choose class and click 'Extract & Save to Library…'."
        )

    # ── NEW: extract statistics and save ─────────────────────────────────────

    def _extract_and_save(self):
        if self.img is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        if self.roi_rect_fullres is None:
            messagebox.showinfo("No region", "Draw a rectangle first.")
            return

        r0, r1, c0, c1 = self.roi_rect_fullres
        class_name = self.class_var.get()
        n_rows = r1 - r0 + 1
        n_cols = c1 - c0 + 1

        # ask where to save
        default_name = f"{class_name.replace(' ', '_')}_r{r0}-{r1}_c{c0}-{c1}.csv"
        save_path = filedialog.asksaveasfilename(
            title="Save library entry as CSV",
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not save_path:
            return

        # progress callback updates status bar
        def on_progress(done, total):
            self.status_var.set(
                f"Extracting '{class_name}' — row {done}/{total} …"
            )
            self.root.update_idletasks()

        self.status_var.set(f"Starting extraction of {n_rows}×{n_cols} region…")
        self.root.update_idletasks()

        try:
            stats = extract_roi_statistics(
                self.img, r0, r1, c0, c1,
                self.ignore_value,
                progress_callback=on_progress,
            )
            save_library_csv(
                save_path, class_name,
                self.wavelengths, stats, self.img.nbands
            )
            self.status_var.set(
                f"Saved '{class_name}' library entry → {save_path}  "
                f"({stats['n_pixels']:,} valid pixels)"
            )
            messagebox.showinfo(
                "Saved",
                f"Library entry saved:\n{save_path}\n\n"
                f"Class: {class_name}\n"
                f"Valid pixels: {stats['n_pixels']:,}\n"
                f"Bands: {self.img.nbands}"
            )
        except Exception as exc:
            messagebox.showerror("Extraction failed", str(exc))
            self.status_var.set("Extraction failed.")

    # ── File loading ──────────────────────────────────────────────────────────

    def _auto_load(self):
        if len(sys.argv) > 1:
            self._load(Path(sys.argv[1]))
            return

        if not DATA_DIR.exists():
            self.status_var.set(f"Data directory not found: {DATA_DIR}")
            return

        hdrs = find_hdr_files(DATA_DIR)
        if not hdrs:
            self.status_var.set(f"No .hdr files found in {DATA_DIR}")
            return
        if len(hdrs) == 1:
            self._load(hdrs[0])
        else:
            self._pick_file(hdrs)

    def _pick_file(self, hdrs: list[Path]):
        dlg = tk.Toplevel(self.root)
        dlg.title("Select dataset")
        dlg.grab_set()
        tk.Label(dlg, text="Multiple datasets found — select one to open:").pack(
            padx=12, pady=8
        )
        lb = tk.Listbox(dlg, width=72, height=min(len(hdrs), 10))
        lb.pack(padx=12, pady=4)
        for h in hdrs:
            lb.insert(tk.END, h.name)
        lb.selection_set(0)

        def on_ok():
            idx = lb.curselection()
            dlg.destroy()
            self._load(hdrs[idx[0]])

        tk.Button(dlg, text="Open", command=on_ok, width=12).pack(pady=8)
        dlg.wait_window()

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open ENVI header (.hdr)",
            initialdir=DATA_DIR if DATA_DIR.exists() else Path.home(),
            filetypes=[("ENVI header", "*.hdr"), ("All files", "*.*")],
        )
        if path:
            self._load(Path(path))

    def _load(self, hdr_path: Path):
        self.status_var.set(f"Loading RGB bands from  {hdr_path.name} …")
        self.root.update_idletasks()
        try:
            self.img = load_image(hdr_path)
            meta = self.img.metadata
            self.wavelengths = parse_wavelengths(meta)
            self.ignore_value = get_ignore_value(meta)
            r, g, b = get_rgb_bands(meta)
            self.rgb_display = read_rgb(self.img, r, g, b, self.ignore_value, self.decimate)
            self.spectrum = None
            self.pixel_pos = None
            self.roi_rect_fullres = None                 # ── NEW: reset ROI
            self.roi_var.set("No region selected.")     # ── NEW
            self._refresh_plots(update_rgb=True)
            self.status_var.set(
                f"{hdr_path.name}  |  "
                f"{self.img.nrows} lines × {self.img.ncols} samples × "
                f"{self.img.nbands} bands  |  "
                "Click a pixel to inspect its spectrum."
            )
        except Exception as exc:
            messagebox.showerror("Error loading file", str(exc))
            self.status_var.set("Load failed.")

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _refresh_plots(self, update_rgb: bool = True):
        if update_rgb:
            self.ax_rgb.clear()
            if self.rgb_display is not None:
                self.ax_rgb.imshow(
                    self.rgb_display, interpolation="bilinear", aspect="auto"
                )
            self.ax_rgb.set_title("RGB preview — click a pixel to inspect")
            self.ax_rgb.axis("off")

        # redraw pixel marker
        for artist in self.ax_rgb.lines:
            artist.remove()
        if self.pixel_pos:
            row, col = self.pixel_pos
            self.ax_rgb.plot(
                col / self.decimate, row / self.decimate,
                "r+", markersize=14, markeredgewidth=2.5
            )

        # ── NEW: redraw ROI rectangle if one is selected ──────────────────────
        if self._rect_patch is not None:
            try:
                self._rect_patch.remove()
            except Exception:
                pass
            self._rect_patch = None

        if self.roi_rect_fullres is not None:
            r0, r1, c0, c1 = self.roi_rect_fullres
            from matplotlib.patches import Rectangle
            self._rect_patch = Rectangle(
                (c0 / self.decimate, r0 / self.decimate),
                (c1 - c0) / self.decimate,
                (r1 - r0) / self.decimate,
                linewidth=2, edgecolor="#006600",
                facecolor="#00aa00", alpha=0.2,
            )
            self.ax_rgb.add_patch(self._rect_patch)

        # spectral panel
        self.ax_spec.clear()
        if self.spectrum is not None:
            row, col = self.pixel_pos
            x = (
                self.wavelengths
                if self.wavelengths is not None
                else np.arange(len(self.spectrum))
            )
            xlabel = "Wavelength (nm)" if self.wavelengths is not None else "Band index"
            self.ax_spec.plot(x, self.spectrum, linewidth=1.2, color="steelblue")
            self.ax_spec.set_title(f"Spectral signature — row {row},  col {col}")
            self.ax_spec.set_xlabel(xlabel)
            self.ax_spec.set_ylabel("Reflectance (× 10⁻⁴)")
            self.ax_spec.grid(True, alpha=0.3)
        else:
            self.ax_spec.set_title("Spectral signature")
            self.ax_spec.text(
                0.5, 0.5,
                "Click a pixel in the RGB image",
                ha="center", va="center",
                transform=self.ax_spec.transAxes,
                color="gray", fontsize=12,
            )

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    # ── Mouse click handler ───────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes is not self.ax_rgb or self.img is None:
            return
        col = int(round(event.xdata)) * self.decimate
        row = int(round(event.ydata)) * self.decimate
        row = min(row, self.img.nrows - 1)
        col = min(col, self.img.ncols - 1)

        self.pixel_pos = (row, col)
        self.spectrum = read_spectrum(self.img, row, col, self.ignore_value)
        self._refresh_plots(update_rgb=False)
        self.status_var.set(
            f"Pixel ({row}, {col})  |  "
            "Use 'Export spectrum to CSV…' to save."
        )

    # ── CSV export (single pixel, unchanged) ─────────────────────────────────

    def _export_csv(self):
        if self.spectrum is None:
            messagebox.showinfo("Nothing to export", "Click on a pixel first.")
            return

        row, col = self.pixel_pos
        default_name = f"spectrum_r{row}_c{col}.csv"
        path = filedialog.asksaveasfilename(
            title="Save spectrum as CSV",
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        x = (
            self.wavelengths
            if self.wavelengths is not None
            else np.arange(len(self.spectrum))
        )
        col_header = "wavelength_nm" if self.wavelengths is not None else "band"

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([col_header, "value"])
            for xi, vi in zip(x, self.spectrum):
                writer.writerow([float(xi), "" if np.isnan(vi) else float(vi)])

        self.status_var.set(f"Saved → {path}")
        messagebox.showinfo("Saved", f"Spectrum exported to:\n{path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    HyperspectralViewer(root)
    root.mainloop()