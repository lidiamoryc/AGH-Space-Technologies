# Lab 5 — Hyperspectral Data Analysis / Odra River

Airborne hyperspectral data analysis tools built for ENVI/BSQ data cubes
from the MGGP Aero survey of the Odra river (Capella SpotLight/Stripmap).

## Files

### Viewers & Tools
- **viewer.py** — main hyperspectral BSQ viewer (tkinter + matplotlib). Displays RGB preview of the data cube, click any pixel to inspect its full spectral signature, export to CSV. Implementation of task 1. 
- **viewer_with_area_types.py** — extended viewer with spectral library building tools. Draw rectangles over land cover areas, extract per-class spectral statistics (mean, std, min, max) using streaming Welford algorithm, export to library CSV. Implementation of task 2. 
- **plot_library.py** — separate script to visualise spectral library CSV files. Plots mean spectrum with ±1 std shaded envelope for one or multiple land cover classes.

### Water Quality
- **EO_Lab5_water_quality_analysis.ipynb** — calculates water quality indices (Chl-a, DOC, turbidity) from the hyperspectral cube and Sentinel-2 L2A bands. Produces side-by-side comparison maps. Implementation of task 3.

### Utilities
- **transfer_coordinates_to_UTM.py** — converts ENVI header geographic coordinates from Poland CS2000 Zone 6 (EPSG:2177) to WGS84 lat/lon for use in Copernicus Browser and other GIS tools.
- **sample_metadata.py** — prints geographic metadata (projection, pixel size, corner coordinates) from ENVI .hdr files.

### Spectral Library
- **spectral_signatures/** — directory containing per-class spectral library CSV files extracted using viewer_with_area_types.py
- **ex_water_spectrum_r1320_c270.csv** — example water spectral signature
- **ex_forest_spectrum_r390_c3280.csv** — example forest spectral signature
- **ex_field_spectrum_r950_c930.csv** — example field spectral signature

### Other
- **requirements.txt** — Python dependencies
- **TODO.pdf** — task specification document
- **README.md** — this file
