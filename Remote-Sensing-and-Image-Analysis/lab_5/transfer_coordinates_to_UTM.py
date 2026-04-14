from pyproj import Transformer

t = Transformer.from_crs("EPSG:2177", "EPSG:4326", always_xy=True)
lon, lat = t.transform(6512464.0, 5580660.0)
print(f"Lat: {lat:.6f}, Lon: {lon:.6f}")