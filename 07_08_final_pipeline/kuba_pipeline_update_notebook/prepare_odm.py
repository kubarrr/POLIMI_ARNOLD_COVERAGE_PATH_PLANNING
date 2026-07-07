#!/usr/bin/env python3
"""
Adapter: turn an OpenDroneMap multispectral orthophoto into the 5 single-band
rasters (NIR, RedEdge, Red, Green, NDVI) that the clustering pipeline expects,
and write a matching paths file. No second pipeline is needed — this just feeds
the existing one a different dataset.

    python prepare_odm.py --ortho ../data2/odm_orthophoto/odm_orthophoto.tif \
                          --out ../data2/derived --paths paths_data2.txt

The DEM is optional: if the ODM run has no DSM/DEM raster, the paths file records
"NONE" for it and the pipeline simply skips the slope / A* elevation stage.
"""
import argparse
import os
import numpy as np
import rasterio


# ODM multispectral orthophoto band order for this project's camera
BAND = {"red": 1, "green": 2, "nir": 3, "rededge": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho", required=True, help="multi-band ODM orthophoto")
    ap.add_argument("--out", required=True, help="output folder for band rasters")
    ap.add_argument("--paths", required=True, help="paths file to write")
    ap.add_argument("--dem", default="NONE", help="DEM path or 'NONE' if absent")
    ap.add_argument("--shape", default="NONE", help="boundary shp/geojson or 'NONE'")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with rasterio.open(args.ortho) as src:
        prof = src.profile
        red = src.read(BAND["red"]).astype(np.float32)
        green = src.read(BAND["green"]).astype(np.float32)
        nir = src.read(BAND["nir"]).astype(np.float32)
        rededge = src.read(BAND["rededge"]).astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)

    prof.update(count=1, dtype="float32")
    layers = {"NIR": nir, "RE": rededge, "R": red, "G": green, "NDVI": ndvi}
    written = {}
    for key, arr in layers.items():
        path = os.path.join(args.out, f"odm_{key}.tif")
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(arr, 1)
        written[key] = path
        print(f"wrote {path}")

    # paths file order: NIR, RE, R, G, NDVI, DEM, SHAPE
    lines = [written["NIR"], written["RE"], written["R"], written["G"],
             written["NDVI"], args.dem, args.shape]
    with open(args.paths, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.paths}  (DEM={args.dem}, SHAPE={args.shape})")


if __name__ == "__main__":
    main()
