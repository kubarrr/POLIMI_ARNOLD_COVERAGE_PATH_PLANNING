#!/usr/bin/env python3
"""
run_pipeline.py — end-to-end project flow (clustering -> zones -> coverage paths).

This is the reproducible entry point that runs the whole project from beginning
to end and writes raw results to disk. The report notebook only calls this with
flags and then visualises / aggregates the outputs (all reporting code lives in
the notebook).

The colleague's original `pipeline.py` is left untouched; this runner uses the
same building blocks but bakes in the quality fixes found during evaluation:
  * train on the full decimated vegetation distribution (a sparse spatial sample
    collapses K=3 into an empty high-vigor zone);
  * cluster a spatially smoothed vigor surface -> contiguous management blocks,
    not per-pixel speckle, while soil (zone 0) stays drivable;
  * geodesic pixel size so slope works for geographic (degree) CRSs too.

Examples
--------
    python run_pipeline.py --name data  --config paths.txt      --method kmeans --k 3
    python run_pipeline.py --name data  --config paths.txt      --method all    --k 3
    python run_pipeline.py --name data2 --config paths_data2.txt --method kmeans --k 3

Raw outputs (per dataset --name), consumed by the notebook
----------------------------------------------------------
    cache/<name>/   preview_map, dem_preview?, ndre_preview, ndvi_preview,
                    false_colour, features_X, veg, meta,
                    mission_water, mission_nitrogen
    compare/<name>/ zones_<Method>, labels_<Method>
"""
import argparse
import os
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling as WarpResampling
from pyproj import Geod
from scipy.ndimage import gaussian_filter
from skimage.transform import resize as sk_resize
from sklearn.cluster import (KMeans, MiniBatchKMeans, AgglomerativeClustering,
                             SpectralClustering)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min

from utils import load_paths_from_txt, get_smallest_dimensions
from processing import sort_labels_by_ndre
from mission_planner import generate_autonomous_mission

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_NAMES = {"kmeans": "K-Means", "mbkmeans": "Mini-Batch K-Means", "gmm": "GMM",
                "agglomerative": "Agglomerative", "spectral": "Spectral",
                "fcm": "Fuzzy C-Means"}
ALL_METHODS = ["kmeans", "mbkmeans", "gmm", "agglomerative", "spectral", "fcm"]


# ---------------------------------------------------------------- geometry ----
def ground_pixel_size_m(path, h, w):
    with rasterio.open(path) as s:
        t, crs = s.transform, s.crs
        if crs and crs.is_geographic:
            lon0, lat0 = t * (w // 2, h // 2)
            lon1, lat1 = t * (w // 2 + 1, h // 2)
            _, _, d = Geod(ellps="WGS84").inv(lon0, lat0, lon1, lat1)
            return d
        return abs(t.a)


def read_decimated(path, oh, ow):
    with rasterio.open(path) as s:
        a = s.read(1, out_shape=(oh, ow), resampling=Resampling.average).astype(np.float32)
    return np.reshape(a, (oh, ow))


# ------------------------------------------------------------- clustering ----
def load_features(paths, step, zone_smooth_m):
    h, w = get_smallest_dimensions(paths)
    oh, ow = h // step, w // step
    pixel_size_m = ground_pixel_size_m(paths[0], h, w) * step

    nir, re, r, g, ndvi = (read_decimated(p, oh, ow) for p in paths)
    ndre = (nir - re) / (nir + re + 1e-6)
    veg = ndre > 0.1

    sigma = max(1.0, zone_smooth_m / max(pixel_size_m, 1e-6))
    wsum = gaussian_filter(veg.astype(np.float32), sigma)
    sm = lambda f: gaussian_filter(np.where(veg, f, 0.0).astype(np.float32), sigma) / np.maximum(wsum, 1e-6)

    feats = np.column_stack([sm(x)[veg] for x in (ndre, ndvi, nir, r, nir)])
    X = MinMaxScaler().fit_transform(feats)
    # unsmoothed per-pixel features -> the "raw" (pre-fix) zoning, kept only so
    # the report can show the old speckled result next to the final contiguous one
    feats_raw = np.column_stack([x[veg] for x in (ndre, ndvi, nir, r, nir)])
    X_raw = MinMaxScaler().fit_transform(feats_raw)
    return dict(oh=oh, ow=ow, veg=veg, ndre=ndre, ndvi=ndvi, nir=nir, r=r, g=g,
                feats=feats, X=X, feats_raw=feats_raw, X_raw=X_raw,
                pixel_size_m=pixel_size_m)


def cluster_labels(method, X, k, seed=123):
    if method == "kmeans":
        return KMeans(k, random_state=seed, n_init=10).fit_predict(X)
    if method == "mbkmeans":
        return MiniBatchKMeans(k, random_state=seed, n_init=10, batch_size=4096).fit_predict(X)
    if method == "gmm":
        return GaussianMixture(k, covariance_type="full", random_state=seed).fit_predict(X)
    if method in ("agglomerative", "spectral"):
        rng = np.random.RandomState(42)
        sub = rng.choice(len(X), min(3000, len(X)), replace=False)
        if method == "agglomerative":
            m = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X[sub])
        else:
            m = SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                   n_neighbors=10, random_state=seed, n_jobs=-1).fit(X[sub])
        cent = np.array([X[sub][m.labels_ == c].mean(0) for c in np.unique(m.labels_)])
        return pairwise_distances_argmin_min(X, cent)[0]
    if method == "fcm":
        import skfuzzy as fuzz
        _, u, *_ = fuzz.cluster.cmeans(X.T, c=k, m=2.0, error=0.005, maxiter=100, seed=42)
        return np.argmax(u, axis=0)
    raise ValueError(method)


def build_zone_map(labels, feat, k):
    mapping = sort_labels_by_ndre(labels, feat["feats"])
    zmap = np.zeros((feat["oh"], feat["ow"]), dtype=np.uint8)
    zmap[feat["veg"]] = np.array([mapping[l] for l in labels], dtype=np.uint8)
    return zmap


def false_colour(feat):
    def n8(a):
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    return np.dstack([n8(feat["nir"]), n8(feat["r"]), n8(feat["g"])]).astype(np.float32)


def align_dem(path_dem, ref_path, oh, ow, shape, step):
    if not path_dem or str(path_dem).upper() == "NONE" or not os.path.exists(path_dem):
        return None
    with rasterio.open(ref_path) as ref:
        preview_transform = ref.transform * ref.transform.scale(step, step)
        crs = ref.crs
    with rasterio.open(path_dem) as dem_src:
        dem_aligned = np.zeros((oh, ow), dtype=np.float32)
        reproject(source=rasterio.band(dem_src, 1), destination=dem_aligned,
                  src_transform=dem_src.transform, src_crs=dem_src.crs,
                  dst_transform=preview_transform, dst_crs=crs,
                  resampling=WarpResampling.bilinear)
    return sk_resize(dem_aligned, shape, order=1, preserve_range=True).astype(np.float32)


def traj_to_array(traj):
    return np.array([[[x1, y1], [x2, y2]] for (x1, y1), (x2, y2) in traj], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="End-to-end vineyard zoning + coverage path planning")
    ap.add_argument("--method", default="kmeans", choices=ALL_METHODS + ["all"],
                    help="clustering method, or 'all' to run every method (default kmeans)")
    ap.add_argument("--k", type=int, default=3, help="number of zones (default 3)")
    ap.add_argument("--config", default="paths.txt", help="paths file")
    ap.add_argument("--name", default="data", help="dataset name -> cache/<name>/")
    ap.add_argument("--zone-smooth-m", type=float, default=2.5,
                    help="metre baseline for smoothing the vigor surface")
    ap.add_argument("--step", type=int, default=7, help="preview decimation (default 7)")
    ap.add_argument("--high-quality", "--high-resolution", action="store_true", dest="high_quality",
                    help="finer preview (step=3) for sharper zones (slower)")
    ap.add_argument("--use-elevation", action="store_true",
                    help="route mission transitions with slope-aware A* (needs a DEM; slower)")
    args = ap.parse_args()

    step = 3 if args.high_quality else args.step
    k = args.k
    cache = os.path.join(HERE, "cache", args.name)
    compare = os.path.join(HERE, "compare", args.name)
    os.makedirs(cache, exist_ok=True)
    os.makedirs(compare, exist_ok=True)

    print("=" * 60)
    print(f" RUN PIPELINE — dataset '{args.name}', method '{args.method}', K={k}")
    print("=" * 60)

    # --- 1. load rasters + smoothed vigor features ---
    print(f"[1/4] Loading rasters from {args.config} ...")
    paths, path_dem, _ = load_paths_from_txt(args.config)
    feat = load_features(paths, step, args.zone_smooth_m)
    print(f"      preview {feat['oh']}x{feat['ow']} | vegetation {int(feat['veg'].sum())} px "
          f"| pixel ~= {feat['pixel_size_m']:.3f} m")

    # --- 2. clustering (every requested method, with the chosen flags) ---
    print(f"[2/4] Clustering ...")
    methods = ALL_METHODS if args.method == "all" else [args.method]
    primary = "kmeans" if args.method == "all" else args.method
    zmaps = {}
    for m in methods:
        name = METHOD_NAMES[m]
        try:
            labels = cluster_labels(m, feat["X"], k)
        except Exception as e:
            print(f"      {name:20s} SKIPPED ({type(e).__name__}: {e})")
            continue
        zmap = build_zone_map(labels, feat, k)
        zmaps[m] = zmap
        np.save(os.path.join(compare, f"zones_{name.replace(' ', '_')}.npy"), zmap)
        np.save(os.path.join(compare, f"labels_{name.replace(' ', '_')}.npy"), labels.astype(np.int16))
        print(f"      {name:20s} zones {np.bincount(zmap[feat['veg']], minlength=k+1)[1:].tolist()}")
    if not zmaps:
        raise SystemExit("No clustering method succeeded.")
    if primary not in zmaps:
        primary = next(iter(zmaps))
    preview_map = zmaps[primary]

    # raw (unsmoothed, per-pixel) zoning for the primary method -> old-vs-final view
    try:
        raw_labels = cluster_labels(primary, feat["X_raw"], k)
        raw_mapping = sort_labels_by_ndre(raw_labels, feat["feats_raw"])
        preview_map_raw = np.zeros((feat["oh"], feat["ow"]), dtype=np.uint8)
        preview_map_raw[feat["veg"]] = np.array([raw_mapping[l] for l in raw_labels], dtype=np.uint8)
    except Exception:
        preview_map_raw = preview_map

    # --- 3. DEM + coverage path planning (water & nitrogen) ---
    print(f"[3/4] Path planning ...")
    dem_preview = align_dem(path_dem, paths[0], feat["oh"], feat["ow"], preview_map.shape, step)
    if args.use_elevation and dem_preview is None:
        print("      --use-elevation requested but no DEM -> simple transitions used")
    for supp in ("water", "nitrogen"):
        traj = generate_autonomous_mission(
            preview_map, supp, spacing_px=4.0,
            use_elevation=args.use_elevation and dem_preview is not None,
            dem=dem_preview, pixel_size_m=feat["pixel_size_m"])
        np.save(os.path.join(cache, f"mission_{supp}.npy"), traj_to_array(traj))
        print(f"      {supp:9s} mission: {len(traj)} segments")

    # --- 4. persist everything the notebook needs ---
    print(f"[4/4] Saving outputs ...")
    np.save(os.path.join(cache, "preview_map.npy"), preview_map)
    np.save(os.path.join(cache, "preview_map_raw.npy"), preview_map_raw)
    np.save(os.path.join(cache, "ndre_preview.npy"), feat["ndre"].astype(np.float32))
    np.save(os.path.join(cache, "ndvi_preview.npy"), feat["ndvi"].astype(np.float32))
    np.save(os.path.join(cache, "false_colour.npy"), false_colour(feat))
    np.save(os.path.join(cache, "features_X.npy"), feat["X"].astype(np.float32))
    np.save(os.path.join(cache, "veg.npy"), feat["veg"])
    if dem_preview is not None:
        np.save(os.path.join(cache, "dem_preview.npy"), dem_preview)
    else:
        print("      no DEM -> slope / A* elevation stage disabled")
    np.savez(os.path.join(cache, "meta.npz"),
             pixel_size_m=feat["pixel_size_m"], step=step, k=k,
             method=METHOD_NAMES[primary],
             methods=[METHOD_NAMES[m] for m in zmaps],
             zone_smooth_m=args.zone_smooth_m,
             has_dem=dem_preview is not None, name=args.name)

    print(f"\nDone. cache -> {cache}\n      compare -> {compare}")


if __name__ == "__main__":
    main()
