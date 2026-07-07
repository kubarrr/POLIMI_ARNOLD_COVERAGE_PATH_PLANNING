#!/usr/bin/env python3
"""
run_pipeline_with_saved_figures.py

Same run as run_pipeline.py, but it also renders the report figures to PNG files
(no notebook / Jupyter needed). It first runs run_pipeline.py to produce the raw
outputs in cache/<name>/ and compare/<name>/, then loads them and saves the plots
to figures/<name>/.

Examples
--------
    python run_pipeline_with_saved_figures.py --name data  --config paths.txt      --method all --k 3
    python run_pipeline_with_saved_figures.py --name data2 --config paths_data2.txt --method all --k 3

Saved figures (figures/<name>/):
    zones.png, old_vs_final.png, distributions.png, dem_slope.png (if a DEM),
    stages.png, mission_water.png, mission_nitrogen.png, mission_p80.png,
    method_zone_maps.png
"""
import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy.ndimage import binary_dilation

from mission_planner import (field_mask, target_mask, _sweep_segments,
                             order_swaths_snake, slope_degrees)

HERE = os.path.dirname(os.path.abspath(__file__))
ZONE_COLORS = ["#2b2b2b", "#d73027", "#fee08b", "#66bd63", "#1a9850"]
VIGOR = {1: "Low vigor", 2: "Medium vigor", 3: "High vigor", 4: "Very high vigor"}


def zlegend(K, path=None, pc=None):
    h = [Line2D([0], [0], color=pc, lw=2, label=path)] if path else []
    return h + [Line2D([0], [0], marker="s", ls="", ms=11, mec="k",
                       markerfacecolor=ZONE_COLORS[z], label=f"Zone {z} — {VIGOR.get(z,'')}")
                for z in range(1, K + 1)]


def load_ds(name):
    c = os.path.join(HERE, "cache", name)
    ds = {"name": name,
          "zones": np.load(f"{c}/preview_map.npy"),
          "ndre": np.load(f"{c}/ndre_preview.npy"),
          "ndvi": np.load(f"{c}/ndvi_preview.npy"),
          "base": np.load(f"{c}/false_colour.npy"),
          "meta": np.load(f"{c}/meta.npz", allow_pickle=True)}
    ds["dem"] = np.load(f"{c}/dem_preview.npy") if os.path.exists(f"{c}/dem_preview.npy") else None
    ds["raw"] = np.load(f"{c}/preview_map_raw.npy") if os.path.exists(f"{c}/preview_map_raw.npy") else None
    ds["px"] = float(ds["meta"]["pixel_size_m"])
    ds["K"] = int(ds["meta"]["k"])
    ds["method"] = str(ds["meta"]["method"])
    ds["mission"] = {s: np.load(f"{c}/mission_{s}.npy")
                     for s in ("water", "nitrogen", "p80")
                     if os.path.exists(f"{c}/mission_{s}.npy")}
    ds["p80_mask"] = np.load(f"{c}/p80_mask.npy") if os.path.exists(f"{c}/p80_mask.npy") else None
    return ds


def _zone_cmap(K):
    return (ListedColormap([ZONE_COLORS[z] for z in range(1, K + 1)]),
            BoundaryNorm(np.arange(1, K + 2) - 0.5, K))


def save(fig, outdir, fname):
    fig.savefig(os.path.join(outdir, fname), dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_zones(ds, outdir):
    K = ds["K"]; cmap, norm = _zone_cmap(K)
    zov = np.ma.masked_where(ds["zones"] == 0, ds["zones"])
    fig, ax = plt.subplots(1, 2, figsize=(20, 8.5), constrained_layout=True)
    ax[0].imshow(ds["base"]); ax[0].set_title(f"False-colour orthophoto (NIR-R-G) — {ds['name']}")
    ax[1].set_facecolor("#0d0d0d")
    ax[1].imshow(zov, cmap=cmap, norm=norm, interpolation="nearest")
    ax[1].set_title(f"Management zones — {ds['method']} (K={K})")
    ax[1].legend(handles=zlegend(K), loc="upper right", framealpha=0.95, fontsize=10)
    for a in ax: a.axis("off")
    save(fig, outdir, "zones.png")


def fig_old_vs_final(ds, outdir):
    if ds["raw"] is None:
        return
    K = ds["K"]; cmap = ListedColormap(ZONE_COLORS[:K + 1]); norm = BoundaryNorm(np.arange(K + 2) - 0.5, K + 1)
    fig, ax = plt.subplots(1, 2, figsize=(18, 7.5), constrained_layout=True)
    ax[0].imshow(ds["raw"], cmap=cmap, norm=norm, interpolation="nearest")
    ax[0].set_title("OLD: per-pixel clustering (speckle)")
    ax[1].imshow(ds["zones"], cmap=cmap, norm=norm, interpolation="nearest")
    ax[1].set_title("FINAL: smoothed contiguous management zones")
    for a in ax: a.legend(handles=zlegend(K), loc="upper right", framealpha=0.9, fontsize=8); a.axis("off")
    save(fig, outdir, "old_vs_final.png")


def fig_distributions(ds, outdir):
    K, z, ndre, ndvi = ds["K"], ds["zones"], ds["ndre"], ds["ndvi"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for j, (arr, nm) in enumerate([(ndre, "NDRE"), (ndvi, "NDVI")]):
        bp = ax[j].boxplot([arr[z == zz] for zz in range(1, K + 1)], patch_artist=True,
                           showfliers=False, medianprops=dict(color="black"))
        for patch, zz in zip(bp["boxes"], range(1, K + 1)):
            patch.set_facecolor(ZONE_COLORS[zz])
        ax[j].set_xticklabels([f"Zone {zz}\n{VIGOR.get(zz,'')}" for zz in range(1, K + 1)])
        ax[j].set_ylabel(nm); ax[j].grid(axis="y", alpha=0.3)
        ax[j].set_title(f"{nm} distribution per zone — {ds['name']}")
    save(fig, outdir, "distributions.png")


def fig_dem_slope(ds, outdir):
    if ds["dem"] is None:
        return
    dem = ds["dem"]; slope = slope_degrees(dem, ds["px"]); veg = ds["zones"] > 0
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    im = ax[0].imshow(np.ma.masked_where(dem <= 0, dem), cmap="terrain")
    ax[0].set_title("DEM — elevation [m]"); ax[0].axis("off")
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04).set_label("Elevation [m]")
    im2 = ax[1].imshow(np.ma.masked_where(~veg, slope), cmap="magma")
    ax[1].set_title("Terrain slope [deg]"); ax[1].axis("off")
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04).set_label("Slope [deg]")
    save(fig, outdir, "dem_slope.png")


def fig_stages(ds, outdir):
    K, base, z = ds["K"], ds["base"], ds["zones"]
    fm = field_mask(z); tgt = target_mask(z, "water"); walk = fm & (z == 0)
    useful = walk & binary_dilation(tgt, iterations=8)
    sw = order_swaths_snake(_sweep_segments(useful, 4.0))
    cmap, norm = _zone_cmap(K)
    fig, ax = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    ax[0, 0].set_facecolor("#0d0d0d")
    ax[0, 0].imshow(np.ma.masked_where(z == 0, z), cmap=cmap, norm=norm, interpolation="nearest")
    ax[0, 0].set_title("1. Management zones (clustering output)")
    ax[0, 0].legend(handles=zlegend(K), loc="upper right", framealpha=0.95, fontsize=9)
    ax[0, 1].imshow(tgt, cmap="Greens", interpolation="nearest")
    ax[0, 1].set_title("2. Target mask — vine zones needing WATER (green = target)")
    ax[1, 0].imshow(useful, cmap="Oranges", interpolation="nearest")
    ax[1, 0].set_title("3. Drivable bare-soil corridors near targets (orange = drivable)")
    ax[1, 1].imshow(base * 0.6)
    for (x1, y1), (x2, y2) in sw:
        ax[1, 1].plot([x1, x2], [y1, y2], color="#00e5ff", lw=1.0, alpha=0.95)
    ax[1, 1].legend(handles=[Line2D([0], [0], color="#00e5ff", lw=2, label="Coverage swath")],
                    loc="upper right", framealpha=0.95, fontsize=10)
    ax[1, 1].set_title("4. Coverage swaths on the corridors (background = NIR-R-G false colour)")
    for a in ax.ravel(): a.axis("off")
    save(fig, outdir, "stages.png")


def fig_mission(ds, supp, color, outdir):
    if supp not in ds["mission"]:
        return
    traj = ds["mission"][supp]
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.imshow(ds["base"] * 0.6)
    for (x1, y1), (x2, y2) in traj:
        ax.plot([x1, x2], [y1, y2], color=color, lw=1.0, alpha=0.95, zorder=5)
    length = sum(np.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in traj) * ds["px"]
    ax.legend(handles=[Line2D([0], [0], color=color, lw=2, label=f"Robot path ({supp})")],
              loc="upper right", framealpha=0.95, fontsize=10)
    ax.set_title(f"Coverage path — {supp.upper()} — {ds['name']} "
                 f"({len(traj)} segments, ~{length:.0f} m)")
    ax.axis("off")
    save(fig, outdir, f"mission_{supp}.png")


def fig_method_grid(ds, outdir):
    cdir = os.path.join(HERE, "compare", ds["name"])
    import glob
    files = sorted(glob.glob(f"{cdir}/zones_*.npy"))
    if not files:
        return
    K = ds["K"]; cmap, norm = _zone_cmap(K)
    cols = 3; rows = int(np.ceil(len(files) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(ax).ravel()
    for a, f in zip(axes, files):
        m = np.load(f); a.set_facecolor("#0d0d0d")
        a.imshow(np.ma.masked_where(m == 0, m), cmap=cmap, norm=norm, interpolation="nearest")
        a.set_title(os.path.basename(f)[6:-4].replace("_", " ")); a.axis("off")
    for a in axes[len(files):]: a.axis("off")
    fig.legend(handles=zlegend(K), loc="lower center", ncol=K, framealpha=0.95, fontsize=10)
    fig.suptitle(f"Management zones by clustering method — {ds['name']}", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, outdir, "method_zone_maps.png")


def main():
    ap = argparse.ArgumentParser(description="Run the pipeline and save report figures")
    ap.add_argument("--name", default="data")
    ap.add_argument("--config", default="paths.txt")
    ap.add_argument("--method", default="all")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--high-quality", "--high-resolution", action="store_true", dest="high_quality")
    ap.add_argument("--use-elevation", action="store_true")
    args = ap.parse_args()

    # 1. run the pipeline (produces cache/<name> and compare/<name>)
    cmd = [sys.executable, os.path.join(HERE, "run_pipeline.py"),
           "--name", args.name, "--config", args.config,
           "--method", args.method, "--k", str(args.k)]
    if args.high_quality:
        cmd.append("--high-resolution")
    if args.use_elevation:
        cmd.append("--use-elevation")
    print("Running:", " ".join(cmd))
    if subprocess.call(cmd) != 0:
        raise SystemExit("run_pipeline.py failed")

    # 2. load the outputs and save figures
    outdir = os.path.join(HERE, "figures", args.name)
    os.makedirs(outdir, exist_ok=True)
    ds = load_ds(args.name)
    print(f"Rendering figures to {outdir} ...")
    fig_zones(ds, outdir)
    fig_old_vs_final(ds, outdir)
    fig_distributions(ds, outdir)
    fig_dem_slope(ds, outdir)
    fig_stages(ds, outdir)
    fig_mission(ds, "water", "#00e5ff", outdir)
    fig_mission(ds, "nitrogen", "#ff5ecb", outdir)
    fig_mission(ds, "p80", "#ffd000", outdir)
    fig_method_grid(ds, outdir)
    print(f"Done. Figures saved in {outdir}")


if __name__ == "__main__":
    main()
