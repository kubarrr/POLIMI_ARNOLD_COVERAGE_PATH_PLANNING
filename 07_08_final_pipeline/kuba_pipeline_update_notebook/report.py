# %% [markdown]
# # UAV-informed Vineyard Coverage Path Planning — Results
#
# This notebook **runs the end-to-end pipeline** (`run_pipeline.py`) with flags and
# then visualises / aggregates its outputs. All reporting code (plots, statistics)
# lives here; `run_pipeline.py` only computes and saves raw results.
#
# Two datasets:
# * **`data`**  — cropped multispectral orthomosaic (NIR/RE/R/G/NDVI) + DEM.
# * **`data2`** — an OpenDroneMap multispectral orthophoto (no DEM), fed through the
#   `prepare_odm.py` adapter.
#
# **Reproduce / add a dataset:** `python run_pipeline.py --name <ds> --config <paths.txt> --method all --k 3`

# %%
import os, glob, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy.ndimage import binary_dilation, label as cc_label
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)
from IPython.display import display

from mission_planner import (field_mask, target_mask, _sweep_segments,
                             order_swaths_snake, generate_zone_swaths, slope_degrees,
                             generate_autonomous_mission)

plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 12, "font.size": 10})
CACHE, COMPARE, FIG = "cache", "compare", "figures"
os.makedirs(FIG, exist_ok=True)
ZONE_COLORS = ["#2b2b2b", "#d73027", "#fee08b", "#66bd63", "#1a9850"]
VIGOR = {1: "Low vigor", 2: "Medium vigor", 3: "High vigor", 4: "Very high vigor"}


def run_pipeline(name, config, method="all", k=3):
    """Invoke the end-to-end pipeline as a concrete full run with flags."""
    cmd = [sys.executable, "run_pipeline.py", "--name", name, "--config", config,
           "--method", method, "--k", str(k)]
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join(l for l in r.stdout.splitlines()
                     if not any(s in l for s in ("Deprecation", "new view", "s.read")))
    print(tail[-1500:])
    if r.returncode:
        print("STDERR:", r.stderr[-1500:])


# %% [markdown]
# ## Run the pipeline for both datasets
# The cell below **actually executes `run_pipeline.py` with flags** (it prints the
# exact command). The flags are:
# * `--name`   dataset name (where outputs are written)
# * `--config` the paths file (bands + DEM + shape)
# * `--method` clustering method — `all` runs every method for the comparison
# * `--k`      number of management zones
#
# ```
# python run_pipeline.py --name data  --config paths.txt      --method all --k 3
# python run_pipeline.py --name data2 --config paths_data2.txt --method all --k 3
# ```
# (add `--high-resolution` for sharper zones, or `--use-elevation` for slope-aware A*).

# %%
RUNS = [("data", "paths.txt"), ("data2", "paths_data2.txt")]
for _name, _cfg in RUNS:
    run_pipeline(_name, _cfg, method="all", k=3)


# %% [markdown]
# ## Reporting helpers (visualisation + statistics — notebook only)

# %%
def load_ds(name):
    c = os.path.join(CACHE, name)
    ds = {"name": name,
          "zones": np.load(f"{c}/preview_map.npy"),
          "ndre": np.load(f"{c}/ndre_preview.npy"),
          "ndvi": np.load(f"{c}/ndvi_preview.npy"),
          "base": np.load(f"{c}/false_colour.npy"),
          "X": np.load(f"{c}/features_X.npy"),
          "veg": np.load(f"{c}/veg.npy"),
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


def zlegend(K, path=None, pc=None):
    h = [Line2D([0], [0], color=pc, lw=2, label=path)] if path else []
    return h + [Line2D([0], [0], marker="s", ls="", ms=11, mec="k",
                       markerfacecolor=ZONE_COLORS[z], label=f"Zone {z} — {VIGOR.get(z,'')}")
                for z in range(1, K + 1)]


def zone_overlay(ax, ds, alpha):
    K = ds["K"]
    zov = np.ma.masked_where(ds["zones"] == 0, ds["zones"])
    cmap = ListedColormap([ZONE_COLORS[z] for z in range(1, K + 1)])
    ax.imshow(zov, cmap=cmap, norm=BoundaryNorm(np.arange(1, K + 2) - 0.5, K),
              alpha=alpha, interpolation="nearest")


def show_zones(ds):
    K = ds["K"]
    zov = np.ma.masked_where(ds["zones"] == 0, ds["zones"])
    cmap = ListedColormap([ZONE_COLORS[z] for z in range(1, K + 1)])
    norm = BoundaryNorm(np.arange(1, K + 2) - 0.5, K)
    fig, ax = plt.subplots(1, 2, figsize=(20, 8.5), constrained_layout=True)
    ax[0].imshow(ds["base"]); ax[0].set_title(f"False-colour orthophoto (NIR-R-G) — {ds['name']}", fontsize=13)
    # Render the zones with solid colours on a dark background so class
    # boundaries stay sharp (nearest-neighbour, no alpha blending).
    ax[1].set_facecolor("#0d0d0d")
    ax[1].imshow(zov, cmap=cmap, norm=norm, interpolation="nearest")
    ax[1].set_title(f"Management zones — {ds['method']} (K={K})", fontsize=13)
    ax[1].legend(handles=zlegend(K), loc="upper right", framealpha=0.95, fontsize=10)
    for a in ax: a.axis("off")
    fig.savefig(f"{FIG}/{ds['name']}_zones.png", dpi=130, bbox_inches="tight"); plt.show()


def zone_table(ds):
    K, z, ndre, ndvi, px = ds["K"], ds["zones"], ds["ndre"], ds["ndvi"], ds["px"]
    veg = z > 0
    rows = []
    for zz in range(1, K + 1):
        m = z == zz; n = int(m.sum())
        rows.append({"Zone": zz, "Vigor": VIGOR.get(zz, ""), "Canopy px": n,
                     "Canopy area [ha]": n * px * px / 1e4,
                     "% of vegetation": 100 * n / max(1, veg.sum()),
                     "NDRE mean": float(ndre[m].mean()), "NDRE std": float(ndre[m].std()),
                     "NDVI mean": float(ndvi[m].mean()), "NDVI std": float(ndvi[m].std())})
    df = pd.DataFrame(rows).set_index("Zone")
    display(df.style.format({"Canopy area [ha]": "{:.3f}", "% of vegetation": "{:.1f}",
                             "NDRE mean": "{:.3f}", "NDRE std": "{:.3f}",
                             "NDVI mean": "{:.3f}", "NDVI std": "{:.3f}", "Canopy px": "{:,}"})
            .set_caption(
        f"Per-zone statistics — {ds['name']}. "
        f"Area = canopy pixels x pixel_size^2 (pixel = {px:.3f} m). "
        f"Total canopy {veg.sum()*px*px/1e4:.2f} ha of a {z.size*px*px/1e4:.2f} ha frame "
        f"(only NDRE>0.1 canopy pixels are counted, not the inter-row soil). "
        f"'% of vegetation' = zone canopy px / all canopy px."))


def show_zone_distributions(ds):
    """Distribution of NDRE and NDVI within each management zone (mean+-std hidden
    in the boxes)."""
    K, z, ndre, ndvi = ds["K"], ds["zones"], ds["ndre"], ds["ndvi"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for j, (arr, nm) in enumerate([(ndre, "NDRE"), (ndvi, "NDVI")]):
        data = [arr[z == zz] for zz in range(1, K + 1)]
        bp = ax[j].boxplot(data, patch_artist=True, showfliers=False,
                           medianprops=dict(color="black"))
        for patch, zz in zip(bp["boxes"], range(1, K + 1)):
            patch.set_facecolor(ZONE_COLORS[zz])
        ax[j].set_xticklabels([f"Zone {zz}\n{VIGOR.get(zz,'')}" for zz in range(1, K + 1)])
        ax[j].set_ylabel(nm); ax[j].grid(axis="y", alpha=0.3)
        ax[j].set_title(f"{nm} distribution per zone — {ds['name']}")
    fig.savefig(f"{FIG}/{ds['name']}_distributions.png", bbox_inches="tight"); plt.show()


def show_mission(ds, supp, color):
    if supp not in ds["mission"]:
        return
    traj = ds["mission"][supp]
    fig, ax = plt.subplots(figsize=(13, 10))
    # Dimmed orthophoto as context; the path is drawn on top without a zone overlay.
    ax.imshow(ds["base"] * 0.6)
    for (x1, y1), (x2, y2) in traj:
        ax.plot([x1, x2], [y1, y2], color=color, lw=1.0, alpha=0.95, zorder=5)
    length = sum(np.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in traj) * ds["px"]
    ax.legend(handles=[Line2D([0], [0], color=color, lw=2, label=f"Robot path ({supp})")],
              loc="upper right", framealpha=0.95, fontsize=10)
    ax.set_title(f"Coverage path — {supp.upper()} — {ds['name']} "
                 f"({len(traj)} segments, ~{length:.0f} m)", fontsize=13)
    ax.axis("off")
    fig.savefig(f"{FIG}/{ds['name']}_{supp}.png", dpi=130, bbox_inches="tight"); plt.show()
    print(f"{supp.upper()}: {len(traj)} segments, driven length ~= {length:.0f} m")


def show_stages(ds):
    K, base, z = ds["K"], ds["base"], ds["zones"]
    fm = field_mask(z); tgt = target_mask(z, "water"); walk = fm & (z == 0)
    useful = walk & binary_dilation(tgt, iterations=8)
    sw = order_swaths_snake(_sweep_segments(useful, 4.0))
    zov = np.ma.masked_where(z == 0, z)
    cmap = ListedColormap([ZONE_COLORS[i] for i in range(1, K + 1)])
    norm = BoundaryNorm(np.arange(1, K + 2) - 0.5, K)
    fig, ax = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    ax[0, 0].set_facecolor("#0d0d0d")
    ax[0, 0].imshow(zov, cmap=cmap, norm=norm, interpolation="nearest")
    ax[0, 0].set_title("1. Management zones (clustering output)", fontsize=14)
    ax[0, 0].legend(handles=zlegend(K), loc="upper right", framealpha=0.95, fontsize=9)
    ax[0, 1].imshow(tgt, cmap="Greens", interpolation="nearest")
    ax[0, 1].set_title("2. Target mask — vine zones needing WATER (green = target)", fontsize=14)
    ax[1, 0].imshow(useful, cmap="Oranges", interpolation="nearest")
    ax[1, 0].set_title("3. Drivable bare-soil corridors near targets (orange = drivable)", fontsize=14)
    ax[1, 1].imshow(base * 0.6)
    for (x1, y1), (x2, y2) in sw:
        ax[1, 1].plot([x1, x2], [y1, y2], color="#00e5ff", lw=1.0, alpha=0.95)
    ax[1, 1].legend(handles=[Line2D([0], [0], color="#00e5ff", lw=2, label="Coverage swath")],
                    loc="upper right", framealpha=0.95, fontsize=10)
    ax[1, 1].set_title("4. Boustrophedon coverage swaths on the corridors "
                       "(background = NIR-R-G false colour)", fontsize=13)
    for a in ax.ravel(): a.axis("off")
    fig.savefig(f"{FIG}/{ds['name']}_stages.png", dpi=120, bbox_inches="tight"); plt.show()


def _fragmentation(zmap, K):
    return 1000.0 * sum(cc_label(zmap == z)[1] for z in range(1, K + 1)) / max(1, (zmap > 0).sum())


def _coverage(zmap, supp, px, reach=8):
    sw = order_swaths_snake(generate_zone_swaths(zmap, supp, 4.0))
    cov = np.zeros(zmap.shape, bool); length = 0.0
    for (x1, y1), (x2, y2) in sw:
        cov[int(y1), int(min(x1, x2)):int(max(x1, x2)) + 1] = True; length += abs(x2 - x1)
    tgt = target_mask(zmap, supp)
    served = binary_dilation(cov, iterations=reach) & tgt
    return round(length * px), len(sw), round(100 * served.sum() / max(1, tgt.sum()), 1)


def method_comparison(name):
    """Aggregate clustering-quality and path-covering metrics for every method."""
    X = np.load(f"{CACHE}/{name}/features_X.npy")
    K = int(np.load(f"{CACHE}/{name}/meta.npz", allow_pickle=True)["k"])
    px = float(np.load(f"{CACHE}/{name}/meta.npz", allow_pickle=True)["pixel_size_m"])
    rng = np.random.RandomState(42); si = rng.choice(len(X), min(5000, len(X)), replace=False)
    clu, pth, maps = [], [], {}
    for lf in sorted(glob.glob(f"{COMPARE}/{name}/labels_*.npy")):
        m = os.path.basename(lf)[7:-4].replace("_", " ")
        labels = np.load(lf); zmap = np.load(f"{COMPARE}/{name}/zones_{m.replace(' ', '_')}.npy")
        maps[m] = zmap
        counts = np.bincount(labels, minlength=K)
        clu.append({"Method": m,
                    "Silhouette": round(silhouette_score(X[si], labels[si]), 3),
                    "Davies-Bouldin": round(davies_bouldin_score(X, labels), 3),
                    "Calinski-Harabasz": int(calinski_harabasz_score(X, labels)),
                    "Balance": round(counts.min() / max(1, counts.max()), 3),
                    "Fragmentation": round(_fragmentation(zmap, K), 2)})
        for supp in ("water", "nitrogen"):
            length, nsw, cov = _coverage(zmap, supp, px)
            pth.append({"Method": m, "Supplement": supp, "Swaths": nsw,
                        "Length [m]": length, "Coverage [%]": cov})
    clu = pd.DataFrame(clu).set_index("Method"); pth = pd.DataFrame(pth).set_index(["Method", "Supplement"])
    display(clu.style.set_caption(
        f"Clustering quality by method — {name} "
        f"(Silhouette/Calinski-Harabasz: higher is better; "
        f"Davies-Bouldin/Fragmentation: lower is better)"))
    display(pth.style.set_caption(f"Path-covering metrics by method — {name}"))
    cmap = ListedColormap(ZONE_COLORS[:K + 1]); norm = BoundaryNorm(np.arange(K + 2) - 0.5, K + 1)
    names = list(maps); cols = 3; rows = int(np.ceil(len(names) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(ax).ravel()
    for a, mm in zip(axes, names):
        a.set_facecolor("#0d0d0d")
        a.imshow(np.ma.masked_where(maps[mm] == 0, maps[mm]), cmap=cmap, norm=norm,
                 interpolation="nearest")          # solid colours, sharp boundaries
        a.set_title(mm); a.axis("off")
    for a in axes[len(names):]: a.axis("off")
    fig.legend(handles=zlegend(K), loc="lower center", ncol=K, framealpha=0.95, fontsize=10)
    fig.suptitle(f"Zone maps by clustering method — {name}", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 1]); plt.show()


def show_old_vs_final(ds):
    """Old approach (per-pixel clustering, speckle) vs final (smoothed, contiguous)."""
    if ds["raw"] is None:
        return
    K = ds["K"]
    cmap = ListedColormap(ZONE_COLORS[:K + 1]); norm = BoundaryNorm(np.arange(K + 2) - 0.5, K + 1)
    fig, ax = plt.subplots(1, 2, figsize=(18, 7.5), constrained_layout=True)
    ax[0].imshow(ds["raw"], cmap=cmap, norm=norm, interpolation="nearest")
    ax[0].set_title("OLD: per-pixel clustering (speckle)")
    ax[1].imshow(ds["zones"], cmap=cmap, norm=norm, interpolation="nearest")
    ax[1].set_title("FINAL: smoothed contiguous management zones")
    for a in ax:
        a.legend(handles=zlegend(K), loc="upper right", framealpha=0.9, fontsize=8)
        a.axis("off")
    fig.savefig(f"{FIG}/{ds['name']}_old_vs_final.png", bbox_inches="tight"); plt.show()


def show_astar_mission(ds, supp="nitrogen", color="#00e5ff"):
    """Full-field slope-aware A* mission (same A* as the reference: it routes the
    transitions between swaths). Left = whole field; right = zoom on the routing."""
    if ds["dem"] is None:
        return
    z, base, px = ds["zones"], ds["base"], ds["px"]
    traj = generate_autonomous_mission(z, supp, 4.0, use_elevation=True,
                                       dem=ds["dem"], pixel_size_m=px)
    ys, xs = np.where(z > 0)
    cy, cx, W = int(ys.mean()), int(xs.mean()), 300
    fig, ax = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    ax[0].imshow(base * 0.4 + 0.05); zone_overlay(ax[0], ds, 0.3)
    for (x1, y1), (x2, y2) in traj:
        ax[0].plot([x1, x2], [y1, y2], color=color, lw=0.4, alpha=0.85)
    ax[0].set_title(f"Full-field slope-aware A* mission — {supp.upper()} — {ds['name']}")
    ax[1].imshow(base)
    for (x1, y1), (x2, y2) in traj:
        ax[1].plot([x1, x2], [y1, y2], color=color, lw=1.3, alpha=0.9)
    ax[1].set_xlim(max(0, cx - W), cx + W); ax[1].set_ylim(cy + W, max(0, cy - W))
    ax[1].set_title("Zoom — A* routes transitions through the soil corridors")
    for a in ax: a.axis("off")
    fig.savefig(f"{FIG}/{ds['name']}_{supp}_astar.png", bbox_inches="tight"); plt.show()
    length = sum(np.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in traj) * px
    print(f"A* {supp}: {len(traj)} segments, driven length ~= {length:.0f} m")


def show_missions_by_method(name, supp="nitrogen", color="#ff5ecb"):
    """Final coverage path for `supp` under every clustering method."""
    base = np.load(f"{CACHE}/{name}/false_colour.npy")
    px = float(np.load(f"{CACHE}/{name}/meta.npz", allow_pickle=True)["pixel_size_m"])
    files = sorted(glob.glob(f"{COMPARE}/{name}/zones_*.npy"))
    cols = 3; rows = int(np.ceil(len(files) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(ax).ravel()
    for a, f in zip(axes, files):
        method = os.path.basename(f)[6:-4].replace("_", " ")
        zmap = np.load(f)
        traj = generate_autonomous_mission(zmap, supp, 4.0, use_elevation=False)
        length = sum(np.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in traj) * px
        a.imshow(base * 0.4 + 0.05)
        for (x1, y1), (x2, y2) in traj:
            a.plot([x1, x2], [y1, y2], color=color, lw=0.7, alpha=0.9)
        a.set_title(f"{method}\n{len(traj)} seg, ~{length:.0f} m"); a.axis("off")
    for a in axes[len(files):]:
        a.axis("off")
    fig.suptitle(f"Final {supp.upper()} coverage path by clustering method — {name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}_missions_by_method.png", bbox_inches="tight"); plt.show()


def show_p80(ds, color="#ffd000"):
    """
    P80 demand coverage — the reference (Side-wise P80 CPP) selection rule applied
    on our raster. The demand region is the top-20% highest-vigor canopy (by NDRE);
    the same corridor coverage planner then plans a mission for it. Left: the P80
    demand region. Right: the resulting coverage path.
    """
    if "p80" not in ds["mission"] or ds["p80_mask"] is None:
        return
    traj, mask = ds["mission"]["p80"], ds["p80_mask"]
    fig, ax = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    ax[0].imshow(ds["base"] * 0.5)
    ax[0].imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap(["#ffd000"]),
                 alpha=0.75, interpolation="nearest")
    ax[0].set_title(f"P80 demand region — top 20% highest-vigor canopy — {ds['name']}")
    ax[1].imshow(ds["base"] * 0.6)
    for (x1, y1), (x2, y2) in traj:
        ax[1].plot([x1, x2], [y1, y2], color=color, lw=1.0, alpha=0.95)
    length = sum(np.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in traj) * ds["px"]
    ax[1].set_title(f"P80 coverage path ({len(traj)} segments, ~{length:.0f} m)")
    for a in ax:
        a.axis("off")
    fig.savefig(f"{FIG}/{ds['name']}_p80.png", dpi=120, bbox_inches="tight"); plt.show()
    print(f"P80: {len(traj)} segments, driven length ~= {length:.0f} m")


# %% [markdown]
# ---
# # PART A — Dataset `data` (multispectral + DEM)

# %% [markdown]
# ## A1. Management zones
# Zone 1 = lowest vigor (red) … Zone K = highest (green). Zone 0 (soil / inter-row)
# is drivable. Zones come from clustering a spatially smoothed vigor surface, so
# they are contiguous management blocks, not per-pixel speckle.

# %%
d1 = load_ds("data")
show_zones(d1)

# %% [markdown]
# ## A1b. Old vs final zones — the improvement
# **Left:** the original per-pixel clustering (speckle; with sparse training K=3
# could even collapse the high-vigor zone to empty). **Right:** the final smoothed,
# contiguous management zones the planner actually uses.
#
# *How the final zones are built:* for every band we spread the canopy values across
# the inter-row gaps with a NaN-aware Gaussian filter (~2.5 m baseline), giving a
# continuous vigor surface; K-Means is trained on that surface over the full
# vegetation, and the resulting labels are sorted by mean NDRE so Zone 1 = lowest
# vigor. Soil pixels stay 0. (Implemented in `run_pipeline.load_features` +
# `build_zone_map`.)

# %%
show_old_vs_final(d1)

# %% [markdown]
# ## A2. Per-zone statistics — mean NDRE rises with the zone number (correct ordering)

# %%
zone_table(d1)

# %% [markdown]
# ## A2b. Index distributions within each zone
# Not just the mean: the full **NDRE / NDVI distribution per zone**. Boxes should
# step up from Zone 1 to Zone K with limited overlap — that is what makes the zones
# meaningful management units rather than an arbitrary split.

# %%
show_zone_distributions(d1)

# %% [markdown]
# ## A3. Topography and slope (DEM smoothed over ~2 m before the gradient)

# %%
if d1["dem"] is not None:
    dem = d1["dem"]; slope = slope_degrees(dem, d1["px"]); veg = d1["zones"] > 0
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    im = ax[0].imshow(np.ma.masked_where(dem <= 0, dem), cmap="terrain")
    ax[0].set_title("DEM — elevation [m]"); ax[0].axis("off")
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04).set_label("Elevation [m]")
    im2 = ax[1].imshow(np.ma.masked_where(~veg, slope), cmap="magma")
    ax[1].set_title("Terrain slope [deg]"); ax[1].axis("off")
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04).set_label("Slope [deg]")
    fig.savefig(f"{FIG}/data_dem.png", bbox_inches="tight"); plt.show()
    print(f"Slope inside field: median={np.median(slope[veg]):.1f}deg, "
          f"90th pct={np.percentile(slope[veg],90):.1f}deg, max={slope[veg].max():.1f}deg")

# %% [markdown]
# ## A4. How the clustering feeds the path planner
# The zone map is the **only** thing passed from clustering to path planning. The
# four panels below show the hand-off, step by step:
#
# 1. **Management zones** — the clustering output (0 = soil/inter-row = drivable,
#    1..K = vine vigor).
# 2. **Target mask** — the zones that need the supplement. For WATER that is *every*
#    vine zone; for NITROGEN the highest-vigor zone is dropped.
# 3. **Drivable corridors** — bare-soil pixels (zone 0) inside the field that lie
#    next to a target zone. This is where the robot is allowed to drive, so it
#    stays off the vines.
# 4. **Coverage swaths** — parallel boustrophedon lines laid on those corridors,
#    then chained into a route. (With `--use-elevation` the between-swath
#    transitions are routed by slope-aware A*.)

# %%
show_stages(d1)

# %% [markdown]
# ## A5. Coverage path — WATER (treats every vine zone)
# *"Treats every vine zone"* means the target is **all** canopy pixels, i.e.
# `preview_map >= 1` (every vigor level), because irrigation is applied to the
# whole vineyard. The planner then lays swaths on the soil corridors next to that
# target and chains them into the route below. (`target_mask(zones, "water")`.)

# %%
show_mission(d1, "water", "#00e5ff")

# %% [markdown]
# ## A6. Coverage path — NITROGEN (skips the highest-vigor zone)
# Target = `1 <= preview_map < max_zone`, i.e. only the low/medium-vigor canopy;
# the highest-vigor zone is dropped (already vigorous → no extra nitrogen). Fewer
# target pixels → a smaller mission than water. (`target_mask(zones, "nitrogen")`.)
#
# *On the path shape:* the swaths are horizontal sweeps over the (fragmented) soil
# corridors, chained by a nearest-neighbour "snake". This covers the field but
# looks jagged, and a few long straight **connectors** can cut across / outside the
# field — those are transition links, not coverage. With `--use-elevation` the
# connectors are routed by A* through the corridors instead (Section A6b). Cleaner
# straight rows would require sweeping along the true row angle (a possible next step).

# %%
show_mission(d1, "nitrogen", "#ff5ecb")

# %% [markdown]
# ## A6b. Slope-aware A\* mission (full field)
# With `--use-elevation`, the transitions between swaths are routed with A\* over a
# DEM-aware cost grid (soil = cheap, vines = obstacle, steep slopes = blocked) —
# the same role A\* plays in the reference pipeline, here run over the whole field.
# The zoom shows the route weaving through the soil corridors instead of cutting
# straight across the vines.

# %%
show_astar_mission(d1, "nitrogen")

# %% [markdown]
# ## A6c. P80 demand coverage (reference method's selection rule)
# The reference pipeline (Side-wise P80 CPP) treats only the **top 20% of the
# priority layer**. Applied here, the demand region is the top-20% highest-vigor
# canopy by NDRE, and our corridor planner covers it. This is a target-selection
# rule — like an extra "supplement" — not a clustering method.

# %%
show_p80(d1)

# %% [markdown]
# ## A7. Method comparison — clustering quality and path covering
# *How computed:* run_pipeline saves each method's feature matrix and labels; here
# we compute **silhouette / Davies-Bouldin / Calinski-Harabasz** on them,
# **balance** = smallest/largest zone size, **fragmentation** = connected
# components per 1000 canopy px, and for path covering **coverage %** = target
# pixels within the working width of a swath.
# `water` coverage is the same across methods (target = all vine zones = one
# vegetation mask); `nitrogen` differs because each method draws the highest-vigor
# zone differently.

# %%
method_comparison("data")

# %% [markdown]
# ## A8. Final coverage path per clustering method (NITROGEN)
# Water coverage is method-independent (it targets every vine zone), so the
# nitrogen mission is shown — it changes with the method because each one draws
# the highest-vigor zone (which nitrogen skips) differently.

# %%
show_missions_by_method("data", "nitrogen")

# %% [markdown]
# ---
# # PART B — Dataset `data2` (OpenDroneMap orthophoto, no DEM)
# Same flow via the ODM adapter; the slope / A\* elevation stage is skipped.

# %% [markdown]
# ## B1. Management zones

# %%
d2 = load_ds("data2")
show_zones(d2)

# %% [markdown]
# ## B1b. Old vs final zones

# %%
show_old_vs_final(d2)

# %% [markdown]
# ## B2. Per-zone statistics

# %%
zone_table(d2)

# %% [markdown]
# ## B2b. Index distributions within each zone

# %%
show_zone_distributions(d2)

# %% [markdown]
# ## B3. Clustering → path planning stages

# %%
show_stages(d2)

# %% [markdown]
# ## B4. Coverage path — WATER

# %%
show_mission(d2, "water", "#00e5ff")

# %% [markdown]
# ## B5. Coverage path — NITROGEN

# %%
show_mission(d2, "nitrogen", "#ff5ecb")

# %% [markdown]
# ## B5b. P80 demand coverage

# %%
show_p80(d2)

# %% [markdown]
# ## B6. Method comparison

# %%
method_comparison("data2")

# %% [markdown]
# ## B7. Final coverage path per clustering method (NITROGEN)

# %%
show_missions_by_method("data2", "nitrogen")

