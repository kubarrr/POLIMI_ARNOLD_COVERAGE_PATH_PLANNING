# 07_08 Final Pipeline

Organised deliverable for the vineyard **clustering + coverage-path-planning**
project, split by author.

## Folders

### `ana_pipeline_preparation/`
The clustering + combined pipeline **as delivered by Ana** (unchanged, from the
original hand-off): multispectral clustering (6 methods) + CaSP + the first
path-planning layer (`mission_planner.py`), with `pipeline.py` as the entry
point and `run.sh` as the Linux launcher.

### `hong_results/`
**Hong's** reference **Side-wise P80 CPP** planner: coverage path planning on
GeoJSON polygons (safe-area, per-side P80 demand regions) from an OpenDroneMap
run in EPSG:32629, plus his standalone results notebook. This is the polygon-based
"other code" the clustering path planner was adapted from.

### `kuba_pipeline_update_notebook/`
**Kuba's** update — the runnable end-to-end flow and the report:
- `run_pipeline.py` — end-to-end run with flags (clustering → contiguous zones →
  water / nitrogen / P80 coverage paths → saved outputs).
- `report.ipynb` — runs `run_pipeline` and visualises / aggregates the results
  for two datasets (`data`, `data2`).
- `prepare_odm.py` — adapter so an OpenDroneMap orthophoto (like `data2`) feeds
  the same flow.
- `run.py` / `run.bat` — cross-platform launcher (creates a venv, installs deps,
  runs the flow) — including the missing Windows path.
- `mission_planner.py` / `processing.py` / `utils.py` — the shared modules the
  flow imports (Kuba's updated `mission_planner`/`utils`).

See `kuba_pipeline_update_notebook/README.txt` for how to run.
