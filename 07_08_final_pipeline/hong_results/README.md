# Reproducible Side-wise P80 CPP Pipeline

This package reproduces the current UAV-informed ground robot coverage path planning result used in the project discussion.

The current planning logic is:

1. Split the field into two road-separated safe-area polygons.
2. Compute P80 demand regions independently inside each side, using the side-specific 80th percentile of the vegetation-priority layer.
3. Generate row-aligned boustrophedon CPP swaths only inside the side-wise P80 demand patches.
4. Do not connect disconnected P80 patches through non-demand areas.
5. Report coverage, path length, disconnected transitions, and crop-collision diagnostics.

Important: this is a reproducible demand-region CPP baseline. It is not yet the final collision-free UGV execution planner. The final execution planner still needs validated row-corridor and headland-transfer lanes.

## Folder contents

- `config.json` — paths and main planning parameters.
- `run_pipeline.py` — reproducible English entry point.
- `sidewise_p80_cpp_pipeline.py` — self-contained implementation of side-wise P80 extraction, row-aligned CPP generation, metrics, and plotting.
- `requirements.txt` — Python package requirements.

The runner no longer hides the implementation in the project experimental scripts. The main algorithm is in `sidewise_p80_cpp_pipeline.py`, while `run_pipeline.py` is only a small CLI wrapper.

## Required input files

Relative to the project root:

- `data/cpp_ready/geometry/safe_area.geojson`
- `data/experiment_d/priority_layer.tif`
- `outputs/evaluation/vegetation_body_mask.geojson`
- `outputs/evaluation/row_orientation.json`
- `data/processed/risk_maps/traversability_risk.tif`

The default CRS for planning outputs is EPSG:32629.

## How to run

From the project root:

```powershell
.\.venv\Scripts\python.exe reproducible\sidewise_p80_cpp\run_pipeline.py --config reproducible\sidewise_p80_cpp\config.json
```

If the virtual environment is not available, install the packages in `requirements.txt` first.

## Main outputs

The default output directory is:

```text
outputs/final_mission_two_side_sidewise_p80_cpp
```

Key files:

- `mission_routes.geojson` — combined route geometry for all side-wise P80 patches.
- `mission_swaths.geojson` — coverage swaths only.
- `mission_waypoints.csv` — waypoint table with heading.
- `side_metrics.csv` — side-level metrics.
- `patch_metrics.csv` — patch-level metrics.
- `targeted_cpp_metrics.json` — complete metrics summary.

The side-wise P80 demand regions are saved to:

```text
outputs/evaluation/target_regions_sidewise_P80.geojson
outputs/evaluation/target_regions_sidewise_P80_stats.csv
```

The figure is saved to:

```text
outputs/figures/fig_two_side_sidewise_p80_cpp.png
```

## Code structure

The implementation is intentionally kept in one readable Python file:

```text
sidewise_p80_cpp_pipeline.py
```

Main function groups:

- Geometry utilities: geometry cleanup, polygon/line splitting.
- Side-wise target extraction: side-specific P80 thresholding from the priority raster.
- CPP generation: row-aligned swath generation and snake ordering.
- Connector handling: within-patch headland connectors only; failed connectors become disconnected transitions.
- Metrics: P80 coverage, outside-P80 length, outside-safe length, crop-collision diagnostics.
- Visualization: reproducible PNG figure for the current baseline.

## Current baseline result

With the default configuration:

- Southwest side P80 threshold: approximately `0.514`
- Northeast side P80 threshold: approximately `0.885`
- Southwest side-wise P80 area: approximately `1971.78 m²`
- Northeast side-wise P80 area: approximately `980.37 m²`
- Global side-wise P80 coverage: approximately `99.34%`
- Outside-P80 route length: `0.0 m`
- Outside-safe-area route length: `0.0 m`

The high disconnected-transition count is expected under the strict setting where the planner is not allowed to leave P80 demand patches for headland transfers.

## Method notes for reviewers

- `P80` is not a global threshold. It is computed separately for each road-separated side.
- The route is constrained to P80 demand regions, not the full safe area.
- Crop-body crossing is reported as a diagnostic because the present CPP is demand-region based. It does not yet convert P80 coverage swaths into adjacent row-corridor execution lanes.
- Disconnected transitions mean a local connector would require leaving the P80 patch; those connectors are intentionally not forced.
