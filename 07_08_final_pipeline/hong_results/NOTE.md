# Hong — Side-wise P80 CPP (reference)

Hong's reproducible coverage-path-planning baseline, working on GeoJSON polygons
(safe-area + per-side P80 demand regions) from an OpenDroneMap run in EPSG:32629.

- `sidewise_p80_cpp_pipeline.py` — core: two-side split, per-side P80 demand
  extraction, row-aligned boustrophedon swaths, A* headland connectors, metrics.
- `run_pipeline.py`, `config.json` — CLI wrapper + parameters.
- `06_standalone_final_results.ipynb` — standalone results notebook.
- `HANDOFF_NOTE.md`, `README.md` — his documentation.

The P80 "top-20% demand" idea is reproduced on the clustering rasters in Kuba's
report (the P80 sections).
