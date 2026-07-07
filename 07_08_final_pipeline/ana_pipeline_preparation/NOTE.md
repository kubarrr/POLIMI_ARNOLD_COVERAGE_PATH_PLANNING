# Ana — clustering + combined pipeline (original)

Delivered by Ana, kept **unchanged** from the original hand-off.

- `pipeline.py` — entry point: multispectral clustering (K-Means, MBKMeans,
  Agglomerative, Spectral, Fuzzy C-Means, GMM) + CaSP smoothing, then the first
  path-planning layer.
- `mission_planner.py` — the two path-planning algorithms (corridor swaths per
  supplement; A* connectors).
- `processing.py`, `utils.py` — zone sorting / CaSP / raster helpers.
- `run.sh` — Linux launcher.

This is the starting point Kuba's update builds on (see
`../kuba_pipeline_update_notebook/`).
