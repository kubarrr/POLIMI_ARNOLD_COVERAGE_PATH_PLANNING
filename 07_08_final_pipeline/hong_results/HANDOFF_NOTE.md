# Handoff Note: Side-wise P80 CPP Reproducible Baseline

Hi team,

I have organized a reproducible baseline for the current UAV-informed UGV coverage path planning workflow.

Please start from:

```text
reproducible/sidewise_p80_cpp/
```

The core implementation is now self-contained in:

```text
reproducible/sidewise_p80_cpp/sidewise_p80_cpp_pipeline.py
```

Run command from the project root:

```powershell
.\.venv\Scripts\python.exe reproducible\sidewise_p80_cpp\run_pipeline.py --config reproducible\sidewise_p80_cpp\config.json
```

The current model does the following:

1. Splits the safe area into the two road-separated field sides.
2. Computes P80 separately within each side using the side-specific 80th percentile of the priority layer.
3. Generates row-aligned CPP swaths only inside the side-wise P80 demand regions.
4. Does not connect swaths through non-demand areas if doing so would leave P80.

Main outputs:

```text
outputs/evaluation/target_regions_sidewise_P80.geojson
outputs/evaluation/target_regions_sidewise_P80_stats.csv
outputs/final_mission_two_side_sidewise_p80_cpp/mission_routes.geojson
outputs/final_mission_two_side_sidewise_p80_cpp/mission_waypoints.csv
outputs/final_mission_two_side_sidewise_p80_cpp/targeted_cpp_metrics.json
outputs/figures/fig_two_side_sidewise_p80_cpp.png
```

Current baseline metrics:

- Southwest side-wise P80 area: ~1971.78 m²
- Northeast side-wise P80 area: ~980.37 m²
- Global side-wise P80 coverage: ~99.34%
- Outside-P80 route length: 0.0 m
- Outside-safe-area route length: 0.0 m

Important caveat:

This is a demand-region CPP baseline, not the final collision-free UGV execution route. Disconnected transitions are expected because the current strict configuration does not allow headland connectors outside P80. The next step is to introduce validated row-corridor and headland-transfer lanes so the UGV can execute the coverage task continuously without crop collision.
