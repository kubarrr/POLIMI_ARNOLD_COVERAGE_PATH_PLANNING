================================================================
  VINEYARD ZONING + COVERAGE PATH PLANNING
================================================================

FILES
-----
run_pipeline.py   End-to-end flow (clustering -> zones -> coverage paths) that
                  writes raw outputs to cache/<name>/ and compare/<name>/.
                  This is the entry point used by the report notebook.
run_pipeline_with_saved_figures.py
                  Same run, but also renders the report figures to PNG files in
                  figures/<name>/ (no notebook needed).
report.ipynb      Report notebook: runs run_pipeline with flags, then visualises
                  and aggregates the results (all reporting code lives here).
prepare_odm.py    Adapter: split an OpenDroneMap multispectral orthophoto into
                  the NIR/RE/R/G/NDVI band rasters the flow expects.
mission_planner.py / processing.py / utils.py   Building blocks (imported by the flow).
run.py / run.bat  Cross-platform launcher: make a venv, install deps, run the flow.


HOW TO RUN (cross-platform; creates the venv and installs deps)
---------------------------------------------------------------
    python run.py --name data --config paths.txt --method all --k 3

On Windows you can also use:  run.bat --name data --config paths.txt --method all --k 3

Or directly (inside an environment that has requirements.txt installed):
    python run_pipeline.py --name data  --config paths.txt      --method kmeans --k 3
    python run_pipeline.py --name data  --config paths.txt      --method all    --k 3
    python run_pipeline.py --name data2 --config paths_data2.txt --method all    --k 3


ARGUMENTS (run_pipeline.py)
---------------------------
--config <file>    Path list: 7 lines -> NIR, RE, R, G, NDVI, DEM, shape.
                   Use "NONE" for a missing DEM or shape (e.g. data2).
--name <str>       Dataset name; outputs go to cache/<name>/ and compare/<name>/.
--method <name>    kmeans (default), mbkmeans, gmm, agglomerative, spectral, fcm,
                   or all (runs every method for the comparison).
--k <int>          Number of management zones (default 3).
--zone-smooth-m    Metre baseline for smoothing the vigor surface (default 2.5;
                   larger -> bigger, more contiguous zones).
--high-quality / --high-resolution   Finer preview (step=3) for sharper zones.
--use-elevation    Route mission transitions with slope-aware A* (needs a DEM).


ADDING A NEW DATASET
--------------------
Cropped multispectral bands + DEM:  point a new paths.txt at them and run.
OpenDroneMap orthophoto:            first run prepare_odm.py, e.g.
    python prepare_odm.py --ortho <odm_orthophoto.tif> --out <dir> \
                          --paths paths_new.txt --dem NONE --shape NONE
then run_pipeline.py --config paths_new.txt.
