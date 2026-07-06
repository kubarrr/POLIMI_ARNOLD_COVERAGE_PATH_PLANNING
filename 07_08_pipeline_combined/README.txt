================================================================
  PATH PLANNING PIPELINE  -  clustering + coverage path planning
================================================================

HOW TO RUN
----------
Cross-platform (Windows / Linux / macOS), creates the venv and installs
dependencies automatically:

    python run.py --config paths.txt --method kmeans --k 3 --suplement nitrogen

On Windows you can also double-click / call:

    run.bat --config paths.txt --method kmeans --k 3 --suplement nitrogen

(The old Linux-only run.sh still works too.)


ARGUMENTS
---------
--config <file>       Path list file (mandatory). 7 lines:
                      NIR, RE, R, G, NDVI, DEM, vineyard_shape.shp

--method <name>       Clustering algorithm. One of:
                      kmeans (default), mbkmeans, agglomerative,
                      spectral, fcm, gmm, all

--k <int>             Number of management zones (default: 3)

--suplement <name>    Treatment to plan for: water (default) or nitrogen
                      water    -> covers corridors next to every vine zone
                      nitrogen -> skips the highest-vigor zone (only treats
                                  low/medium vigor areas)

--high-quality        Full-resolution processing in RAM (slower, sharper).
--high-resolution     Alias for --high-quality.

--use-elevation       Plan transitions with slope-aware A* using the DEM,
                      so the route avoids steep terrain (slower).


EXAMPLES
--------
    python run.py --config paths.txt --method all --high-resolution
    python run.py --config paths.txt --suplement water --use-elevation
