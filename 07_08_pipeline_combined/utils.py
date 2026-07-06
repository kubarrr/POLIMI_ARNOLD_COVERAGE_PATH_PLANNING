import os
import numpy as np
import rasterio
from rasterio.windows import Window

def load_paths_from_txt(txt_path):

    """
    Reads a text file containing the paths for orthomosaic and DEM 
    
    Args:
        txt_path: path to the .txt file containing the filenames
        
    Returns:
        paths: list of 5 spectral band paths
        path_dem: string path to the DEM file
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File '{txt_path}' not found!")
        
    with open(txt_path, 'r') as f:
        # Remove all lines and removes blank spaces and empty lines
        lines = []
        for line in f.readlines():
            line_clean = line.strip()
            if line_clean:
                lines.append(line_clean)

                
    if len(lines) < 6:
        raise ValueError(f"File'{txt_path}' must contain 6 paths (5 spectral + 1 DEM)!")
        
    paths = lines[:5]       # spectral bands
    path_dem = lines[5]     # dem path
    path_shape = lines[6]
    
    return paths, path_dem, path_shape


def get_smallest_dimensions(paths):
    """
    Get the dimension of the smallest .tif in order
    to avoid out of bounds errors (IndexErrors)
    
    Args:
        paths: paths for all the .tif files to be used
    
    Returns:
        h_min: minimum height
        w_min: minimum width
    """
    h_min, w_min = float('inf'), float('inf')
    for p in paths:
        with rasterio.open(p) as src:
            if src.height < h_min: h_min = src.height
            if src.width < w_min: w_min = src.width
    return h_min, w_min



def collect_training_samples(paths, h_min, w_min, sample_size=15000):

    """
    Collect a representative subset of pixels across a regular
    spatial grid to train the ML models -> sample size to prevent RAM crash

    Currently sampling with largely less than the image amount -> with flag will change
    
    Args:
        paths: paths for all the spectral band .tif files
        h_min: minimum height boundary
        w_min: minimum width boundary
        sample_size: fixed total number of pixels to sample for training
                     ajusted due to RAM necessities
    
    Returns:
        A 2D numpy array of shape (sample_size, 5) containing the sub-sampled 
        multispectral features (NDRE, NDVI, NIR, R, G) from active vegetation.
    """

    collected = []
    path_nir, _, path_r, _, path_ndvi = paths

    #36 tiles instead of readin the whole window
    steps_i = np.linspace(0, h_min - 512, 6, dtype=int)
    steps_j = np.linspace(0, w_min - 512, 6, dtype=int)
    
    with rasterio.open(path_nir) as nir_src, \
         rasterio.open(path_r) as r_src, \
         rasterio.open(path_ndvi) as ndvi_src:
         
        for i in steps_i:
            for j in steps_j:
                wnd = Window(j, i, 512, 512)

                #load tiles
                nir_t = nir_src.read(1, window=wnd).astype(np.float32)
                r_t = r_src.read(1, window=wnd).astype(np.float32)
                ndvi_t = ndvi_src.read(1, window=wnd).astype(np.float32)

                #calculate ndre and invalidate paths and etc.
                ndre_t = (nir_t - r_t) / (nir_t + r_t + 1e-6)
                mask_valid = ndre_t > 0.1


                if np.sum(mask_valid) > 50:

                    #extract only the valid pixels
                    feat = np.column_stack((
                        ndre_t[mask_valid], ndvi_t[mask_valid],
                        nir_t[mask_valid], r_t[mask_valid], nir_t[mask_valid]
                    ))

                    #downsampling -> collect only every 10th pixel (for RAM!)
                    collected.append(feat[::10])

    #stack the tiles 2D (pixel * bands)
    all_sampled = np.vstack(collected)

    #extract defined size of points for training randomly 
    idx = np.random.choice(all_sampled.shape[0], min(sample_size, all_sampled.shape[0]), replace=False)
    return all_sampled[idx]
