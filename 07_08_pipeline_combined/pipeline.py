import argparse
import gc
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min

from utils import get_smallest_dimensions, collect_training_samples, load_paths_from_txt
from processing import sort_labels_by_ndre, apply_tile_casp

from mission_planner import *

warnings.filterwarnings('ignore')



def run_prediction_memmap(method_name, model_obj, scaler, label_mapping, paths, h_min, w_min, tile_size=2048):
    """
    Execute full-scale image classification and spatial smoothing block-by-block, 
    writing results directly to an on-disk binary file to handle massive rasters on low RAM.

    Args:
        method_name: Name of the clustering algorithm
        model_obj: Trained ML with centroids
        scaler: Scaler to normalize pixels values
        label_mapping: Dictionary with agrnomical label starting in 1
        paths: Paths for the text files
        h_min: minimum height
        w_min: minimum width
        tile_size: window size for processing

    Returns:
        final_map: Final 2D of the whole map smoothed
        filename: name of the biianry file in disk

    """

    # Create file in disk
    # Replace blank spaces with uderscores and force lowercase
    filename = f"temp_{method_name.lower().replace(' ', '_')}.dat"

    # Array in disk -> initialize 0
    final_map = np.memmap(filename, dtype=np.uint8, mode='w+', shape=(h_min, w_min))
    final_map[:] = 0
    
    path_nir, path_re, path_r, path_g, path_ndvi = paths
    
    with rasterio.open(path_nir) as nir_src, rasterio.open(path_re) as re_src, \
         rasterio.open(path_r) as r_src, rasterio.open(path_g) as g_src, \
         rasterio.open(path_ndvi) as ndvi_src:

         #run by tiles
        for i in range(0, h_min, tile_size):
            for j in range(0, w_min, tile_size):

                #prevent out of bounds
                th = min(tile_size, h_min - i)
                tw = min(tile_size, w_min - j)
                
                wnd = Window(j, i, tw, th)

                #get window of each block
                nir_t = nir_src.read(1, window=wnd).astype(np.float32)
                re_t  = re_src.read(1, window=wnd).astype(np.float32)
                r_t   = r_src.read(1, window=wnd).astype(np.float32)
                ndvi_t= ndvi_src.read(1, window=wnd).astype(np.float32)
                
                #calculate block ndre
                ndre_t = (nir_t - re_t) / (nir_t + re_t + 1e-6)
                valid_mask_2d = ndre_t > 0.1
                
                #noprocesses if not enough valid pixels
                if np.sum(valid_mask_2d) < 10:
                    continue

                #organize valid pixels and scale
                feats = np.column_stack((
                    ndre_t[valid_mask_2d], ndvi_t[valid_mask_2d],
                    nir_t[valid_mask_2d], r_t[valid_mask_2d], nir_t[valid_mask_2d]
                ))
                
                scaled = scaler.transform(feats)


                #chose algorithm
                if method_name in ['K-Means', 'Mini-Batch K-Means']:
                    raw_lbls = model_obj.predict(scaled)
                elif method_name == 'GMM':
                    raw_lbls = model_obj.predict(scaled)
                elif method_name in ['Agglomerative', 'Spectral Clustering']:
                    raw_lbls, _ = pairwise_distances_argmin_min(scaled, model_obj)
                elif method_name == 'Fuzzy C-Means':
                    import skfuzzy as fuzz
                    u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(scaled.T, model_obj, m=2.0, error=0.005, maxiter=150)
                    raw_lbls = np.argmax(u, axis=0)
                
                # Translate result with the previously created dictionary
                sorted_lbls_list = []
                for lbl in raw_lbls:
                    # assume zone 1 if can't find trasnlation
                    correct_zone = label_mapping.get(lbl, 1)
                    sorted_lbls_list.append(correct_zone)
                sorted_lbls = np.array(sorted_lbls_list)

                # reconstruct 2D image
                clusters_2d = np.zeros_like(ndre_t, dtype=np.uint8)
                clusters_2d[valid_mask_2d] = sorted_lbls

                #apply casp
                final_casp = apply_tile_casp(clusters_2d, valid_mask_2d, len(label_mapping))
                final_map[i:i+th, j:j+tw] = final_casp
                final_map.flush()
                
    return final_map, filename


def run_prediction_ram(method_name, model_obj, scaler, label_mapping, paths, h_min, w_min):


    """
    Execute classification on RAM
    
    Args:
        method_name: Name of the clustering algorithm
        model_obj: Trained ML with centroids
        scaler: Scaler to normalize pixels values
        label_mapping: Dictionary with agrnomical label starting in 1
        paths: Paths for the text files
        h_min: minimum height
        w_min: minimum width

    Returns:
        final_map: Final 2D of the whole map smoothed

    """

    # create matrix on RAM memoria
    final_map = np.zeros((h_min, w_min), dtype=np.uint8)
    
    path_nir, path_re, path_r, path_g, path_ndvi = paths
    
    # open and reada images
    with rasterio.open(path_nir) as nir_src, rasterio.open(path_re) as re_src, \
         rasterio.open(path_r) as r_src, rasterio.open(path_ndvi) as ndvi_src:
         
        nir_t = nir_src.read(1).astype(np.float32)
        re_t  = re_src.read(1).astype(np.float32)
        r_t   = r_src.read(1).astype(np.float32)
        ndvi_t = ndvi_src.read(1).astype(np.float32)
        
    # calculate ndre
    ndre_t = (nir_t - re_t) / (nir_t + re_t + 1e-6)
    valid_mask_2d = ndre_t > 0.1
    
    if np.sum(valid_mask_2d) < 10:
        return final_map
        
    # organize and scale all the attributes
    feats = np.column_stack((
        ndre_t[valid_mask_2d], ndvi_t[valid_mask_2d],
        nir_t[valid_mask_2d], r_t[valid_mask_2d], nir_t[valid_mask_2d]
    ))
    scaled = scaler.transform(feats)
    
    # Predict according to chosen method
    if method_name in ['K-Means', 'Mini-Batch K-Means']:
        raw_lbls = model_obj.predict(scaled)
    elif method_name == 'GMM':
        raw_lbls = model_obj.predict(scaled)
    elif method_name in ['Agglomerative', 'Spectral Clustering']:
        from sklearn.metrics import pairwise_distances_argmin_min
        raw_lbls, _ = pairwise_distances_argmin_min(scaled, model_obj)
    elif method_name == 'Fuzzy C-Means':
        import skfuzzy as fuzz
        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(scaled.T, model_obj, m=2.0, error=0.005, maxiter=150)
        raw_lbls = np.argmax(u, axis=0)
    
    # Translate result with the previously created dictionary
    sorted_lbls = np.array([label_mapping.get(lbl, 1) for lbl in raw_lbls])
    
    # Reconstruct 2D image RAM
    clusters_2d = np.zeros_like(ndre_t, dtype=np.uint8)
    clusters_2d[valid_mask_2d] = sorted_lbls
    
    #apply casp
    final_map = apply_tile_casp(clusters_2d, valid_mask_2d, len(label_mapping))
    
    return final_map


def main():
    # Get terminal arguments
    parser = argparse.ArgumentParser(description="Pipeline Multimétodo de Zonamento Vinícola")
    parser.add_argument('--method', type=str, default='kmeans', 
                        choices=['kmeans', 'mbkmeans', 'agglomerative', 'spectral', 'fcm', 'gmm', 'all'],
                        help="Chose clustering algorith (Default: kmeans)")
    parser.add_argument('--k', type=int, default=3, help="Zone number (Default: 3)")
    parser.add_argument('--high-quality', '--high-resolution', action='store_true', dest='high_quality',
                        help="High quality full-resolution RAM processing (alias: --high-resolution).")
    parser.add_argument('--config', type=str, default='paths.txt',
                        help="Txt file with paths for orthometry and DEM (Default: paths.txt)")
    parser.add_argument('--suplement', type=str, default='water', choices=['water', 'nitrogen'],
                        help="Supplement type for path planning (Default: water)")
    parser.add_argument('--use-elevation', action='store_true',
                        help="Use full pixel-by-pixel A* to plan paths avoiding slope obstacles (Slower).")
    args = parser.parse_args()


    print(f"Loading paths: {args.config}")
    paths, path_dem, path_shape = load_paths_from_txt(args.config)
    
    h_min, w_min = get_smallest_dimensions(paths)
    print(f"Image size {h_min}x{w_min}")

    # get sample and train
    X_sample = collect_training_samples(paths, h_min, w_min)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_sample)
    
    methods_to_run = {}
    print("Training models...")
    
    if args.method in ['kmeans', 'all']:
        km = KMeans(n_clusters=args.k, random_state=123, n_init=5).fit(X_scaled)
        mapping = sort_labels_by_ndre(km.labels_, X_sample)
        methods_to_run['K-Means'] = (km, mapping)
        
    if args.method in ['mbkmeans', 'all']:
        mbkm = MiniBatchKMeans(n_clusters=args.k, random_state=123, batch_size=2048, n_init=3).fit(X_scaled)
        mapping = sort_labels_by_ndre(mbkm.labels_, X_sample)
        methods_to_run['Mini-Batch K-Means'] = (mbkm, mapping)
        
    if args.method in ['agglomerative', 'all']:
        X_sub = X_scaled[:3000]
        agg = AgglomerativeClustering(n_clusters=args.k, linkage='ward').fit(X_sub)
        mapping = sort_labels_by_ndre(agg.labels_, X_sample[:3000])
        centroids = np.array([X_sub[agg.labels_ == c].mean(axis=0) for c in np.unique(agg.labels_)])
        methods_to_run['Agglomerative'] = (centroids, mapping)
        
    if args.method in ['spectral', 'all']:
        X_sub = X_scaled[:2000]
        spec = SpectralClustering(n_clusters=args.k, affinity='nearest_neighbors', n_neighbors=10, random_state=123, n_jobs=-1).fit(X_sub)
        mapping = sort_labels_by_ndre(spec.labels_, X_sample[:2000])
        centroids = np.array([X_sub[spec.labels_ == c].mean(axis=0) for c in np.unique(spec.labels_)])
        methods_to_run['Spectral Clustering'] = (centroids, mapping)
        
    if args.method in ['fcm', 'all']:
        try:
            import skfuzzy as fuzz
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-fuzzy', '-q'])
            import skfuzzy as fuzz
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_scaled.T, c=args.k, m=2.0, error=0.005, maxiter=100, seed=42)
        mapping = sort_labels_by_ndre(np.argmax(u, axis=0), X_sample)
        methods_to_run['Fuzzy C-Means'] = (cntr, mapping)
        
    if args.method in ['gmm', 'all']:
        gmm = GaussianMixture(n_components=args.k, covariance_type='full', random_state=123).fit(X_scaled)
        mapping = sort_labels_by_ndre(gmm.predict(X_scaled), X_sample)
        methods_to_run['GMM'] = (gmm, mapping)



    # =========================================================================
    # Configuration for output map
    # =========================================================================

    # Color configuration
    color_list = ['black', 'red', 'yellow', 'limegreen', 'dodgerblue']
    cmap_zones = ListedColormap(color_list[:args.k + 1])
    norm = BoundaryNorm(np.arange(args.k + 2) - 0.5, cmap_zones.N)

    # Define resolution according to computer power
    step = 1 if args.high_quality else 7
    h_v, w_v = h_min // step, w_min // step
    
    with rasterio.open(paths[0]) as ref_src:
        #transform pixels into real world coordinates and save the coordinate reference system
        original_transform = ref_src.transform
        target_crs = ref_src.crs

        # Adjust geographic trasnform matrix according to step
        preview_transform = original_transform * original_transform.scale(step, step)

    with rasterio.open(path_dem) as dem_src:

        # Target array with previous configuration
        dem_aligned = np.zeros((h_v, w_v), dtype=np.float32)
        reproject(
            source=rasterio.band(dem_src, 1), destination=dem_aligned,
            src_transform=dem_src.transform, src_crs=dem_src.crs,
            dst_transform=preview_transform, dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
    # Mask invalid or non positive elevation values
    dem_preview_masked = np.ma.masked_where(dem_aligned <= 0, dem_aligned)



    # =========================================================================
    # Processing loop
    # =========================================================================


    for name, (model_obj, mapping) in methods_to_run.items():
        print(f"\n {name} + CaSP")
        
        # Either disk or RAM processing
        if args.high_quality:
            print(f"High Quality mode -> RAM processing")
            final_map = run_prediction_ram(name, model_obj, scaler, mapping, paths, h_min, w_min)
            filename = None # No disk creation
        else:
            print(f"Default mode -> Disk processing")
            final_map, filename = run_prediction_memmap(name, model_obj, scaler, mapping, paths, h_min, w_min)
        
        preview_map = np.array(final_map[::step, ::step])
        

        # =========================================================================
        #  Path planning by supplement type ----- CHANGE HERE!!
        # =========================================================================

        
        print(f"\nPlanning for supplement: {args.suplement})...")

        # Analyze preview map and generate raw trajectory vectors based on supplemnet - WE CAN ADD OTHER GENERATION METHODS
        filtered_lines = generate_zone_swaths(preview_map, suplement=args.suplement, spacing_px=4.0)
        
        if len(filtered_lines) > 0:
            #Order lines in zigzag
            ordered_swaths = order_swaths_snake(filtered_lines)
            
            final_trajectory = generate_autonomous_mission(
                preview_map=preview_map,
                suplement=args.suplement,
                spacing_px=4.0,
                use_elevation=args.use_elevation # Deciding if it uses A* or not
            )
            
            filename_mission = f"final_path_{args.suplement}_{name.replace(' ', '_')}.png"
            
            # Exexcute final drawing
            plot_and_save_robot_mission(
                preview_map=preview_map,
                cmap_zones=cmap_zones,
                norm=norm,
                trajectory=final_trajectory,
                filename_output=filename_mission,
                suplement=args.suplement
            )
        else:
            print(f"Impossible to generate route for {args.suplement}.")

            
        # =========================================================================
        # Clustering and DEM graphs
        # =========================================================================

        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].format_coord = lambda x, y: ""
        axes[1].format_coord = lambda x, y: ""
        
        im1 = axes[0].imshow(dem_preview_masked, cmap='terrain')
        axes[0].set_title('Topographic ALignment DEM')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04).set_label('Elevation [m]')

        im2 = axes[1].imshow(preview_map, cmap=cmap_zones, norm=norm, interpolation='nearest')
        axes[1].set_title(f'{name} + CaSP (K={args.k})')
        axes[1].axis('off')
        
        try:
            contours = axes[1].contour(dem_preview_masked, levels=14, colors='white', alpha=0.5, linewidths=1.0)
            axes[1].clabel(contours, inline=True, fontsize=8, fmt='%.1fm')
        except:
            pass
            
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, ticks=range(args.k + 1))
        plt.tight_layout()
        
        del final_map
        gc.collect()
        if filename and os.path.exists(filename):
            os.remove(filename)
            
        plt.show(block=True)
        plt.close(fig)

if __name__ == "__main__":
    main()
