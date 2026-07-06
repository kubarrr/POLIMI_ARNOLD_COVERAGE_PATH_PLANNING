import numpy as np
from skimage.morphology import remove_small_objects
from scipy.ndimage import distance_transform_edt

def sort_labels_by_ndre(labels, features_unscaled):
    """
    Sort unsupervised cluster labels deterministically based on their average NDRE,
    ensuring that Zone 1 always represents low vigor and the maximum zone represents high vigor.
    Basically renaming and numerical reordering the labels
    
    Args:
        labels: 1D array of cluster IDs predicted by the untrained model
        features_unscaled: 2D array of the corresponding raw unscaled multispectral data
    
    Returns:
        mapping: a dictionary mapping old arbitrary labels to new ordered agronomic zones (starting from 1)
    """
    #get each unique label generated from model
    uniq = np.unique(labels)
    
    #get ndre for each label group
    means = []
    
    for c in uniq:
        
        # isolate pixel from each group 
        group_pixels = features_unscaled[labels == c]
        
        # get the ndre column and get the mean of the column
        group_ndre = group_pixels[:, 0]
        media_ndre = np.mean(group_ndre)
        
        # Save label, mean pair
        means.append((c, media_ndre))
    
    # Order the list from lowest to highest NDRE (second element of the pair!)
    means.sort(key=lambda x: x[1])
    
    # Mapping dictioary -> starting zones in 1 and not 0
    mapping = {}
    for new_index, (old_label, _) in enumerate(means):
        mapping[old_label] = new_index + 1
        
    return mapping

def apply_tile_casp(clusters_2d, valid_mask_2d, n_clusters, min_island_size=5000):
    """
    Apply Context-Aware Spatial Post-processing (CaSP) to clean individual pixel noise,
    remove small isolated zones (islands) below a threshold, and merge them into neighboring areas.
    
    Args:
        clusters_2d: 2D array representing the raw predicted management zone map of the tile
        valid_mask_2d: 2D boolean mask isolating active vegetation from background soil
        n_clusters: number of target management zones (K)
        min_island_size: minimum pixel area threshold for a zone patch to be preserved
    
    Returns:
        out: a 2D numpy array containing the smooth, tractor-ready management zone map
    """

    #code produced previously
    smoothed = np.zeros_like(clusters_2d)
    valid_large = np.zeros_like(clusters_2d, dtype=bool)
    for c in range(1, n_clusters + 1):
        bm = (clusters_2d == c)
        if np.sum(bm) > min_island_size:
            cm = remove_small_objects(bm, min_size=min_island_size)
            smoothed[cm] = c
            valid_large |= cm
    if not np.any(valid_large):
        return clusters_2d
    _, idx = distance_transform_edt(~valid_large, return_indices=True)
    out = smoothed[idx[0], idx[1]]
    out[~valid_mask_2d] = 0
    return out
