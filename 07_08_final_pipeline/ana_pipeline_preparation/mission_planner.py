import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, rotate, sobel, binary_dilation, median_filter

def create_corridor_cost_grid(preview_map, suplement="water"):

    """
    Creates cost matrix for vineyard - zone 0 is walkable.

    Args:
        preview_map: 2D matrix representing the classified zones
        suplement: Type of input application (water for default)

    Returns:
        Cost matrix
        
    """

    cost_grid = np.ones(preview_map.shape, dtype="float32")
    max_zone = np.max(preview_map)

    if suplement == "nitrogen":
        # Nitrogen needed zoes block transit
        obstacle_mask = (preview_map >= 1) & (preview_map < max_zone)
        cost_grid[obstacle_mask] = np.inf
        # High vigor can be used as transit but higher cost than nothing
        high_vigor_mask = preview_map == max_zone
        cost_grid[high_vigor_mask] = 2.5
    else:
        # All vine blocks
        obstacle_mask = preview_map > 0
        cost_grid[obstacle_mask] = np.inf

    # Mask to isolate background
    field_mask = preview_map > 0
    field_mask = binary_closing(field_mask, structure=np.ones((5, 5)))
    field_mask = binary_fill_holes(field_mask)

    # Outside the field cost is infinity
    cost_grid[~field_mask] = np.inf

    # Zone 0 is the preferable
    cost_grid[(preview_map == 0) & field_mask] = 1.0
    return cost_grid


def astar_pixel_search(cost_grid, start, end):
    """Based on the algorithm not by me - not commented."""
    h, w = cost_grid.shape
    start_cell = (start[1], start[0])  # (row, col)
    end_cell = (end[1], end[0])

    #this might be the problem it should be able to move to any direction but then it gets veryyyyyy slow
    moves = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.414),
        (-1, 1, 1.414),
        (1, -1, 1.414),
        (1, 1, 1.414),
    ]

    queue = [(0.0, start_cell)]
    cost_so_far = {start_cell: 0.0}
    came_from = {start_cell: None}

    while queue:
        _, current = heapq.heappop(queue)

        if current == end_cell:
            break

        for dr, dc, step in moves:
            nb = (current[0] + dr, current[1] + dc)

            if 0 <= nb[0] < h and 0 <= nb[1] < w:
                cell_cost = cost_grid[nb[0], nb[1]]
                if np.isinf(cell_cost):
                    continue

                new_cost = cost_so_far[current] + step * float(cell_cost)
                if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                    cost_so_far[nb] = new_cost
                    came_from[nb] = current
                    # Heurística de distância Euclidiana até ao alvo
                    heuristic = math.hypot(
                        nb[0] - end_cell[0], nb[1] - end_cell[1]
                    )
                    heapq.heappush(queue, (new_cost + heuristic, nb))

    if end_cell not in cost_so_far:
        return [start, end]

    path = []
    curr = end_cell
    while curr is not None:
        path.append((curr[1], curr[0]))  
        curr = came_from[curr]
    path.reverse()
    return path


def detect_row_angle(preview_map):
    """Angle detection?? Not working...."""
    dilated_map = binary_dilation(preview_map > 0, structure=np.ones((3,3)))
    
    map_clean = median_filter(dilated_map.astype(float), size=3)
    
    dx = sobel(map_clean, axis=1)
    dy = sobel(map_clean, axis=0)
    
    magnitude = np.hypot(dx, dy)
    angles = np.arctan2(dy, dx)

    threshold = np.percentile(magnitude[magnitude > 0], 85) 
    mask = magnitude > threshold
    
    angles_deg = np.degrees(angles[mask])
    
    row_angles = (angles_deg + 90) % 180
    row_angles[row_angles > 90] -= 180
    
    hist, bin_edges = np.histogram(row_angles, bins=180, range=(-90, 90))
    return bin_edges[np.argmax(hist)]


def build_complete_mission_path(preview_map, swaths_ordered, suplement='water', use_elevation=False):
    """Join eeverything."""
    full_trajectory = []
    cost_grid = create_corridor_cost_grid(preview_map, suplement) if use_elevation else None
    
    for i in range(len(swaths_ordered)):
        p1, p2 = swaths_ordered[i]
        full_trajectory.append((p1, p2))
        
        if i < len(swaths_ordered) - 1:
            next_p1, _ = swaths_ordered[i + 1]
            if use_elevation:
                transition = astar_pixel_search(cost_grid, p2, next_p1)
                for j in range(len(transition) - 1):
                    full_trajectory.append((transition[j], transition[j+1]))
            else:
                full_trajectory.append((p2, next_p1))
    return full_trajectory


def rotate_point_back(x, y, cx_rot, cy_rot, cx_orig, cy_orig, angle_deg):
    """Roatet to original --- to be perfected/implemented."""
    rad = math.radians(angle_deg)
    dx = x - cx_rot
    dy = y - cy_rot
    
    new_x = cx_orig + (dx * math.cos(rad) - dy * math.sin(rad))
    new_y = cy_orig + (dx * math.sin(rad) + dy * math.cos(rad))
    return (new_x, new_y)


def generate_autonomous_mission(preview_map, suplement, spacing_px, use_elevation):

    #No elevation -> simple mode
    #current problem -> robot over vines... i couldnt fix it i tried but i think its because i have to work on low resolution...
    if not use_elevation:
        raw_swaths = generate_zone_swaths(preview_map, suplement=suplement, spacing_px=spacing_px)
        ordered_swaths = order_swaths_snake(raw_swaths)
        return ordered_swaths

    
    # With elevaton -> to detected obstacles and what else -> a* makes it too slow for my pc + low resolution quality
    auto_angle = detect_row_angle(preview_map)
    
    h_orig, w_orig = preview_map.shape
    cy_orig, cx_orig = h_orig / 2.0, w_orig / 2.0
    rotated_map = rotate(preview_map, -auto_angle, order=0, reshape=True, cval=0)
    h_rot, w_rot = rotated_map.shape
    cy_rot, cx_rot = h_rot / 2.0, w_rot / 2.0

    raw_swaths = generate_zone_swaths(rotated_map, suplement=suplement, spacing_px=spacing_px)
    if not raw_swaths: return []
        
    ordered_swaths = order_swaths_snake(raw_swaths)
    trajectory_rot = build_complete_mission_path(rotated_map, ordered_swaths, suplement=suplement, use_elevation=True)

    final_trajectory = []
    for p1, p2 in trajectory_rot:
        pt1_orig = rotate_point_back(p1[0], p1[1], cx_rot, cy_rot, cx_orig, cy_orig, auto_angle)
        pt2_orig = rotate_point_back(p2[0], p2[1], cx_rot, cy_rot, cx_orig, cy_orig, auto_angle)
        final_trajectory.append((pt1_orig, pt2_orig))
    return final_trajectory

def generate_zone_swaths(preview_map, suplement="water", spacing_px=4.0):
    """
    Trajectory lines based on supplement.
    Supplement == 'water':
        - All vines

    Supplement == 'nitrogen':
        - Map low vigor zones
        - High level zones only as transit between zones

    Zone 0 (Background) discarded ALWAYS!.

    Args:
        preview_map: 2D matrix representing the classified zones
        suplement: Type of input application (water for default)
        spacing_px: Row spacing density measured in matrix pixels, how far apart the horizontal parallel sweeps are (4 for deafult)

    Returns:
        list of tuple: A list of raw coordinate pairs defining the straight line segments constrained to valid vineyard masks.


    """
    h, w = preview_map.shape
    lines_px = []

    # Determine highest vigor in themap
    max_zone = np.max(preview_map)

    for r in range(0, h, int(spacing_px)):
        in_line = False
        start_c = None

        for c in range(w):
            pixel_zone = preview_map[r, c]

            # Any pixel with plants (>0) is valid and used for the trajectory
            is_valid_pixel = pixel_zone > 0

            if is_valid_pixel:
                if not in_line:
                    in_line = True
                    start_c = c
            else:
                # Out of the vine (into zone 0) -> close segment
                if in_line:
                    if (c - start_c) > 3:
                        # If nitrogen we remove the segent if its only fot transit (max vigor)
                        if suplement == "nitrogen":
                            real_segment = preview_map[r, start_c:c]
                            # Only valid for the necessary zones (between 1 and max_zone - 1)
                            if np.any(
                                (real_segment >= 1) & (real_segment < max_zone)
                            ):
                                lines_px.append(((start_c, r), (c - 1, r)))
                            else:
                                # In case we are on a hig vigor zone, see if any zones to the left or right need transit between them
                                left_needed = np.any(
                                    (preview_map[r, :start_c] >= 1)
                                    & (preview_map[r, :start_c] < max_zone)
                                )
                                right_needed = np.any(
                                    (preview_map[r, c:] >= 1)
                                    & (preview_map[r, c:] < max_zone)
                                )
                                if left_needed and right_needed:
                                    lines_px.append(((start_c, r), (c - 1, r)))
                        else:
                            lines_px.append(((start_c, r), (c - 1, r)))
                    in_line = False

        # End image but line still active
        if in_line:
            if (w - start_c) > 3:
                if suplement == "nitrogen":
                    real_segment = preview_map[r, start_c:w]
                    if np.any((real_segment >= 1) & (real_segment < max_zone)):
                        lines_px.append(((start_c, r), (w - 1, r)))
                    else:
                        left_needed = np.any(
                            (preview_map[r, :start_c] >= 1)
                            & (preview_map[r, :start_c] < max_zone)
                        )
                        # In case we are on a hig vigor zone, see if any zones to the left need transit between them (end of image to the right)!
                        if left_needed:
                            lines_px.append(((start_c, r), (w - 1, r)))
                else:
                    lines_px.append(((start_c, r), (w - 1, r)))

    return lines_px


def order_swaths_snake(swaths):
    """
    Orders a list of parallel line segments into a continuous zigzag snake path.

    This optimization alternates the steering direction of every odd row index to
    ensure the robot can navigate from the end of one row directly to the beginning
    of the next adjacent row without redundant backtracking.

    Args:
        swaths: Unordered trajectory lines containing start and end pixel coordinates.

    Returns:
        list of tuple: Ordered trajectory where connected endpoints match the continuous zigzag movement.
    """

    ordered = []
    for idx, (p1, p2) in enumerate(swaths):
        if idx % 2 == 1:
            ordered.append((p2, p1))
        else:
            ordered.append((p1, p2))
    return ordered


def plot_and_save_robot_mission(
    preview_map, cmap_zones, norm, trajectory, filename_output, suplement
):
    """
    Renders and exports the final technical autonomous mission map overlaying the robot's route.

    Args:
        preview_map: 2D grid matrix of the agronomic classification.
        cmap_zones: Colormap mapping integers for zone colors.
        norm: Boundary normalization for discrete integer pixel bounds to colormap slots.
        trajectory: The sequenced (snake ordered) coordinate pairs representing the robot's final physical route.
        filename_output: Destination file path where the generated PNG image will be saved.
        suplement: Selected treatment flag (water or nitrogen)

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(12, 10))

    # Originnal map
    ax.imshow(
        preview_map, cmap=cmap_zones, norm=norm, interpolation="nearest"
    )

    for p1, p2 in trajectory:
        col_coords = [p1[0], p2[0]]
        row_coords = [p1[1], p2[1]]

        # cian blue for path
        ax.plot(
            col_coords,
            row_coords,
            color="#00FFFF",
            linewidth=2.0,
            alpha=0.9,
            zorder=50,
        )

    plt.title(
        f"Robotic path for: {suplement.upper()}", fontsize=14, color="white"
    )
    ax.set_xlim(0, preview_map.shape[1])
    ax.set_ylim(preview_map.shape[0], 0)
    plt.axis("off")

    plt.savefig(
        filename_output,
        dpi=300,
        bbox_inches="tight",
        facecolor="black",
        edgecolor="none",
    )
    plt.close()
    print(f"Path saved in: {filename_output}")
