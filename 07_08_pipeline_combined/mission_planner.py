import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (binary_closing, binary_fill_holes, binary_dilation,
                           gaussian_filter)


# =====================================================================
#  Masks shared by the swath generator and the A* cost grid
#  (single source of truth for "where can the robot drive / what to treat")
# =====================================================================

def field_mask(preview_map):
    """Boolean mask of the vineyard interior (closed + hole-filled)."""
    fm = preview_map > 0
    fm = binary_closing(fm, structure=np.ones((5, 5)))
    fm = binary_fill_holes(fm)
    return fm


def target_mask(preview_map, suplement):
    """
    Zones that actually need the treatment for the chosen supplement.

    water    -> every vine zone (all vigor levels are irrigated)
    nitrogen -> only low/medium vigor zones; the highest-vigor zone is
                deliberately skipped (it does not need extra nitrogen).
    """
    max_zone = int(np.max(preview_map)) if preview_map.size else 0
    if suplement == "nitrogen":
        return (preview_map >= 1) & (preview_map < max_zone)
    return preview_map >= 1


# =====================================================================
#  Cost grid for the A* planner  (now slope/DEM aware)
# =====================================================================

def slope_degrees(dem, pixel_size_m=1.0, smooth_m=2.0):
    """
    Terrain slope in degrees from a DEM.

    The DEM is first smoothed over a ~`smooth_m` metre baseline, ignoring NoData
    (values <= 0). Without this, a per-pixel gradient on a centimetre-resolution
    DEM is dominated by elevation noise and NoData cliffs, producing meaningless
    slopes of tens of degrees. Smoothing recovers the true, broad-scale terrain
    slope.
    """
    dem = np.asarray(dem, dtype=np.float32)
    valid = dem > 0
    if smooth_m and smooth_m > 0:
        sigma = max(1.0, smooth_m / max(pixel_size_m, 1e-6))
        num = gaussian_filter(np.where(valid, dem, 0.0), sigma)
        den = gaussian_filter(valid.astype(np.float32), sigma)
        dem = num / np.maximum(den, 1e-6)
    gy, gx = np.gradient(dem, float(max(pixel_size_m, 1e-6)))
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope[~valid] = 0.0
    return slope


def create_corridor_cost_grid(preview_map, suplement="water",
                              dem=None, pixel_size_m=1.0,
                              slope_block_deg=20.0, slope_weight=3.0):
    """
    Build the per-pixel travel-cost matrix used by A*.

    Drivable space = bare-soil corridors (zone 0) inside the field. Vine
    blocks are obstacles. When a DEM is supplied, steep terrain is penalised
    and slopes above `slope_block_deg` become impassable -> this is the
    "obstacle avoidance for slopes" the elevation mode is meant to provide.

    Args:
        preview_map:   2D classified zone map.
        suplement:     "water" or "nitrogen".
        dem:           optional 2D elevation array aligned to preview_map.
        pixel_size_m:  ground size of one preview pixel, in metres.
        slope_block_deg: slopes steeper than this are blocked (inf cost).
        slope_weight:  how strongly sub-threshold slopes inflate travel cost.

    Returns:
        2D float32 cost grid (np.inf = not traversable).
    """
    cost_grid = np.ones(preview_map.shape, dtype="float32")
    max_zone = np.max(preview_map)

    if suplement == "nitrogen":
        # low/mid vigor vines block transit; highest-vigor zone is drivable
        # (as a corridor) but more expensive than bare soil.
        obstacle_mask = (preview_map >= 1) & (preview_map < max_zone)
        cost_grid[obstacle_mask] = np.inf
        cost_grid[preview_map == max_zone] = 2.5
    else:
        cost_grid[preview_map > 0] = np.inf  # all vine blocks are obstacles

    # outside the field boundary -> impassable
    fm = field_mask(preview_map)
    cost_grid[~fm] = np.inf

    # bare-soil corridors are the preferred driving surface
    cost_grid[(preview_map == 0) & fm] = 1.0

    # --- DEM / slope term -------------------------------------------------
    if dem is not None:
        dem = np.asarray(dem, dtype=np.float32)
        if dem.shape != preview_map.shape:
            raise ValueError(
                f"DEM shape {dem.shape} does not match map {preview_map.shape}")
        slope = slope_degrees(dem, pixel_size_m)
        drivable = np.isfinite(cost_grid)
        # penalise slopes below the block threshold, block those above it
        penalty = 1.0 + slope_weight * np.clip(slope / slope_block_deg, 0, 1)
        cost_grid[drivable] *= penalty[drivable]
        cost_grid[(slope > slope_block_deg) & drivable] = np.inf

    return cost_grid


def astar_pixel_search(cost_grid, start, end):
    """A* over an 8-connected pixel grid; returns a list of (x, y) points."""
    h, w = cost_grid.shape
    start_cell = (start[1], start[0])  # (row, col)
    end_cell = (end[1], end[0])

    moves = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
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
                    heuristic = math.hypot(nb[0] - end_cell[0], nb[1] - end_cell[1])
                    heapq.heappush(queue, (new_cost + heuristic, nb))

    if end_cell not in cost_so_far:
        # unreachable (e.g. fully blocked by slope) -> straight fallback link
        return [start, end]

    path = []
    curr = end_cell
    while curr is not None:
        path.append((curr[1], curr[0]))
        curr = came_from[curr]
    path.reverse()
    return path


# =====================================================================
#  Coverage swath generation  (now driven along corridors, not over vines)
# =====================================================================

def _sweep_segments(mask, spacing_px, min_len=3):
    """Horizontal boustrophedon segments over the True pixels of a mask."""
    h, w = mask.shape
    lines = []
    stride = max(1, int(spacing_px))
    for r in range(0, h, stride):
        row = mask[r]
        in_seg = False
        start_c = 0
        for c in range(w):
            if row[c]:
                if not in_seg:
                    in_seg = True
                    start_c = c
            elif in_seg:
                if (c - start_c) > min_len:
                    lines.append(((start_c, r), (c - 1, r)))
                in_seg = False
        if in_seg and (w - start_c) > min_len:
            lines.append(((start_c, r), (w - 1, r)))
    return lines


def generate_zone_swaths(preview_map, suplement="water", spacing_px=4.0,
                         corridor_reach_px=8):
    """
    Generate coverage swaths for the mission.

    Unlike the original version (which swept straight across the vine blocks),
    the robot is kept on the bare-soil corridors (zone 0) *adjacent* to the
    zones that need treatment. This stops the path from running on top of the
    vineyard rows.

    Water    -> corridors next to any vine zone.
    Nitrogen -> corridors next to low/mid-vigor zones only (the highest-vigor
                zone is ignored, exactly as requested).

    If the map resolution is too coarse for the inter-row corridors to survive
    (a known low-resolution problem), the function falls back to sweeping the
    target zones directly so a mission still gets produced, and warns the user.

    Returns:
        list of ((x1, y1), (x2, y2)) segment endpoints.
    """
    if preview_map.size == 0:
        return []

    fm = field_mask(preview_map)
    walkable = fm & (preview_map == 0)
    target = target_mask(preview_map, suplement)
    if not np.any(target):
        return []

    # corridors within reach of a zone that needs treatment
    reach = binary_dilation(target, iterations=int(corridor_reach_px))
    useful = walkable & reach

    lines = _sweep_segments(useful, spacing_px, min_len=3)

    if len(lines) < 3:
        print("  [warn] drivable corridors too sparse at this resolution; "
              "falling back to in-block sweeps. Try --high-resolution.")
        lines = _sweep_segments(target, spacing_px, min_len=3)

    return lines


def order_swaths_snake(swaths):
    """
    Order fragmented swaths into a continuous route with a greedy
    nearest-neighbour chain, flipping each segment so the robot enters at its
    closer endpoint. This behaves like a boustrophedon "snake" for clean rows
    but also copes with the many short segments that corridors produce.

    Args:
        swaths: unordered list of ((x1, y1), (x2, y2)) segments.

    Returns:
        Ordered list of ((x1, y1), (x2, y2)) segments.
    """
    if not swaths:
        return []

    remaining = sorted(swaths, key=lambda s: (s[0][1], s[0][0]))
    ordered = [remaining.pop(0)]
    pos = ordered[0][1]

    while remaining:
        best_i, best_d, best_flip = 0, float("inf"), False
        for i, (p1, p2) in enumerate(remaining):
            d1 = (p1[0] - pos[0]) ** 2 + (p1[1] - pos[1]) ** 2
            d2 = (p2[0] - pos[0]) ** 2 + (p2[1] - pos[1]) ** 2
            if d1 < best_d:
                best_i, best_d, best_flip = i, d1, False
            if d2 < best_d:
                best_i, best_d, best_flip = i, d2, True
        p1, p2 = remaining.pop(best_i)
        if best_flip:
            p1, p2 = p2, p1
        ordered.append((p1, p2))
        pos = p2

    return ordered


def build_complete_mission_path(preview_map, swaths_ordered, suplement='water',
                                use_elevation=False, cost_grid=None):
    """
    Stitch the ordered swaths into one trajectory. Between consecutive swaths
    the robot either takes a straight connector (simple mode) or an A* route
    through the drivable, slope-aware cost grid (elevation mode).
    """
    full_trajectory = []
    if use_elevation and cost_grid is None:
        cost_grid = create_corridor_cost_grid(preview_map, suplement)

    for i in range(len(swaths_ordered)):
        p1, p2 = swaths_ordered[i]
        full_trajectory.append((p1, p2))

        if i < len(swaths_ordered) - 1:
            next_p1, _ = swaths_ordered[i + 1]
            if use_elevation:
                transition = astar_pixel_search(cost_grid, p2, next_p1)
                for j in range(len(transition) - 1):
                    full_trajectory.append((transition[j], transition[j + 1]))
            else:
                full_trajectory.append((p2, next_p1))
    return full_trajectory


def generate_autonomous_mission(preview_map, suplement, spacing_px,
                                use_elevation, dem=None, pixel_size_m=1.0):
    """
    Full mission planner.

    Simple mode (use_elevation=False): corridor swaths joined by straight
    connectors.

    Elevation mode (use_elevation=True): the same swaths, but transitions are
    routed with A* over a DEM-aware cost grid so steep slopes are avoided.

    (The previous auto-rotation step relied on a row-angle detector that did
    not work reliably on this low-resolution data, so it has been removed in
    favour of this simpler, robust pipeline.)
    """
    raw_swaths = generate_zone_swaths(preview_map, suplement=suplement,
                                      spacing_px=spacing_px)
    if not raw_swaths:
        return []

    ordered_swaths = order_swaths_snake(raw_swaths)

    cost_grid = None
    if use_elevation:
        cost_grid = create_corridor_cost_grid(
            preview_map, suplement, dem=dem, pixel_size_m=pixel_size_m)

    return build_complete_mission_path(
        preview_map, ordered_swaths, suplement=suplement,
        use_elevation=use_elevation, cost_grid=cost_grid)


# =====================================================================
#  Rendering
# =====================================================================

def plot_and_save_robot_mission(preview_map, cmap_zones, norm, trajectory,
                                filename_output, suplement):
    """Render the classified map with the robot route overlaid and save a PNG."""
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.imshow(preview_map, cmap=cmap_zones, norm=norm, interpolation="nearest")

    for p1, p2 in trajectory:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color="#00FFFF", linewidth=2.0, alpha=0.9, zorder=50)

    plt.title(f"Robotic path for: {suplement.upper()}", fontsize=14, color="white")
    ax.set_xlim(0, preview_map.shape[1])
    ax.set_ylim(preview_map.shape[0], 0)
    plt.axis("off")

    plt.savefig(filename_output, dpi=300, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()
    print(f"Path saved in: {filename_output}")
