from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize, shapes
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Point, shape
from shapely.ops import unary_union


DEFAULT_METRIC_CRS = "EPSG:32629"
MIN_SWATH_LENGTH_M = 1.0
WAYPOINT_SPACING_M = 0.5
COVERAGE_RESOLUTION_M = 0.5
ROBOT_WIDTH_M = 0.6
VEHICLE_RADIUS_M = 0.4
PLOT_MAX_ARROWS_PER_PATCH = 14


def clean_geometry(geometry):
    """Repair minor geometry defects while preserving the original geometry when possible."""
    if geometry is None or geometry.is_empty:
        return geometry
    repaired = geometry.buffer(0)
    return repaired if repaired is not None and not repaired.is_empty else geometry


def polygon_parts(geometry) -> list:
    """Return all polygon parts from Polygon/MultiPolygon/GeometryCollection input."""
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    parts = []
    for part in getattr(geometry, "geoms", []):
        parts.extend(polygon_parts(part))
    return parts


def line_parts(geometry) -> list[LineString]:
    """Return all non-empty LineString parts from a line-like geometry."""
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry] if geometry.length > 0 else []
    if isinstance(geometry, MultiLineString):
        return [part for part in geometry.geoms if part.length > 0]
    parts = []
    for part in getattr(geometry, "geoms", []):
        parts.extend(line_parts(part))
    return parts


def side_label(side_id: int) -> str:
    return "southwest" if side_id == 1 else "northeast"


def make_patch_name(side_id: int, side_name: str, patch_id: int) -> str:
    return f"mission_{side_id:02d}_{side_name}_p80_patch_{patch_id:02d}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


class HardCorridorGrid:
    """Small A* grid used only for within-patch headland connectors."""

    def __init__(
        self,
        free_geometry,
        crs,
        risk_path: Path | None,
        resolution_m: float = 0.5,
    ):
        self.resolution_m = float(resolution_m)
        self.crs = crs
        xmin, ymin, xmax, ymax = free_geometry.bounds
        xmin = math.floor(xmin / resolution_m) * resolution_m
        ymin = math.floor(ymin / resolution_m) * resolution_m
        xmax = math.ceil(xmax / resolution_m) * resolution_m
        ymax = math.ceil(ymax / resolution_m) * resolution_m
        self.width = max(1, int(round((xmax - xmin) / resolution_m)))
        self.height = max(1, int(round((ymax - ymin) / resolution_m)))
        self.transform = from_origin(xmin, ymax, resolution_m, resolution_m)

        self.free_mask = geometry_mask(
            [part.__geo_interface__ for part in polygon_parts(free_geometry)],
            out_shape=(self.height, self.width),
            transform=self.transform,
            invert=True,
        )
        self.cost = np.ones((self.height, self.width), dtype="float32")

        if risk_path is not None and risk_path.exists():
            risk = np.ones((self.height, self.width), dtype="float32")
            with rasterio.open(risk_path) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=risk,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=self.transform,
                    dst_crs=crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.average,
                )
            risk = np.where(np.isfinite(risk), np.clip(risk, 0.0, 1.0), 1.0)
            self.cost = 1.0 + 5.0 * risk

        self.cost[~self.free_mask] = np.inf

    def point_to_cell(self, point: Point) -> tuple[int, int]:
        col = int(math.floor((point.x - self.transform.c) / self.resolution_m))
        row = int(math.floor((self.transform.f - point.y) / self.resolution_m))
        return row, col

    def cell_to_point(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        x = self.transform.c + (col + 0.5) * self.resolution_m
        y = self.transform.f - (row + 0.5) * self.resolution_m
        return float(x), float(y)

    def nearest_free(self, cell: tuple[int, int], radius: int = 8) -> tuple[int, int] | None:
        row, col = cell
        candidates = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = row + dr, col + dc
                if rr < 0 or cc < 0 or rr >= self.height or cc >= self.width:
                    continue
                if np.isfinite(self.cost[rr, cc]):
                    candidates.append((dr * dr + dc * dc, rr, cc))
        return (min(candidates)[1], min(candidates)[2]) if candidates else None

    def search(self, start: Point, end: Point, margin_m: float = 45.0) -> LineString | None:
        start_cell = self.nearest_free(self.point_to_cell(start))
        end_cell = self.nearest_free(self.point_to_cell(end))
        if start_cell is None or end_cell is None:
            return None

        distance = start.distance(end)
        margin = int(math.ceil(max(margin_m, distance * 0.35) / self.resolution_m))
        r0 = max(0, min(start_cell[0], end_cell[0]) - margin)
        r1 = min(self.height - 1, max(start_cell[0], end_cell[0]) + margin)
        c0 = max(0, min(start_cell[1], end_cell[1]) - margin)
        c1 = min(self.width - 1, max(start_cell[1], end_cell[1]) + margin)

        moves = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
        queue = [(0.0, start_cell)]
        cost_so_far = {start_cell: 0.0}
        came_from = {}

        while queue:
            _, current = heapq.heappop(queue)
            if current == end_cell:
                break
            for dr, dc, step in moves:
                nb = (current[0] + dr, current[1] + dc)
                if nb[0] < r0 or nb[0] > r1 or nb[1] < c0 or nb[1] > c1:
                    continue
                cell_cost = self.cost[nb]
                if not np.isfinite(cell_cost):
                    continue
                new_cost = cost_so_far[current] + step * float(cell_cost)
                if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                    cost_so_far[nb] = new_cost
                    came_from[nb] = current
                    heuristic = math.hypot(nb[0] - end_cell[0], nb[1] - end_cell[1])
                    heapq.heappush(queue, (new_cost + heuristic, nb))

        if end_cell not in cost_so_far:
            return None

        cells = [end_cell]
        while cells[-1] != start_cell:
            cells.append(came_from[cells[-1]])
        cells.reverse()
        coords = [tuple(start.coords[0])] + [self.cell_to_point(cell) for cell in cells[1:-1]] + [tuple(end.coords[0])]
        return LineString(coords).simplify(self.resolution_m * 0.25)


def raster_target_coverage(
    route: gpd.GeoDataFrame,
    target_geometry,
    sensor_radius_m: float,
    resolution_m: float = COVERAGE_RESOLUTION_M,
) -> tuple[float, int, int]:
    """Estimate target coverage from the buffered route footprint."""
    if route.empty or target_geometry is None or target_geometry.is_empty:
        return 0.0, 0, 0

    minx, miny, maxx, maxy = target_geometry.bounds
    minx -= sensor_radius_m + 1.0
    miny -= sensor_radius_m + 1.0
    maxx += sensor_radius_m + 1.0
    maxy += sensor_radius_m + 1.0
    width = max(1, int(math.ceil((maxx - minx) / resolution_m)))
    height = max(1, int(math.ceil((maxy - miny) / resolution_m)))
    transform = from_origin(minx, maxy, resolution_m, resolution_m)

    target_mask = rasterize(
        [(target_geometry, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    if not target_mask.any():
        return 0.0, 0, 0

    buffered = ((geom.buffer(sensor_radius_m, cap_style=2), 1) for geom in route.geometry)
    covered_mask = rasterize(
        buffered,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    covered_target = covered_mask & target_mask
    return float(covered_target.sum() / target_mask.sum()), int(covered_target.sum()), int(target_mask.sum())


def build_sidewise_p80_targets(
    safe_parts: list,
    safe_crs,
    priority_path: Path,
    side_percentile: float,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Compute P80 independently inside each road-separated side."""
    rows = []
    stats = []
    with rasterio.open(priority_path) as src:
        priority = src.read(1).astype("float64")
        valid = np.isfinite(priority)
        if src.nodata is not None:
            valid &= priority != src.nodata

        for side_id, side_polygon in enumerate(safe_parts[:2], start=1):
            side_name = side_label(side_id)
            side_raster = gpd.GeoSeries([side_polygon], crs=safe_crs).to_crs(src.crs).iloc[0]
            side_mask = geometry_mask(
                [side_raster.__geo_interface__],
                out_shape=priority.shape,
                transform=src.transform,
                invert=True,
            )
            side_valid = side_mask & valid
            side_values = priority[side_valid]
            if side_values.size == 0:
                raise RuntimeError(f"No priority raster cells found for side {side_id} ({side_name}).")

            threshold = float(np.nanpercentile(side_values, side_percentile))
            selected = side_valid & (priority >= threshold)
            selected_count = int(selected.sum())
            if selected_count == 0:
                raise RuntimeError(f"No side-wise P80 cells selected for side {side_id} ({side_name}).")

            polygons = []
            for geom, value in shapes(selected.astype("uint8"), mask=selected, transform=src.transform):
                if int(value) != 1:
                    continue
                geom_shape = shape(geom)
                clipped = clean_geometry(geom_shape.intersection(side_raster))
                if clipped is not None and not clipped.is_empty:
                    polygons.append(clipped)

            if not polygons:
                raise RuntimeError(f"Selected cells did not produce polygons for {side_name}.")

            side_union_raster = clean_geometry(unary_union(polygons))
            side_union_metric = gpd.GeoSeries([side_union_raster], crs=src.crs).to_crs(safe_crs).iloc[0]
            patch_geometries = sorted(polygon_parts(side_union_metric), key=lambda p: p.area, reverse=True)
            for patch_id, patch in enumerate(patch_geometries, start=1):
                if patch.area <= 0:
                    continue
                rows.append(
                    {
                        "side_id": side_id,
                        "side_name": side_name,
                        "side_percentile": side_percentile,
                        "side_threshold": threshold,
                        "patch_id": patch_id,
                        "patch_area_m2": float(patch.area),
                        "geometry": patch,
                    }
                )

            stats.append(
                {
                    "side_id": side_id,
                    "side_name": side_name,
                    "side_valid_cell_count": int(side_values.size),
                    "selected_cell_count": selected_count,
                    "selected_cell_fraction": float(selected_count / side_values.size),
                    "side_threshold": threshold,
                    "side_priority_min": float(np.nanmin(side_values)),
                    "side_priority_mean": float(np.nanmean(side_values)),
                    "side_priority_median": float(np.nanmedian(side_values)),
                    "side_priority_p80": threshold,
                    "side_priority_max": float(np.nanmax(side_values)),
                    "sidewise_p80_area_m2": float(side_union_metric.area),
                    "sidewise_p80_patch_count": int(len(patch_geometries)),
                    "safe_side_area_m2": float(side_polygon.area),
                    "sidewise_p80_fraction_of_side_area": float(side_union_metric.area / max(side_polygon.area, 1e-9)),
                }
            )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=safe_crs), pd.DataFrame(stats)


def prepare_p80_patches(safe_parts: list, target_geometry) -> list[dict[str, Any]]:
    """Split side-wise P80 into patch sub-missions."""
    patches = []
    for side_id, side_polygon in enumerate(safe_parts[:2], start=1):
        current_side_label = side_label(side_id)
        side_target = clean_geometry(target_geometry.intersection(side_polygon))
        side_patches = sorted(polygon_parts(side_target), key=lambda p: p.area, reverse=True)
        for patch_id, patch in enumerate(side_patches, start=1):
            if patch.area <= 0:
                continue
            patches.append(
                {
                    "side_id": side_id,
                    "side_name": current_side_label,
                    "patch_id": patch_id,
                    "mission_name": make_patch_name(side_id, current_side_label, patch_id),
                    "side_polygon": side_polygon,
                    "patch_geometry": clean_geometry(patch),
                }
            )
    return patches


def generate_patch_swaths(patch_geometry, angle_deg: float, spacing_m: float, crs) -> gpd.GeoDataFrame:
    """Generate row-aligned swaths clipped to one P80 patch."""
    origin = patch_geometry.centroid
    rotated = affinity.rotate(patch_geometry, -angle_deg, origin=origin)
    minx, miny, maxx, maxy = rotated.bounds
    start_y = math.floor(miny / spacing_m) * spacing_m

    rows = []
    segment_id = 1
    row_index = 0
    y = start_y
    while y <= maxy:
        base = LineString([(minx - 20.0, y), (maxx + 20.0, y)])
        clipped = base.intersection(rotated)
        pieces = sorted(line_parts(clipped), key=lambda part: part.centroid.x)
        for piece_index, piece in enumerate(pieces):
            if piece.length < MIN_SWATH_LENGTH_M:
                continue
            line = affinity.rotate(piece, angle_deg, origin=origin)
            rotated_line = affinity.rotate(line, -angle_deg, origin=origin)
            rows.append(
                {
                    "segment_id": segment_id,
                    "row_index": row_index,
                    "piece_index": piece_index,
                    "row_y_rotated_m": float(y),
                    "x_order_rotated_m": float(rotated_line.centroid.x),
                    "length_m": float(line.length),
                    "geometry": line,
                }
            )
            segment_id += 1
        y += spacing_m
        row_index += 1

    columns = [
        "segment_id",
        "row_index",
        "piece_index",
        "row_y_rotated_m",
        "x_order_rotated_m",
        "length_m",
        "geometry",
    ]
    if not rows:
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def order_swaths_snake(swaths: gpd.GeoDataFrame, angle_deg: float) -> list[dict[str, Any]]:
    """Order swaths in a boustrophedon snake sequence."""
    if swaths.empty:
        return []
    origin = unary_union(swaths.geometry).centroid
    ordered = []
    for order_i, (_, group) in enumerate(swaths.groupby("row_index", sort=True)):
        row = group.sort_values("x_order_rotated_m", ascending=(order_i % 2 == 0))
        for _, record in row.iterrows():
            geom = record.geometry
            rotated = affinity.rotate(geom, -angle_deg, origin=origin)
            coords = list(geom.coords)
            if order_i % 2 == 0 and rotated.coords[0][0] > rotated.coords[-1][0]:
                coords.reverse()
            elif order_i % 2 == 1 and rotated.coords[0][0] < rotated.coords[-1][0]:
                coords.reverse()
            ordered.append({**record.to_dict(), "geometry": LineString(coords)})
    return ordered


def direct_or_grid_connector(
    start: Point,
    end: Point,
    patch_geometry,
    grid: HardCorridorGrid,
    tolerance_m: float,
) -> LineString | None:
    """Return a connector only if it remains inside the P80 patch."""
    direct = LineString([start, end])
    if patch_geometry.buffer(tolerance_m).covers(direct):
        return direct
    connector = grid.search(start, end, margin_m=45.0)
    if connector is not None and patch_geometry.buffer(tolerance_m).covers(connector):
        return connector
    return None


def build_patch_route(
    patch_record: dict[str, Any],
    swaths_ordered: list[dict[str, Any]],
    crs,
    risk_path: Path | None,
    connector_tolerance_m: float,
) -> tuple[gpd.GeoDataFrame, int]:
    """Build a patch-level route; failed connectors are kept as disconnected transitions."""
    patch_geometry = patch_record["patch_geometry"]
    grid = HardCorridorGrid(patch_geometry, crs, risk_path=risk_path, resolution_m=0.5)
    rows = []
    mission_order = 1
    failed_connectors = 0
    previous_end: Point | None = None

    for item in swaths_ordered:
        line = item["geometry"]
        start = Point(line.coords[0])
        if previous_end is not None:
            connector = direct_or_grid_connector(start=previous_end, end=start, patch_geometry=patch_geometry, grid=grid, tolerance_m=connector_tolerance_m)
            if connector is None:
                failed_connectors += 1
            else:
                rows.append(
                    {
                        "mission_id": int(patch_record["side_id"]),
                        "mission_name": patch_record["mission_name"],
                        "side_name": patch_record["side_name"],
                        "patch_id": int(patch_record["patch_id"]),
                        "segment_type": "headland_connector",
                        "mission_order": mission_order,
                        "row_index": np.nan,
                        "segment_id": np.nan,
                        "length_m": float(connector.length),
                        "geometry": connector,
                    }
                )
                mission_order += 1

        rows.append(
            {
                "mission_id": int(patch_record["side_id"]),
                "mission_name": patch_record["mission_name"],
                "side_name": patch_record["side_name"],
                "patch_id": int(patch_record["patch_id"]),
                "segment_type": "coverage_swath",
                "mission_order": mission_order,
                "row_index": int(item["row_index"]),
                "segment_id": int(item["segment_id"]),
                "length_m": float(line.length),
                "geometry": line,
            }
        )
        mission_order += 1
        previous_end = Point(line.coords[-1])

    columns = [
        "mission_id",
        "mission_name",
        "side_name",
        "patch_id",
        "segment_type",
        "mission_order",
        "row_index",
        "segment_id",
        "length_m",
        "geometry",
    ]
    if not rows:
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs), failed_connectors
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs), failed_connectors


def plan_patch(
    patch_record: dict[str, Any],
    angle_deg: float,
    spacing_m: float,
    crs,
    risk_path: Path | None,
    connector_tolerance_m: float,
) -> dict[str, Any]:
    swaths = generate_patch_swaths(patch_record["patch_geometry"], angle_deg, spacing_m, crs)
    if not swaths.empty:
        swaths.insert(0, "mission_id", int(patch_record["side_id"]))
        swaths.insert(1, "mission_name", patch_record["mission_name"])
        swaths.insert(2, "side_name", patch_record["side_name"])
        swaths.insert(3, "patch_id", int(patch_record["patch_id"]))
    ordered = order_swaths_snake(swaths, angle_deg)
    route, failed_connectors = build_patch_route(patch_record, ordered, crs, risk_path, connector_tolerance_m)
    return {
        **patch_record,
        "swaths": swaths,
        "route": route,
        "failed_connectors": int(failed_connectors),
    }


def make_waypoints(route: gpd.GeoDataFrame, spacing_m: float = WAYPOINT_SPACING_M) -> pd.DataFrame:
    rows = []
    waypoint_id = 1
    for _, record in route.sort_values(["mission_id", "patch_id", "mission_order"]).iterrows():
        geom = record.geometry
        n = max(1, int(math.ceil(geom.length / spacing_m)))
        points = [geom.interpolate(i / n, normalized=True) for i in range(n + 1)]
        for i, point in enumerate(points):
            if i < len(points) - 1:
                next_point = points[i + 1]
                heading = math.degrees(math.atan2(next_point.y - point.y, next_point.x - point.x))
            elif i > 0:
                previous_point = points[i - 1]
                heading = math.degrees(math.atan2(point.y - previous_point.y, point.x - previous_point.x))
            else:
                heading = float("nan")
            rows.append(
                {
                    "waypoint_id": waypoint_id,
                    "mission_id": int(record.mission_id),
                    "mission_name": record.mission_name,
                    "side_name": record.side_name,
                    "patch_id": int(record.patch_id),
                    "mission_order": int(record.mission_order),
                    "segment_type": record.segment_type,
                    "x": float(point.x),
                    "y": float(point.y),
                    "heading_deg": float(heading),
                    "speed_mps": 0.5,
                }
            )
            waypoint_id += 1
    return pd.DataFrame(rows)


def route_union(route: gpd.GeoDataFrame):
    return unary_union(route.geometry) if not route.empty else LineString()


def route_outside_length(route: gpd.GeoDataFrame, geometry) -> float:
    if route.empty:
        return 0.0
    return float(route_union(route).difference(geometry).length)


def patch_metrics(
    planned: dict[str, Any],
    target_all,
    safe_all,
    vegetation_all,
    sensor_radius_m: float,
) -> dict[str, Any]:
    route = planned["route"]
    swaths = planned["swaths"]
    patch = planned["patch_geometry"]
    coverage_ratio, covered_cells, total_cells = raster_target_coverage(route, patch, sensor_radius_m=sensor_radius_m)
    union = route_union(route)
    vehicle_buffer = union.buffer(VEHICLE_RADIUS_M, cap_style=2) if not route.empty else LineString().buffer(0)
    crop_crossing_area = float(vehicle_buffer.intersection(vegetation_all).area)
    vehicle_buffer_area = float(vehicle_buffer.area)
    return {
        "side_id": int(planned["side_id"]),
        "side_name": planned["side_name"],
        "patch_id": int(planned["patch_id"]),
        "mission_name": planned["mission_name"],
        "p80_patch_area_m2": float(patch.area),
        "swath_count": int(len(swaths)),
        "route_part_count": int(len(route)),
        "failed_connector_count": int(planned["failed_connectors"]),
        "path_length_m": float(route.length.sum()) if not route.empty else 0.0,
        "swath_length_m": float(swaths.length.sum()) if not swaths.empty else 0.0,
        "connector_length_m": float(route[route["segment_type"].eq("headland_connector")].length.sum()) if not route.empty else 0.0,
        "p80_patch_coverage_ratio": float(coverage_ratio),
        "p80_patch_covered_cells": int(covered_cells),
        "p80_patch_total_cells": int(total_cells),
        "outside_p80_patch_length_m": route_outside_length(route, patch),
        "outside_all_p80_length_m": route_outside_length(route, target_all),
        "outside_safe_area_length_m": route_outside_length(route, safe_all),
        "vehicle_crop_crossing_area_m2": crop_crossing_area,
        "vehicle_buffer_area_m2": vehicle_buffer_area,
        "vehicle_crop_crossing_area_ratio": float(crop_crossing_area / max(vehicle_buffer_area, 1e-9)),
        "centerline_vegetation_intersection_length_m": float(union.intersection(vegetation_all).length) if not route.empty else 0.0,
    }


def side_metrics(
    patch_metric_rows: list[dict[str, Any]],
    combined_route: gpd.GeoDataFrame,
    target_all,
    safe_parts: list,
    vegetation_all,
    sensor_radius_m: float,
) -> list[dict[str, Any]]:
    df = pd.DataFrame(patch_metric_rows)
    rows = []
    for side_id, group in df.groupby("side_id", sort=True):
        side_route = combined_route[combined_route["mission_id"].eq(int(side_id))]
        side_union = route_union(side_route)
        side_target = clean_geometry(target_all.intersection(safe_parts[int(side_id) - 1]))
        side_safe = safe_parts[int(side_id) - 1]
        side_vehicle_buffer = side_union.buffer(VEHICLE_RADIUS_M, cap_style=2) if not side_route.empty else LineString().buffer(0)
        coverage, _, _ = raster_target_coverage(side_route, side_target, sensor_radius_m=sensor_radius_m)
        rows.append(
            {
                "side_id": int(side_id),
                "side_name": group.iloc[0]["side_name"],
                "p80_patch_count": int(len(group)),
                "side_p80_area_m2": float(group["p80_patch_area_m2"].sum()),
                "swath_count": int(group["swath_count"].sum()),
                "route_part_count": int(len(side_route)),
                "failed_connector_count": int(group["failed_connector_count"].sum()),
                "path_length_m": float(side_route.length.sum()) if not side_route.empty else 0.0,
                "swath_length_m": float(group["swath_length_m"].sum()),
                "connector_length_m": float(side_route[side_route["segment_type"].eq("headland_connector")].length.sum()) if not side_route.empty else 0.0,
                "side_p80_coverage_ratio": float(coverage),
                "outside_p80_length_m": float(side_union.difference(side_target).length) if not side_route.empty else 0.0,
                "outside_safe_area_length_m": float(side_union.difference(side_safe).length) if not side_route.empty else 0.0,
                "vehicle_crop_crossing_area_m2": float(side_vehicle_buffer.intersection(vegetation_all).area),
                "centerline_vegetation_intersection_length_m": float(side_union.intersection(vegetation_all).length) if not side_route.empty else 0.0,
            }
        )
    return rows


def plot_result(
    safe: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    vegetation: gpd.GeoDataFrame,
    planned_patches: list[dict[str, Any]],
    metrics: dict[str, Any],
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.4, 9.4))
    ax.set_facecolor("#F8FAFC")
    safe.plot(ax=ax, facecolor="white", edgecolor="#0F2747", linewidth=2.1, zorder=1)
    vegetation.plot(ax=ax, facecolor="#B7E4C7", edgecolor="none", alpha=0.24, zorder=2)
    target.plot(ax=ax, facecolor="#FDBA74", edgecolor="#9A3412", linewidth=1.25, alpha=0.56, zorder=3)

    palettes = {
        1: {"swath": "#0284C7", "connector": "#7C3AED", "arrow": "#075985"},
        2: {"swath": "#DC2626", "connector": "#9333EA", "arrow": "#991B1B"},
    }

    for planned in planned_patches:
        route = planned["route"]
        if route.empty:
            continue
        side_id = int(planned["side_id"])
        palette = palettes[side_id]
        swath_route = route[route["segment_type"].eq("coverage_swath")]
        connector_route = route[route["segment_type"].eq("headland_connector")]
        if not connector_route.empty:
            connector_route.plot(ax=ax, color=palette["connector"], linewidth=1.05, alpha=0.70, zorder=4)
        if not swath_route.empty:
            swath_route.plot(ax=ax, color=palette["swath"], linewidth=1.55, alpha=0.96, zorder=5)

        ordered = route.sort_values("mission_order")
        start = Point(ordered.iloc[0].geometry.coords[0])
        end = Point(ordered.iloc[-1].geometry.coords[-1])
        ax.scatter([start.x], [start.y], s=62, c=palette["swath"], edgecolors="black", linewidths=0.7, zorder=8)
        ax.scatter([end.x], [end.y], s=42, c="white", edgecolors=palette["swath"], linewidths=1.6, zorder=8)

        waypoints = make_waypoints(route, spacing_m=2.8)
        if not waypoints.empty:
            step = max(1, math.ceil(len(waypoints) / PLOT_MAX_ARROWS_PER_PATCH))
            sampled = waypoints.iloc[::step]
            dx = np.cos(np.deg2rad(sampled["heading_deg"].to_numpy(dtype=float)))
            dy = np.sin(np.deg2rad(sampled["heading_deg"].to_numpy(dtype=float)))
            ax.quiver(
                sampled["x"],
                sampled["y"],
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=0.55,
                width=0.0025,
                color=palette["arrow"],
                alpha=0.58,
                zorder=6,
            )

    side_lines = []
    for side in metrics["side_metrics"]:
        side_lines.append(
            f"{side['side_name']}: {side['p80_patch_count']} patch(es), "
            f"{side['swath_count']} swaths, P80 cov {100 * side['side_p80_coverage_ratio']:.1f}%"
        )
    text = (
        "Two-side side-wise P80 CPP\n"
        "Routes are generated only inside side-clipped P80 demand patches.\n"
        + "\n".join(side_lines)
        + "\n"
        f"Total swaths: {metrics['combined_swath_count']} | path length: {metrics['combined_path_length_m']:.1f} m\n"
        f"Global P80 coverage: {100 * metrics['global_p80_coverage_ratio']:.2f}%\n"
        f"Outside P80 length: {metrics['outside_p80_length_m']:.3f} m; disconnected transitions: {metrics['combined_failed_connector_count']}\n"
        f"Status: {metrics['feasibility_label']}"
    )
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=9.8,
        family="DejaVu Sans Mono",
        color="#111827",
        bbox=dict(facecolor="white", edgecolor="#CBD5E1", boxstyle="round,pad=0.55", alpha=0.96),
        zorder=20,
    )
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#0F2747", lw=2, label="road-separated safe_area boundary"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#FDBA74", edgecolor="#9A3412", alpha=0.56, label="side-wise P80 demand regions"),
            plt.Line2D([0], [0], color="#0284C7", lw=2, label="southwest P80 CPP"),
            plt.Line2D([0], [0], color="#DC2626", lw=2, label="northeast P80 CPP"),
            plt.Line2D([0], [0], color="#7C3AED", lw=1.6, label="within-patch headland connectors"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#B7E4C7", alpha=0.24, label="vegetation diagnostic"),
        ],
        loc="upper right",
        framealpha=0.95,
        fontsize=8.9,
    )
    ax.set_title("Two-Side CPP Using Side-Specific P80 Demand Regions", fontsize=17.0, weight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.grid(color="#E2E8F0", linewidth=0.55, alpha=0.50)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=270)
    plt.close(fig)


def run(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    outputs = config["outputs"]
    parameters = config["parameters"]

    safe_path = resolve_path(project_root, inputs["safe_area"])
    priority_path = resolve_path(project_root, inputs["priority_layer"])
    vegetation_path = resolve_path(project_root, inputs["vegetation_body"])
    row_orientation_path = resolve_path(project_root, inputs["row_orientation"])
    risk_path = resolve_path(project_root, inputs["risk_map"])

    mission_dir = resolve_path(project_root, outputs["mission_dir"])
    evaluation_dir = resolve_path(project_root, outputs["evaluation_dir"])
    figure_dir = resolve_path(project_root, outputs["figure_dir"])
    sidewise_p80_path = resolve_path(project_root, outputs["sidewise_p80_geojson"])
    sidewise_stats_path = resolve_path(project_root, outputs["sidewise_p80_stats_csv"])
    figure_path = figure_dir / "fig_two_side_sidewise_p80_cpp.png"

    side_percentile = float(parameters["side_percentile"])
    working_width_m = float(parameters["working_width_m"])
    sensor_radius_m = working_width_m / 2.0
    connector_tolerance_m = float(parameters["demand_connector_tolerance_m"])

    mission_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    row_info = read_json(row_orientation_path)
    row_angle = float(row_info["row_angle_deg_map"])
    safe = gpd.read_file(safe_path).to_crs(DEFAULT_METRIC_CRS)
    vegetation = gpd.read_file(vegetation_path).to_crs(safe.crs)
    safe_union = clean_geometry(unary_union(safe.geometry))
    vegetation_union = clean_geometry(unary_union(vegetation.geometry))
    safe_parts = sorted(polygon_parts(safe_union), key=lambda p: p.centroid.y)

    sidewise_target, sidewise_stats = build_sidewise_p80_targets(
        safe_parts=safe_parts,
        safe_crs=safe.crs,
        priority_path=priority_path,
        side_percentile=side_percentile,
    )
    sidewise_target.to_file(sidewise_p80_path, driver="GeoJSON")
    sidewise_stats.to_csv(sidewise_stats_path, index=False)
    target_union = clean_geometry(unary_union(sidewise_target.geometry))

    patch_records = prepare_p80_patches(safe_parts, target_union)
    if not patch_records:
        raise RuntimeError("No side-wise P80 patches were generated.")

    planned = [
        plan_patch(
            patch_record=record,
            angle_deg=row_angle,
            spacing_m=working_width_m,
            crs=safe.crs,
            risk_path=risk_path,
            connector_tolerance_m=connector_tolerance_m,
        )
        for record in patch_records
    ]
    routes = [p["route"] for p in planned if not p["route"].empty]
    swaths = [p["swaths"] for p in planned if not p["swaths"].empty]
    if not routes:
        raise RuntimeError("No side-wise P80 CPP routes were generated.")

    combined_route = gpd.GeoDataFrame(pd.concat(routes, ignore_index=True), geometry="geometry", crs=safe.crs)
    combined_swaths = gpd.GeoDataFrame(pd.concat(swaths, ignore_index=True), geometry="geometry", crs=safe.crs)
    waypoints = make_waypoints(combined_route)

    for p in planned:
        name = p["mission_name"]
        if not p["route"].empty:
            p["route"].to_file(mission_dir / f"{name}_route.geojson", driver="GeoJSON")
        if not p["swaths"].empty:
            p["swaths"].to_file(mission_dir / f"{name}_swaths.geojson", driver="GeoJSON")

    combined_route.to_file(mission_dir / "mission_routes.geojson", driver="GeoJSON")
    combined_swaths.to_file(mission_dir / "mission_swaths.geojson", driver="GeoJSON")
    waypoints.to_csv(mission_dir / "mission_waypoints.csv", index=False)

    patch_metric_rows = [
        patch_metrics(
            planned=p,
            target_all=target_union,
            safe_all=safe_union,
            vegetation_all=vegetation_union,
            sensor_radius_m=sensor_radius_m,
        )
        for p in planned
    ]
    side_metric_rows = side_metrics(
        patch_metric_rows=patch_metric_rows,
        combined_route=combined_route,
        target_all=target_union,
        safe_parts=safe_parts,
        vegetation_all=vegetation_union,
        sensor_radius_m=sensor_radius_m,
    )
    pd.DataFrame(patch_metric_rows).to_csv(mission_dir / "patch_metrics.csv", index=False)
    pd.DataFrame(side_metric_rows).to_csv(mission_dir / "side_metrics.csv", index=False)

    global_p80_coverage, global_p80_covered, global_p80_total = raster_target_coverage(
        combined_route,
        target_union,
        sensor_radius_m=sensor_radius_m,
    )
    combined_union = route_union(combined_route)
    vehicle_buffer = combined_union.buffer(VEHICLE_RADIUS_M, cap_style=2)
    failed_connectors = int(sum(p["failed_connectors"] for p in planned))
    outside_p80_length = float(combined_union.difference(target_union).length)
    outside_safe_length = float(combined_union.difference(safe_union).length)
    vehicle_crop_crossing_area = float(vehicle_buffer.intersection(vegetation_union).area)

    metrics = {
        "mission_model": "two_side_sidewise_p80_row_aligned_cpp",
        "interpretation": (
            "P80 is computed independently within each road-separated safe-area side. "
            "CPP routes are then generated only inside the side-wise P80 demand patches. "
            "Disconnected transitions are retained when a connector would leave a P80 demand region."
        ),
        "priority_source": str(priority_path),
        "side_percentile": side_percentile,
        "sidewise_p80_geojson": str(sidewise_p80_path),
        "sidewise_threshold_stats_csv": str(sidewise_stats_path),
        "selected_angle_deg": row_angle,
        "row_angle_source": row_info,
        "swath_spacing_m": working_width_m,
        "working_width_m": working_width_m,
        "sensor_radius_m": sensor_radius_m,
        "demand_connector_tolerance_m": connector_tolerance_m,
        "side_count": 2,
        "p80_patch_count": int(len(planned)),
        "combined_swath_count": int(len(combined_swaths)),
        "combined_route_part_count": int(len(combined_route)),
        "combined_failed_connector_count": failed_connectors,
        "combined_path_length_m": float(combined_route.length.sum()),
        "combined_swath_length_m": float(combined_swaths.length.sum()),
        "combined_connector_length_m": float(
            combined_route[combined_route["segment_type"].eq("headland_connector")].length.sum()
        ),
        "global_p80_coverage_ratio": float(global_p80_coverage),
        "global_p80_covered_cells": int(global_p80_covered),
        "global_p80_total_cells": int(global_p80_total),
        "outside_p80_length_m": outside_p80_length,
        "outside_safe_area_length_m": outside_safe_length,
        "waypoint_count": int(len(waypoints)),
        "vehicle_crop_crossing_area_m2": vehicle_crop_crossing_area,
        "vehicle_buffer_area_m2": float(vehicle_buffer.area),
        "vehicle_crop_crossing_area_ratio": float(vehicle_crop_crossing_area / max(vehicle_buffer.area, 1e-9)),
        "centerline_vegetation_intersection_length_m": float(combined_union.intersection(vegetation_union).length),
        "sidewise_threshold_stats": sidewise_stats.to_dict("records"),
        "patch_metrics": patch_metric_rows,
        "side_metrics": side_metric_rows,
        "figure": str(figure_path),
        "feasibility_label": (
            "SIDEWISE_P80_CPP_WITHIN_DEMAND_REGIONS"
            if failed_connectors == 0 and outside_p80_length <= 0.5
            else "SIDEWISE_P80_CPP_WITH_MINOR_PATCH_CONNECTOR_WARNINGS"
        ),
    }

    plot_result(safe, sidewise_target, vegetation, planned, metrics, figure_path)
    (mission_dir / "targeted_cpp_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
