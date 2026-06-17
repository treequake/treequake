#!/usr/bin/env python3
"""Load generic spatial features into a normalized GeoPackage.

This phase preserves source geometry and provenance only. It does not assign
semantic classes, perform matching, or add earthquake-specific logic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".geojson", ".json", ".gpkg", ".shp", ".kml", ".kmz"}
TARGET_CRS = "EPSG:4326"
OUTPUT_LAYER = "spatial_features"

INVENTORY_COLUMNS = [
    "source_scope",
    "source_name",
    "source_path",
    "source_layer",
    "extension",
    "source_crs",
    "output_crs",
    "features_loaded",
    "point_features",
    "line_features",
    "polygon_features",
    "invalid_geometries_found",
    "invalid_geometries_repaired",
    "features_skipped",
    "status",
    "message",
]
LOG_COLUMNS = ["level", "source_path", "source_layer", "message"]
OUTPUT_COLUMNS = [
    "feature_id",
    "geometry_type",
    "source_scope",
    "source_name",
    "source_path",
    "source_layer",
    "source_feature_id",
    "original_geometry_type",
    "source_crs",
    "properties_json",
    "geometry",
]


@dataclass
class LayerResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    inventory: list[dict[str, str]] = field(default_factory=list)
    logs: list[dict[str, str]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize generic local spatial features into catalogs/normalized/spatial_features.gpkg."
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="TREEQUAKE project root. Paths are resolved relative to this directory.",
    )
    return parser.parse_args()


def import_geospatial_stack():
    try:
        import geopandas as gpd  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GeoPandas is required for the Spatial Feature Loader. Install geopandas with a Fiona or Pyogrio backend."
        ) from exc

    try:
        import shapely  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Shapely is required for geometry validation and repair.") from exc

    return gpd, shapely


def discover_spatial_files(catalog_root: Path) -> list[Path]:
    if not catalog_root.exists():
        return []
    normalized_root = catalog_root / "normalized"
    files: list[Path] = []
    for path in catalog_root.rglob("*"):
        if not path.is_file():
            continue
        if normalized_root in path.parents:
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def source_scope_for(catalog_root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(catalog_root)
    except ValueError:
        return "user"
    if rel.parts and rel.parts[0].lower() == "provided":
        return "provided"
    if rel.parts and rel.parts[0].lower() == "user":
        return "user"
    return "user"


def source_name_for(catalog_root: Path, path: Path, layer: str = "") -> str:
    try:
        rel = path.relative_to(catalog_root)
        parts = rel.parts
    except ValueError:
        parts = path.parts

    scope = parts[0].lower() if parts else ""
    candidates: list[str] = []
    if scope in {"provided", "user"} and len(parts) > 2:
        if len(parts) > 3:
            candidates.append(parts[-2])
        candidates.append(parts[1])
    if layer and layer != path.stem:
        candidates.append(layer)
    candidates.append(path.stem)

    for candidate in candidates:
        clean = str(candidate).strip()
        if clean:
            return clean
    return "unknown_source"


def geometry_family(geometry_type: str | None) -> str:
    if not geometry_type:
        return ""
    clean = geometry_type.strip().lower()
    if clean.startswith("multi"):
        clean = clean[5:]
    if clean == "point":
        return "point"
    if clean in {"line", "linestring"}:
        return "line"
    if clean == "polygon":
        return "polygon"
    return ""


def log_row(level: str, path: Path | str, layer: str, message: str) -> dict[str, str]:
    return {
        "level": level,
        "source_path": str(path),
        "source_layer": layer,
        "message": message,
    }


def stable_feature_id(path: Path, layer: str, source_feature_id: str, geometry_mapping: Any) -> str:
    token = json.dumps(
        {
            "path": path.as_posix(),
            "layer": layer,
            "source_feature_id": source_feature_id,
            "geometry": geometry_mapping,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha1(token.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def properties_json(row: Any, geometry_column: str) -> str:
    properties: dict[str, Any] = {}
    for key, value in row.items():
        if key == geometry_column:
            continue
        properties[str(key)] = json_safe(value)
    return json.dumps(properties, ensure_ascii=False, sort_keys=True, default=str)


def list_layers(path: Path) -> list[str]:
    if path.suffix.lower() != ".gpkg":
        return [path.stem]

    try:
        import pyogrio  # type: ignore

        layers = pyogrio.list_layers(path)
        return [str(layer[0]) for layer in layers]
    except Exception:
        pass

    try:
        import fiona  # type: ignore

        return [str(layer) for layer in fiona.listlayers(path)]
    except Exception:
        return [path.stem]


def read_layer(gpd: Any, path: Path, layer: str) -> Any:
    if path.suffix.lower() == ".gpkg":
        return gpd.read_file(path, layer=layer, encoding="utf-8")
    return gpd.read_file(path, encoding="utf-8")


def repair_geometry(geometry: Any, shapely_module: Any) -> tuple[Any, bool]:
    if geometry is None or geometry.is_empty or geometry.is_valid:
        return geometry, False

    make_valid = getattr(shapely_module, "make_valid", None)
    if make_valid is not None:
        try:
            repaired = make_valid(geometry)
            if repaired is not None and not repaired.is_empty and repaired.is_valid:
                return repaired, True
        except Exception:
            pass

    try:
        repaired = geometry.buffer(0)
        if repaired is not None and not repaired.is_empty and repaired.is_valid:
            return repaired, True
    except Exception:
        pass

    return geometry, False


def inventory_row(
    path: Path,
    layer: str,
    source_scope: str,
    source_name: str,
    source_crs: str,
    counts: Counter[str],
    invalid_found: int,
    invalid_repaired: int,
    skipped: int,
    status: str,
    message: str,
) -> dict[str, str]:
    return {
        "source_scope": source_scope,
        "source_name": source_name,
        "source_path": path.as_posix(),
        "source_layer": layer,
        "extension": path.suffix.lower().lstrip("."),
        "source_crs": source_crs,
        "output_crs": TARGET_CRS,
        "features_loaded": str(sum(counts.values())),
        "point_features": str(counts["point"]),
        "line_features": str(counts["line"]),
        "polygon_features": str(counts["polygon"]),
        "invalid_geometries_found": str(invalid_found),
        "invalid_geometries_repaired": str(invalid_repaired),
        "features_skipped": str(skipped),
        "status": status,
        "message": message,
    }


def load_layer(gpd: Any, shapely_module: Any, catalog_root: Path, path: Path, layer: str) -> LayerResult:
    result = LayerResult()
    source_scope = source_scope_for(catalog_root, path)
    source_name = source_name_for(catalog_root, path, layer)

    try:
        gdf = read_layer(gpd, path, layer)
    except Exception as exc:
        message = f"vector read failed: {exc}"
        result.logs.append(log_row("error", path, layer, message))
        result.inventory.append(
            inventory_row(path, layer, source_scope, source_name, "", Counter(), 0, 0, 0, "error", message)
        )
        return result

    source_crs = gdf.crs.to_string() if gdf.crs is not None else ""
    if not source_crs:
        result.logs.append(log_row("warning", path, layer, "source CRS was not declared; geometries were assumed to already be EPSG:4326"))
        gdf = gdf.set_crs(TARGET_CRS, allow_override=True)
        source_crs = TARGET_CRS

    invalid_found = 0
    invalid_repaired = 0
    skipped = 0
    geometry_column = gdf.geometry.name
    original_geometry_types = {
        index: str(geometry.geom_type) if geometry is not None and not geometry.is_empty else ""
        for index, geometry in gdf.geometry.items()
    }
    repaired_geometries = []
    for geometry in gdf.geometry:
        repaired, was_repaired = repair_geometry(geometry, shapely_module)
        if geometry is not None and not geometry.is_empty and not geometry.is_valid:
            invalid_found += 1
            if was_repaired:
                invalid_repaired += 1
        repaired_geometries.append(repaired)
    gdf[geometry_column] = gpd.GeoSeries(repaired_geometries, index=gdf.index, crs=gdf.crs)
    gdf = gdf.set_geometry(geometry_column, crs=gdf.crs)

    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS, allow_override=True)
    elif gdf.crs.to_string() != TARGET_CRS:
        try:
            gdf = gdf.to_crs(TARGET_CRS)
        except Exception as exc:
            message = f"CRS reprojection to {TARGET_CRS} failed: {exc}"
            result.logs.append(log_row("error", path, layer, message))
            result.inventory.append(
                inventory_row(path, layer, source_scope, source_name, source_crs, Counter(), invalid_found, invalid_repaired, len(gdf), "error", message)
            )
            return result

    counts: Counter[str] = Counter()
    for index, row in gdf.iterrows():
        geometry = row.geometry
        source_feature_id = str(index)
        if geometry is None or geometry.is_empty:
            skipped += 1
            result.logs.append(log_row("warning", path, layer, f"feature {source_feature_id} skipped: empty geometry"))
            continue
        if not geometry.is_valid:
            skipped += 1
            result.logs.append(log_row("warning", path, layer, f"feature {source_feature_id} skipped: invalid geometry could not be repaired"))
            continue

        original_geometry_type = original_geometry_types.get(index) or str(geometry.geom_type)
        family = geometry_family(original_geometry_type)
        if not family:
            family = geometry_family(str(geometry.geom_type))
        if not family:
            skipped += 1
            result.logs.append(log_row("warning", path, layer, f"feature {source_feature_id} skipped: unsupported geometry type '{original_geometry_type}'"))
            continue

        counts[family] += 1
        result.rows.append(
            {
                "feature_id": stable_feature_id(path, layer, source_feature_id, geometry.__geo_interface__),
                "geometry_type": family,
                "source_scope": source_scope,
                "source_name": source_name,
                "source_path": path.as_posix(),
                "source_layer": layer,
                "source_feature_id": source_feature_id,
                "original_geometry_type": original_geometry_type,
                "source_crs": source_crs,
                "properties_json": properties_json(row, geometry_column),
                "geometry": geometry,
            }
        )

    status = "ok"
    message = "loaded"
    if not result.rows:
        status = "skipped"
        message = "no supported point, line, or polygon features found"
    result.inventory.append(
        inventory_row(
            path,
            layer,
            source_scope,
            source_name,
            source_crs,
            counts,
            invalid_found,
            invalid_repaired,
            skipped,
            status,
            message,
        )
    )
    return result


def load_spatial_files(gpd: Any, shapely_module: Any, catalog_root: Path, spatial_files: list[Path]) -> LayerResult:
    merged = LayerResult()
    for path in spatial_files:
        for layer in list_layers(path):
            result = load_layer(gpd, shapely_module, catalog_root, path, layer)
            merged.rows.extend(result.rows)
            merged.inventory.extend(result.inventory)
            merged.logs.extend(result.logs)
    return merged


def write_geopackage(gpd: Any, gpkg_path: Path, rows: list[dict[str, Any]]) -> None:
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    if gpkg_path.exists():
        gpkg_path.unlink()

    gdf = gpd.GeoDataFrame(rows, columns=OUTPUT_COLUMNS, geometry="geometry", crs=TARGET_CRS)
    gdf.to_file(gpkg_path, layer=OUTPUT_LAYER, driver="GPKG", encoding="utf-8")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    catalog_root: Path,
    spatial_files: list[Path],
    rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, str]],
    log_rows: list[dict[str, str]],
    gpkg_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(str(row["geometry_type"]) for row in rows)
    status_counts = Counter(row["status"] for row in inventory_rows)
    invalid_found = sum(int(row.get("invalid_geometries_found", "0") or 0) for row in inventory_rows)
    invalid_repaired = sum(int(row.get("invalid_geometries_repaired", "0") or 0) for row in inventory_rows)
    skipped = sum(int(row.get("features_skipped", "0") or 0) for row in inventory_rows)

    lines = [
        "Spatial feature loader summary:",
        f"catalog root inspected: {catalog_root}",
        f"supported spatial files discovered: {len(spatial_files)}",
        f"features loaded: {len(rows)}",
        f"point features: {counts['point']}",
        f"line features: {counts['line']}",
        f"polygon features: {counts['polygon']}",
        f"invalid geometries found: {invalid_found}",
        f"invalid geometries repaired: {invalid_repaired}",
        f"features skipped: {skipped}",
        f"inventory rows: {len(inventory_rows)}",
        f"log rows: {len(log_rows)}",
        f"status counts: {', '.join(f'{key}={value}' for key, value in sorted(status_counts.items())) if status_counts else 'none'}",
        f"normalized GeoPackage: {gpkg_path}",
        "",
        "Scope note:",
        "- This loader preserves only geometry and provenance.",
        "- It normalizes geometry only to point, line, or polygon families.",
        "- It reprojects output geometries to EPSG:4326.",
        "- It does not assign semantic classes or perform spatial matching.",
    ]
    if not spatial_files:
        lines.extend(["", "Caveat:", "- No supported local spatial files were present under catalogs/ in this run."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    catalog_root = project_root / "catalogs"
    normalized_dir = catalog_root / "normalized"
    inventory_dir = project_root / "outputs" / "inventories"
    log_dir = project_root / "outputs" / "logs"

    gpkg_path = normalized_dir / "spatial_features.gpkg"
    inventory_path = inventory_dir / "spatial_features_inventory.csv"
    log_path = log_dir / "spatial_feature_loader_log.csv"
    summary_path = log_dir / "spatial_feature_loader_summary.txt"

    try:
        gpd, shapely_module = import_geospatial_stack()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    catalog_root.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    spatial_files = discover_spatial_files(catalog_root)
    result = LayerResult()
    if spatial_files:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = load_spatial_files(gpd, shapely_module, catalog_root, spatial_files)
            for warning in caught:
                result.logs.append(log_row("warning", catalog_root, "", str(warning.message)))
    else:
        result.logs.append(log_row("warning", catalog_root, "", "no supported spatial files found under catalogs/"))

    write_geopackage(gpd, gpkg_path, result.rows)
    write_csv(inventory_path, INVENTORY_COLUMNS, result.inventory)
    write_csv(log_path, LOG_COLUMNS, result.logs)
    write_summary(summary_path, catalog_root, spatial_files, result.rows, result.inventory, result.logs, gpkg_path)

    counts = Counter(str(row["geometry_type"]) for row in result.rows)
    errors = sum(1 for row in result.logs if row["level"] == "error")
    print(f"Wrote normalized spatial features to {gpkg_path}")
    print(f"Wrote {len(result.inventory)} inventory rows to {inventory_path}")
    print(f"Wrote {len(result.logs)} log rows to {log_path}")
    print(f"Wrote summary to {summary_path}")
    print(
        "Geometry counts: "
        f"point={counts['point']}, line={counts['line']}, polygon={counts['polygon']}, errors={errors}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
