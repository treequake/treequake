#!/usr/bin/env python3
"""Load generic spatial features into a normalized GeoPackage.

This phase intentionally preserves only source geometry and provenance. It does
not assign domain classes or perform spatial matching.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import struct
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_EXTENSIONS = {".geojson", ".json", ".gpkg", ".shp", ".kml", ".kmz"}
GEOMETRY_COLUMNS = [
    "feature_id",
    "geometry_type",
    "source_scope",
    "source_name",
    "source_path",
    "source_layer",
    "source_feature_id",
    "original_geometry_type",
    "properties_json",
]
INVENTORY_COLUMNS = [
    "source_scope",
    "source_name",
    "source_path",
    "source_layer",
    "extension",
    "features_loaded",
    "point_features",
    "line_features",
    "polygon_features",
    "status",
    "message",
]
LOG_COLUMNS = ["level", "source_path", "source_layer", "message"]


@dataclass
class FeatureRecord:
    feature_id: str
    geometry_type: str
    source_scope: str
    source_name: str
    source_path: str
    source_layer: str
    source_feature_id: str
    original_geometry_type: str
    properties_json: str
    gpkg_geometry: bytes


@dataclass
class InventoryRecord:
    row: dict[str, str]


@dataclass
class LoadResult:
    features: list[FeatureRecord] = field(default_factory=list)
    inventory: list[InventoryRecord] = field(default_factory=list)
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


def discover_spatial_files(catalog_root: Path) -> list[Path]:
    if not catalog_root.exists():
        return []
    normalized_root = catalog_root / "normalized"
    files = []
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
    if clean in {"point", "line", "polygon"}:
        return clean
    if clean == "linestring":
        return "line"
    if clean == "multilinestring":
        return "line"
    if clean == "geometrycollection":
        return ""
    return ""


def log_row(level: str, path: Path | str, layer: str, message: str) -> dict[str, str]:
    return {
        "level": level,
        "source_path": str(path),
        "source_layer": layer,
        "message": message,
    }


def stable_feature_id(path: Path, layer: str, source_feature_id: str, geometry: dict[str, Any]) -> str:
    token = json.dumps(
        {
            "path": path.as_posix(),
            "layer": layer,
            "source_feature_id": source_feature_id,
            "geometry": geometry,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(token.encode("utf-8")).hexdigest()


def gpkg_header(srs_id: int = 4326) -> bytes:
    return b"GP" + bytes([0, 1]) + struct.pack("<i", srs_id)


def wkb_point(coords: list[Any]) -> bytes:
    return struct.pack("<BI", 1, 1) + struct.pack("<dd", float(coords[0]), float(coords[1]))


def wkb_line(coords: list[Any]) -> bytes:
    body = struct.pack("<BI", 1, 2) + struct.pack("<I", len(coords))
    for coord in coords:
        body += struct.pack("<dd", float(coord[0]), float(coord[1]))
    return body


def wkb_polygon(coords: list[Any]) -> bytes:
    body = struct.pack("<BI", 1, 3) + struct.pack("<I", len(coords))
    for ring in coords:
        body += struct.pack("<I", len(ring))
        for coord in ring:
            body += struct.pack("<dd", float(coord[0]), float(coord[1]))
    return body


def wkb_multipoint(coords: list[Any]) -> bytes:
    body = struct.pack("<BI", 1, 4) + struct.pack("<I", len(coords))
    for coord in coords:
        body += wkb_point(coord)
    return body


def wkb_multiline(coords: list[Any]) -> bytes:
    body = struct.pack("<BI", 1, 5) + struct.pack("<I", len(coords))
    for line in coords:
        body += wkb_line(line)
    return body


def wkb_multipolygon(coords: list[Any]) -> bytes:
    body = struct.pack("<BI", 1, 6) + struct.pack("<I", len(coords))
    for polygon in coords:
        body += wkb_polygon(polygon)
    return body


def geometry_to_gpkg_blob(geometry: dict[str, Any]) -> bytes:
    geometry_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geometry_type == "Point":
        wkb = wkb_point(coords)
    elif geometry_type == "LineString":
        wkb = wkb_line(coords)
    elif geometry_type == "Polygon":
        wkb = wkb_polygon(coords)
    elif geometry_type == "MultiPoint":
        wkb = wkb_multipoint(coords)
    elif geometry_type == "MultiLineString":
        wkb = wkb_multiline(coords)
    elif geometry_type == "MultiPolygon":
        wkb = wkb_multipolygon(coords)
    else:
        raise ValueError(f"unsupported geometry type for GeoPackage output: {geometry_type}")
    return gpkg_header() + wkb


def read_geojson(path: Path, catalog_root: Path) -> LoadResult:
    result = LoadResult()
    layer = path.stem
    source_scope = source_scope_for(catalog_root, path)
    source_name = source_name_for(catalog_root, path, layer)
    counts = Counter()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        message = f"could not read GeoJSON: {exc}"
        result.logs.append(log_row("error", path, layer, message))
        result.inventory.append(inventory_row(path, layer, source_scope, source_name, counts, "error", message))
        return result

    if data.get("type") == "FeatureCollection":
        raw_features = data.get("features") or []
    elif data.get("type") == "Feature":
        raw_features = [data]
    elif data.get("type") in {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"}:
        raw_features = [{"type": "Feature", "properties": {}, "geometry": data}]
    else:
        message = f"unsupported GeoJSON root type: {data.get('type', '')}"
        result.logs.append(log_row("warning", path, layer, message))
        result.inventory.append(inventory_row(path, layer, source_scope, source_name, counts, "skipped", message))
        return result

    for index, raw_feature in enumerate(raw_features):
        geometry = raw_feature.get("geometry") or {}
        original_type = str(geometry.get("type", ""))
        family = geometry_family(original_type)
        if not family:
            result.logs.append(log_row("warning", path, layer, f"feature {index} skipped: unsupported geometry type '{original_type}'"))
            continue
        try:
            gpkg_geometry = geometry_to_gpkg_blob(geometry)
        except (TypeError, ValueError, IndexError) as exc:
            result.logs.append(log_row("warning", path, layer, f"feature {index} skipped: invalid geometry: {exc}"))
            continue

        source_feature_id = str(raw_feature.get("id", index))
        properties = raw_feature.get("properties") or {}
        feature_id = stable_feature_id(path, layer, source_feature_id, geometry)
        result.features.append(
            FeatureRecord(
                feature_id=feature_id,
                geometry_type=family,
                source_scope=source_scope,
                source_name=source_name,
                source_path=path.as_posix(),
                source_layer=layer,
                source_feature_id=source_feature_id,
                original_geometry_type=original_type,
                properties_json=json.dumps(properties, sort_keys=True, ensure_ascii=False),
                gpkg_geometry=gpkg_geometry,
            )
        )
        counts[family] += 1

    status = "ok"
    message = "loaded"
    if not result.features:
        status = "skipped"
        message = "no supported point, line, or polygon features found"
    result.inventory.append(inventory_row(path, layer, source_scope, source_name, counts, status, message))
    return result


def inventory_row(
    path: Path,
    layer: str,
    source_scope: str,
    source_name: str,
    counts: Counter[str],
    status: str,
    message: str,
) -> InventoryRecord:
    return InventoryRecord(
        {
            "source_scope": source_scope,
            "source_name": source_name,
            "source_path": path.as_posix(),
            "source_layer": layer,
            "extension": path.suffix.lower().lstrip("."),
            "features_loaded": str(sum(counts.values())),
            "point_features": str(counts["point"]),
            "line_features": str(counts["line"]),
            "polygon_features": str(counts["polygon"]),
            "status": status,
            "message": message,
        }
    )


def read_with_optional_vector_stack(path: Path, catalog_root: Path) -> LoadResult:
    try:
        import geopandas as gpd  # type: ignore
    except ModuleNotFoundError:
        return unsupported_dependency_result(path, catalog_root, "geopandas is not installed")

    result = LoadResult()
    layers = [path.stem]
    try:
        if path.suffix.lower() == ".gpkg":
            try:
                import fiona  # type: ignore

                layers = list(fiona.listlayers(path))
            except Exception:
                layers = [path.stem]
        for layer in layers:
            gdf = gpd.read_file(path, layer=layer if path.suffix.lower() == ".gpkg" else None)
            source_scope = source_scope_for(catalog_root, path)
            source_name = source_name_for(catalog_root, path, layer)
            counts: Counter[str] = Counter()
            loaded = 0
            for index, row in gdf.iterrows():
                geometry = row.geometry
                if geometry is None or geometry.is_empty:
                    result.logs.append(log_row("warning", path, layer, f"feature {index} skipped: empty geometry"))
                    continue
                original_type = geometry.geom_type
                family = geometry_family(original_type)
                if not family:
                    result.logs.append(log_row("warning", path, layer, f"feature {index} skipped: unsupported geometry type '{original_type}'"))
                    continue
                properties = row.drop(labels=[gdf.geometry.name]).to_dict()
                properties_json = json.dumps(properties, default=str, sort_keys=True, ensure_ascii=False)
                source_feature_id = str(index)
                feature_id = stable_feature_id(path, layer, source_feature_id, dict(geometry.__geo_interface__))
                result.features.append(
                    FeatureRecord(
                        feature_id=feature_id,
                        geometry_type=family,
                        source_scope=source_scope,
                        source_name=source_name,
                        source_path=path.as_posix(),
                        source_layer=layer,
                        source_feature_id=source_feature_id,
                        original_geometry_type=original_type,
                        properties_json=properties_json,
                        gpkg_geometry=gpkg_header() + geometry.wkb,
                    )
                )
                counts[family] += 1
                loaded += 1
            result.inventory.append(inventory_row(path, layer, source_scope, source_name, counts, "ok", "loaded" if loaded else "no supported point, line, or polygon features found"))
    except Exception as exc:
        source_scope = source_scope_for(catalog_root, path)
        source_name = source_name_for(catalog_root, path, path.stem)
        result.logs.append(log_row("error", path, path.stem, f"vector read failed: {exc}"))
        result.inventory.append(inventory_row(path, path.stem, source_scope, source_name, Counter(), "error", f"vector read failed: {exc}"))
    return result


def unsupported_dependency_result(path: Path, catalog_root: Path, message: str) -> LoadResult:
    source_scope = source_scope_for(catalog_root, path)
    source_name = source_name_for(catalog_root, path, path.stem)
    result = LoadResult()
    result.logs.append(log_row("warning", path, path.stem, f"skipped {path.suffix.lower()} input: {message}"))
    result.inventory.append(inventory_row(path, path.stem, source_scope, source_name, Counter(), "skipped", message))
    return result


def load_file(path: Path, catalog_root: Path) -> LoadResult:
    if path.suffix.lower() in {".geojson", ".json"}:
        return read_geojson(path, catalog_root)
    return read_with_optional_vector_stack(path, catalog_root)


def create_empty_gpkg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA application_id = 1196437808")
        conn.execute("PRAGMA user_version = 10400")
        conn.executescript(
            """
            CREATE TABLE gpkg_spatial_ref_sys (
                srs_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL PRIMARY KEY,
                organization TEXT NOT NULL,
                organization_coordsys_id INTEGER NOT NULL,
                definition TEXT NOT NULL,
                description TEXT
            );
            INSERT INTO gpkg_spatial_ref_sys VALUES
                ('Undefined cartesian SRS', -1, 'NONE', -1, 'undefined', 'undefined cartesian coordinate reference system'),
                ('Undefined geographic SRS', 0, 'NONE', 0, 'undefined', 'undefined geographic coordinate reference system'),
                ('WGS 84 geodetic', 4326, 'EPSG', 4326, 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]', 'longitude/latitude coordinates in decimal degrees');

            CREATE TABLE gpkg_contents (
                table_name TEXT NOT NULL PRIMARY KEY,
                data_type TEXT NOT NULL,
                identifier TEXT UNIQUE,
                description TEXT DEFAULT '',
                last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                min_x DOUBLE,
                min_y DOUBLE,
                max_x DOUBLE,
                max_y DOUBLE,
                srs_id INTEGER,
                CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
            );

            CREATE TABLE gpkg_geometry_columns (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL,
                m TINYINT NOT NULL,
                PRIMARY KEY (table_name, column_name),
                CONSTRAINT fk_ggc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
                CONSTRAINT fk_ggc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
            );

            CREATE TABLE spatial_features (
                fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                feature_id TEXT NOT NULL UNIQUE,
                geometry_type TEXT NOT NULL,
                source_scope TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_layer TEXT NOT NULL,
                source_feature_id TEXT NOT NULL,
                original_geometry_type TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                geom BLOB
            );

            INSERT INTO gpkg_contents
                (table_name, data_type, identifier, description, srs_id)
            VALUES
                ('spatial_features', 'features', 'spatial_features', 'Generic source spatial features with provenance only', 4326);

            INSERT INTO gpkg_geometry_columns
                (table_name, column_name, geometry_type_name, srs_id, z, m)
            VALUES
                ('spatial_features', 'geom', 'GEOMETRY', 4326, 0, 0);
            """
        )
        conn.commit()
    finally:
        conn.close()


def write_features(path: Path, features: Iterable[FeatureRecord]) -> None:
    create_empty_gpkg(path)
    conn = sqlite3.connect(path)
    try:
        conn.executemany(
            """
            INSERT INTO spatial_features (
                feature_id, geometry_type, source_scope, source_name, source_path,
                source_layer, source_feature_id, original_geometry_type, properties_json, geom
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    feature.feature_id,
                    feature.geometry_type,
                    feature.source_scope,
                    feature.source_name,
                    feature.source_path,
                    feature.source_layer,
                    feature.source_feature_id,
                    feature.original_geometry_type,
                    feature.properties_json,
                    feature.gpkg_geometry,
                )
                for feature in features
            ],
        )
        conn.execute("UPDATE gpkg_contents SET last_change = ?", (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),))
        conn.commit()
    finally:
        conn.close()


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
    features: list[FeatureRecord],
    inventory_rows: list[dict[str, str]],
    log_rows: list[dict[str, str]],
    gpkg_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_counts = Counter(feature.geometry_type for feature in features)
    status_counts = Counter(row["status"] for row in inventory_rows)
    lines = [
        "Spatial feature loader summary:",
        f"catalog root inspected: {catalog_root}",
        f"supported spatial files discovered: {len(spatial_files)}",
        f"features loaded: {len(features)}",
        f"point features: {feature_counts['point']}",
        f"line features: {feature_counts['line']}",
        f"polygon features: {feature_counts['polygon']}",
        f"inventory rows: {len(inventory_rows)}",
        f"log rows: {len(log_rows)}",
        f"status counts: {', '.join(f'{key}={value}' for key, value in sorted(status_counts.items())) if status_counts else 'none'}",
        f"normalized GeoPackage: {gpkg_path}",
        "",
        "Scope note:",
        "- This loader preserves only geometry and provenance.",
        "- It normalizes geometry only to point, line, or polygon families.",
        "- It does not assign semantic classes or perform spatial matching.",
    ]
    if not spatial_files:
        lines.extend(
            [
                "",
                "Caveat:",
                "- No supported local spatial files were present under catalogs/ in this run.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    catalog_root = project_root / "catalogs"
    normalized_dir = catalog_root / "normalized"
    inventory_dir = project_root / "outputs" / "inventories"
    log_dir = project_root / "outputs" / "logs"

    catalog_root.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    spatial_files = discover_spatial_files(catalog_root)
    all_features: list[FeatureRecord] = []
    inventory_rows: list[dict[str, str]] = []
    log_rows: list[dict[str, str]] = []

    if not spatial_files:
        log_rows.append(log_row("warning", catalog_root, "", "no supported spatial files found under catalogs/"))

    for path in spatial_files:
        result = load_file(path, catalog_root)
        all_features.extend(result.features)
        inventory_rows.extend(record.row for record in result.inventory)
        log_rows.extend(result.logs)

    gpkg_path = normalized_dir / "spatial_features.gpkg"
    inventory_path = inventory_dir / "spatial_features_inventory.csv"
    log_path = log_dir / "spatial_feature_loader_log.csv"
    summary_path = log_dir / "spatial_feature_loader_summary.txt"

    write_features(gpkg_path, all_features)
    write_csv(inventory_path, INVENTORY_COLUMNS, inventory_rows)
    write_csv(log_path, LOG_COLUMNS, log_rows)
    write_summary(summary_path, catalog_root, spatial_files, all_features, inventory_rows, log_rows, gpkg_path)

    counts = Counter(feature.geometry_type for feature in all_features)
    print(f"Wrote normalized spatial features to {gpkg_path}")
    print(f"Wrote {len(inventory_rows)} inventory rows to {inventory_path}")
    print(f"Wrote {len(log_rows)} log rows to {log_path}")
    print(f"Wrote summary to {summary_path}")
    print(
        "Geometry counts: "
        f"point={counts['point']}, line={counts['line']}, polygon={counts['polygon']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
