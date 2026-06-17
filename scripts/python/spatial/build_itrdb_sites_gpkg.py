#!/usr/bin/env python3
"""Build a GeoPackage of coordinate-bearing ITRDB sites."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


WGS84 = "EPSG:4326"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert outputs/inventories/itrdb_sites.csv to a normalized GeoPackage."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Treequake project root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional explicit input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output GeoPackage path.",
    )
    return parser.parse_args()


def require_geopandas():
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: geopandas. Install the project geospatial dependencies "
            "before running this script."
        ) from exc
    return gpd


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    input_path = args.input or project_root / "outputs" / "inventories" / "itrdb_sites.csv"
    output_path = args.output or project_root / "catalogs" / "normalized" / "itrdb_sites.gpkg"

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    gpd = require_geopandas()

    sites = pd.read_csv(input_path, encoding="utf-8")
    total_rows = len(sites)

    latitude = pd.to_numeric(sites.get("latitude"), errors="coerce")
    longitude = pd.to_numeric(sites.get("longitude"), errors="coerce")
    valid_coordinates = (
        latitude.between(-90, 90, inclusive="both")
        & longitude.between(-180, 180, inclusive="both")
    )

    sites = sites.copy()
    sites["latitude"] = latitude
    sites["longitude"] = longitude
    sites["has_valid_coordinates"] = valid_coordinates

    points = gpd.points_from_xy(
        sites.loc[valid_coordinates, "longitude"],
        sites.loc[valid_coordinates, "latitude"],
        crs=WGS84,
    )
    coordinate_sites = gpd.GeoDataFrame(
        sites.loc[valid_coordinates].copy(),
        geometry=points,
        crs=WGS84,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    coordinate_sites.to_file(
        output_path,
        layer="itrdb_sites",
        driver="GPKG",
        index=False,
        encoding="UTF-8",
    )

    all_geometry = pd.Series([None] * len(sites), index=sites.index, dtype=object)
    all_geometry.loc[valid_coordinates] = list(points)
    all_sites = gpd.GeoDataFrame(sites.copy(), geometry=all_geometry, crs=WGS84)
    all_sites.to_file(
        output_path,
        layer="itrdb_sites_all_rows",
        driver="GPKG",
        index=False,
        encoding="UTF-8",
    )

    missing_coordinates = total_rows - int(valid_coordinates.sum())
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total rows: {total_rows}")
    print(f"Valid coordinates: {int(valid_coordinates.sum())}")
    print(f"Missing coordinates: {missing_coordinates}")


if __name__ == "__main__":
    main()
