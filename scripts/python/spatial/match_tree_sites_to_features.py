#!/usr/bin/env python3
"""Match coordinate-bearing ITRDB sites to buffered spatial features."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


WGS84 = "EPSG:4326"

COUNTRY_NAME_TEXT = """
Afghanistan|Albania|Algeria|Andorra|Angola|Argentina|Armenia|Australia|Austria|Azerbaijan|Bahamas|Bahrain|Bangladesh|Belarus|Belgium|Belize|Benin|Bhutan|Bolivia|Bosnia and Herzegovina|Botswana|Brazil|Bulgaria|Burkina Faso|Burundi|Cambodia|Cameroon|Canada|Central African Republic|Chad|Chile|China|Colombia|Costa Rica|Croatia|Cuba|Cyprus|Czechia|Democratic Republic of the Congo|Denmark|Dominican Republic|Ecuador|Egypt|El Salvador|Eritrea|Estonia|Eswatini|Ethiopia|Finland|France|Georgia|Germany|Ghana|Greece|Guatemala|Haiti|Honduras|Hungary|Iceland|India|Indonesia|Iran|Iraq|Ireland|Israel|Italy|Japan|Jordan|Kazakhstan|Kenya|Kosovo|Kyrgyzstan|Laos|Latvia|Lebanon|Libya|Liechtenstein|Lithuania|Luxembourg|Madagascar|Malawi|Malaysia|Mali|Malta|Mexico|Moldova|Mongolia|Montenegro|Morocco|Mozambique|Myanmar|Namibia|Nepal|Netherlands|New Zealand|Nicaragua|Niger|Nigeria|North Macedonia|Norway|Oman|Pakistan|Panama|Papua New Guinea|Paraguay|Peru|Philippines|Poland|Portugal|Romania|Russia|Rwanda|Saudi Arabia|Senegal|Serbia|Slovakia|Slovenia|Somalia|South Africa|South Korea|Spain|Sri Lanka|Sudan|Sweden|Switzerland|Syria|Taiwan|Tajikistan|Tanzania|Thailand|Tunisia|Turkey|Turkmenistan|Uganda|Ukraine|United Kingdom|United States|Uruguay|Uzbekistan|Venezuela|Vietnam|Yemen|Zambia|Zimbabwe
""".strip()

COUNTRY_ALIASES = {name.lower(): name for name in COUNTRY_NAME_TEXT.split("|")}
COUNTRY_ALIASES.update(
    {
        "bosnia-herzegovina": "Bosnia and Herzegovina",
        "czech republic": "Czechia",
        "drc": "Democratic Republic of the Congo",
        "england": "United Kingdom",
        "great britain": "United Kingdom",
        "iran, islamic republic of": "Iran",
        "italia": "Italy",
        "korea": "South Korea",
        "korea, republic of": "South Korea",
        "lao pdr": "Laos",
        "macedonia": "North Macedonia",
        "republic of korea": "South Korea",
        "russian federation": "Russia",
        "scotland": "United Kingdom",
        "soviet union": "Russia",
        "spain mainland": "Spain",
        "suisse": "Switzerland",
        "syrian arab republic": "Syria",
        "turkiye": "Turkey",
        "u.k.": "United Kingdom",
        "uk": "United Kingdom",
        "u.s.": "United States",
        "u.s.a.": "United States",
        "usa": "United States",
        "united states of america": "United States",
        "viet nam": "Vietnam",
        "wales": "United Kingdom",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find ITRDB sites that fall inside buffered spatial feature polygons."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Treequake project root.",
    )
    parser.add_argument(
        "--buffer-km",
        type=float,
        required=True,
        help="Buffer distance in kilometers.",
    )
    parser.add_argument(
        "--spatial-features",
        type=Path,
        default=None,
        help="Optional explicit spatial features GeoPackage path.",
    )
    parser.add_argument(
        "--itrdb-sites",
        type=Path,
        default=None,
        help="Optional explicit ITRDB sites GeoPackage path.",
    )
    parser.add_argument(
        "--itrdb-products",
        type=Path,
        default=None,
        help="Optional explicit ITRDB products CSV path.",
    )
    parser.add_argument(
        "--polygon-buffer",
        action="store_true",
        help="Also buffer polygon features. By default polygon features are used as-is.",
    )
    return parser.parse_args()


def require_geospatial_stack():
    try:
        import geopandas as gpd
        from pyproj import CRS
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: geopandas and pyproj are required for spatial matching."
        ) from exc
    return gpd, CRS


def list_layers(path: Path) -> list[str]:
    try:
        import pyogrio

        return [row[0] for row in pyogrio.list_layers(path)]
    except Exception:
        try:
            import fiona

            return list(fiona.listlayers(path))
        except Exception as exc:
            raise SystemExit(f"Could not list layers in {path}: {exc}") from exc


def read_first_layer(gpd, path: Path, preferred_layers: Iterable[str]):
    layers = list_layers(path)
    for layer in preferred_layers:
        if layer in layers:
            return gpd.read_file(path, layer=layer), layer
    if not layers:
        raise SystemExit(f"No layers found in {path}")
    return gpd.read_file(path, layer=layers[0]), layers[0]


def progress_iter(iterable, total: int | None = None, desc: str = "Working"):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        return iterable


def geometry_type_label(geom) -> str:
    if geom is None or geom.is_empty:
        return "unknown"
    geom_type = geom.geom_type.lower()
    if "point" in geom_type:
        return "point"
    if "line" in geom_type:
        return "line"
    if "polygon" in geom_type:
        return "polygon"
    return geom_type


def utm_crs_for_geometry(CRS, geometry) -> str:
    centroid = geometry.centroid
    lon = ((centroid.x + 180) % 360) - 180
    lat = centroid.y
    zone = int((lon + 180) // 6) + 1
    zone = max(1, min(60, zone))
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg).to_string()


def unwrap_antimeridian_geometry(geometry):
    """Make dateline-crossing coordinate sequences continuous before projection."""
    if geometry is None or geometry.is_empty:
        return geometry

    minx, _, maxx, _ = geometry.bounds
    if maxx - minx <= 180:
        return geometry

    try:
        from shapely.ops import transform
    except ImportError:
        return geometry

    def shift_negative_longitudes(x, y, z=None):
        shifted_x = x + 360 if x < 0 else x
        return (shifted_x, y) if z is None else (shifted_x, y, z)

    return transform(shift_negative_longitudes, geometry)


def repair_antimeridian_polygon(geometry):
    """Split dateline-spanning WGS84 polygons so they do not cover the world."""
    if geometry is None or geometry.is_empty:
        return geometry

    minx, _, maxx, _ = geometry.bounds
    if maxx - minx <= 180:
        return geometry

    try:
        from shapely.geometry import LineString, MultiPolygon
        from shapely.ops import split, transform
    except ImportError:
        return geometry

    def shift_negative_longitudes(x, y, z=None):
        shifted_x = x + 360 if x < 0 else x
        return (shifted_x, y) if z is None else (shifted_x, y, z)

    def restore_longitudes(x, y, z=None):
        restored_x = x - 360 if x > 180 else x
        return (restored_x, y) if z is None else (restored_x, y, z)

    shifted = transform(shift_negative_longitudes, geometry)
    if shifted.is_empty:
        return geometry

    splitter = LineString([(180, -90), (180, 90)])
    try:
        parts = list(split(shifted, splitter).geoms)
    except Exception:
        parts = [shifted]

    restored_parts = []
    for part in parts:
        if part.is_empty:
            continue
        restored = transform(restore_longitudes, part)
        if restored.is_empty:
            continue
        if restored.geom_type == "Polygon":
            restored_parts.append(restored)
        elif restored.geom_type == "MultiPolygon":
            restored_parts.extend([piece for piece in restored.geoms if not piece.is_empty])

    if not restored_parts:
        return geometry
    if len(restored_parts) == 1:
        return restored_parts[0]
    return MultiPolygon(restored_parts)


def is_world_width_thin_artifact(geometry) -> bool:
    if geometry is None or geometry.is_empty:
        return False
    minx, miny, maxx, maxy = geometry.bounds
    width = maxx - minx
    height = maxy - miny
    return width > 300 and height < 5


def build_analysis_polygons(gpd, CRS, features, buffer_km: float, polygon_buffer: bool):
    buffer_m = buffer_km * 1000.0
    rows = []

    if features.crs is None:
        raise SystemExit("Spatial features GeoPackage has no CRS; refusing to buffer.")

    features_wgs84 = features.to_crs(WGS84)

    feature_rows = features_wgs84.reset_index(drop=True)
    for feature_index, row in progress_iter(
        feature_rows.iterrows(),
        total=len(feature_rows),
        desc="Building analysis polygons",
    ):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        geometry_type = (
            str(row.get("geometry_type")).lower()
            if pd.notna(row.get("geometry_type"))
            else geometry_type_label(geom)
        )

        if geometry_type not in {"point", "line", "polygon"}:
            geometry_type = geometry_type_label(geom)

        should_buffer = geometry_type in {"point", "line"} or (
            geometry_type == "polygon" and polygon_buffer
        )
        if geometry_type in {"line", "polygon"}:
            geom = unwrap_antimeridian_geometry(geom)

        projected_crs = utm_crs_for_geometry(CRS, geom)
        one_feature = gpd.GeoDataFrame([row.drop(labels="geometry")], geometry=[geom], crs=WGS84)
        projected = one_feature.to_crs(projected_crs)

        if projected.crs is None or projected.crs.is_geographic:
            raise SystemExit(f"Refusing to buffer feature {feature_index} in geographic CRS.")

        final_geom = projected.geometry.iloc[0].buffer(buffer_m) if should_buffer else projected.geometry.iloc[0]
        output_row = row.drop(labels="geometry").to_dict()
        output_row.update(
            {
                "feature_index": feature_index,
                "analysis_geometry_type": geometry_type,
                "buffer_km": buffer_km if should_buffer else 0.0,
                "analysis_crs": projected_crs,
            }
        )
        rows.append((output_row, final_geom, projected_crs))

    if not rows:
        return gpd.GeoDataFrame(
            columns=["feature_index", "analysis_geometry_type", "buffer_km", "analysis_crs", "geometry"],
            geometry="geometry",
            crs=WGS84,
        )

    polygon_frames = []
    for output_row, geom, projected_crs in rows:
        frame = gpd.GeoDataFrame([output_row], geometry=[geom], crs=projected_crs).to_crs(WGS84)
        frame["geometry"] = frame.geometry.map(repair_antimeridian_polygon)
        artifact_mask = frame.geometry.map(is_world_width_thin_artifact)
        if artifact_mask.any():
            feature_index = output_row.get("feature_index", "unknown")
            print(f"Skipping antimeridian artifact analysis polygon for feature_index={feature_index}")
            frame = frame.loc[~artifact_mask].copy()
        if frame.empty:
            continue
        polygon_frames.append(frame)

    if not polygon_frames:
        return gpd.GeoDataFrame(
            columns=["feature_index", "analysis_geometry_type", "buffer_km", "analysis_crs", "geometry"],
            geometry="geometry",
            crs=WGS84,
        )

    return pd.concat(polygon_frames, ignore_index=True).pipe(
        lambda df: gpd.GeoDataFrame(df, geometry="geometry", crs=WGS84)
    )


def first_existing_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None





def clean_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "-------------"}:
        return None
    return text


def lower_clean_text(value) -> str | None:
    text = clean_text(value)
    return text.lower() if text is not None else None


def infer_site_code_base(value) -> str | None:
    """Return the stable alphanumeric site-code stem, preserving nonmatching codes."""
    text = lower_clean_text(value)
    if text is None:
        return None
    match = re.match(r"^([a-z]+[0-9]+)", text)
    return match.group(1) if match else text


def normalize_country_text(value) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text).strip(" .;:,")
    if not text or any(char.isdigit() for char in text):
        return None

    candidates = [text]
    for part in re.split(r"[,;/|()]+", text):
        part = part.strip(" .;:,")
        if part:
            candidates.append(part)

    for candidate in candidates:
        normalized = candidate.lower()
        normalized = normalized.replace("&", "and")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[normalized]
    return None


def load_country_lookup(gpd, project_root: Path):
    candidate_paths = [
        project_root / "catalogs" / "normalized" / "countries.gpkg",
        project_root / "catalogs" / "normalized" / "country_boundaries.gpkg",
        project_root / "catalogs" / "reference" / "countries.gpkg",
        project_root / "catalogs" / "reference" / "ne_10m_admin_0_countries.shp",
        project_root / "catalogs" / "reference" / "ne_50m_admin_0_countries.shp",
        project_root / "catalogs" / "reference" / "ne_110m_admin_0_countries.shp",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            frame = gpd.read_file(path)
            if frame.empty or frame.geometry.isna().all():
                continue
            if frame.crs is None:
                frame = frame.set_crs(WGS84)
            else:
                frame = frame.to_crs(WGS84)
            name_column = first_existing_column(
                frame,
                ["ADMIN", "NAME_EN", "NAME", "SOVEREIGNT", "country", "Country", "name"],
            )
            if name_column:
                print(f"Loaded coordinate country lookup: {path}")
                return {"frame": frame, "name_column": name_column}
        except Exception as exc:
            print(f"Could not load country lookup from {path}: {exc}")
    return None


def lookup_country_from_coordinates(country_lookup, latitude, longitude) -> str | None:
    if country_lookup is None:
        return None
    try:
        from shapely.geometry import Point
    except ImportError:
        return None

    lat = pd.to_numeric(pd.Series([latitude]), errors="coerce").iloc[0]
    lon = pd.to_numeric(pd.Series([longitude]), errors="coerce").iloc[0]
    if pd.isna(lat) or pd.isna(lon):
        return None

    point = Point(float(lon), float(lat))
    frame = country_lookup["frame"]
    name_column = country_lookup["name_column"]
    matches = frame.loc[frame.geometry.contains(point) | frame.geometry.touches(point)]
    if matches.empty:
        return None

    for value in matches[name_column]:
        country = normalize_country_text(value) or clean_text(value)
        if country:
            return country
    return None


def resolve_country_for_group(
    group: pd.DataFrame,
    metadata_country_column: str | None,
    parsed_location_column: str | None,
    latitude_column: str | None,
    longitude_column: str | None,
    country_lookup,
) -> tuple[str, str]:
    if metadata_country_column:
        for value in group[metadata_country_column]:
            country = normalize_country_text(value)
            if country:
                return country, "metadata"

    if parsed_location_column:
        for value in group[parsed_location_column]:
            country = normalize_country_text(value)
            if country:
                return country, "parsed_location"

    if latitude_column and longitude_column:
        lat = numeric_min(group[latitude_column])
        lon = numeric_min(group[longitude_column])
        country = lookup_country_from_coordinates(country_lookup, lat, lon)
        if country:
            return country, "coordinate_lookup"

    return "Unknown", "unknown"


def temporal_warnings_for_values(first_year, last_year, current_year: int | None = None) -> str:
    current_year = current_year or datetime.now().year
    first_value = pd.to_numeric(pd.Series([first_year]), errors="coerce").iloc[0]
    last_value = pd.to_numeric(pd.Series([last_year]), errors="coerce").iloc[0]
    warnings = []
    if pd.notna(last_value) and int(last_value) > current_year + 1:
        warnings.append(f"last_year {int(last_value)} exceeds current_year+1 ({current_year + 1})")
    if pd.notna(first_value) and pd.notna(last_value) and int(first_value) > int(last_value):
        warnings.append(f"first_year {int(first_value)} exceeds last_year {int(last_value)}")
    return "; ".join(warnings)


def unique_clean_values(series: pd.Series) -> list[str]:
    values = [clean_text(value) for value in series.dropna()]
    return sorted({value for value in values if value is not None})


def unique_normalized_values(series: pd.Series) -> set[str]:
    return {value for value in (lower_clean_text(value) for value in series.dropna()) if value is not None}


def first_clean_value(series: pd.Series) -> str | None:
    for value in series:
        text = clean_text(value)
        if text is not None:
            return text
    return None


def numeric_min(series: pd.Series):
    value = pd.to_numeric(series, errors="coerce").min()
    return None if pd.isna(value) else value


def numeric_max(series: pd.Series):
    value = pd.to_numeric(series, errors="coerce").max()
    return None if pd.isna(value) else value


def bool_any(series: pd.Series) -> bool:
    if series.empty:
        return False
    return bool(series.map(is_truthy).any())


def count_product_type(products: pd.DataFrame, product_type: str) -> int:
    if products.empty or "product_type" not in products.columns:
        return 0
    return int(products["product_type"].fillna("").astype(str).str.lower().eq(product_type).sum())


def is_truthy(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def count_truthy(series: pd.Series) -> int:
    if series.empty:
        return 0
    return int(series.map(is_truthy).sum())


def count_other_products(products: pd.DataFrame) -> int:
    if products.empty or "product_type" not in products.columns:
        return 0
    product_types = products["product_type"].fillna("").astype(str).str.lower()
    return int((~product_types.isin({"rwl", "crn"})).sum())


def joined_unique(values: list[str], limit: int = 12) -> str:
    if not values:
        return ""
    shown = values[:limit]
    suffix = f"; ... (+{len(values) - limit} more)" if len(values) > limit else ""
    return "; ".join(shown) + suffix


def build_physical_site_id_map(site_rows: pd.DataFrame, site_key: str) -> tuple[dict[str, str], list[str]]:
    """Build a parser-assisted physical-site map from matched site rows.

    The parser-derived root_site_code_guess is the starting point, but product
    suffixes such as ital012e/ital012ea are collapsed only when the parsed site
    attributes support a shared physical location.
    """
    if site_key not in site_rows.columns:
        return {}, [f"missing required site key column: {site_key}"]

    reference = site_rows.copy()
    reference[site_key] = reference[site_key].map(lower_clean_text)
    reference = reference[reference[site_key].notna()].copy()
    if reference.empty:
        return {}, ["no matched rows with non-empty site key"]

    reference["_candidate_physical_site_id"] = reference[site_key].map(infer_site_code_base)
    mapping: dict[str, str] = {}
    ambiguities: list[str] = []

    name_column = first_existing_column(reference, ["representative_site_name", "site_name", "site_name_site"])
    country_column = first_existing_column(reference, ["country", "country_site", "location", "location_site"])
    latitude_column = first_existing_column(reference, ["latitude", "latitude_site"])
    longitude_column = first_existing_column(reference, ["longitude", "longitude_site"])

    for candidate_id, group in reference.groupby("_candidate_physical_site_id", dropna=True):
        codes = sorted(group[site_key].dropna().astype(str).unique())
        if len(codes) == 1:
            mapping[codes[0]] = codes[0]
            continue

        names = unique_clean_values(group[name_column]) if name_column else []
        normalized_names = unique_normalized_values(group[name_column]) if name_column else set()
        countries = unique_clean_values(group[country_column]) if country_column else []
        normalized_countries = unique_normalized_values(group[country_column]) if country_column else set()
        latitudes = pd.to_numeric(group[latitude_column], errors="coerce") if latitude_column else pd.Series(dtype=float)
        longitudes = pd.to_numeric(group[longitude_column], errors="coerce") if longitude_column else pd.Series(dtype=float)

        coordinate_signal = False
        coordinate_ambiguous = False
        if not latitudes.dropna().empty and not longitudes.dropna().empty:
            lat_range = float(latitudes.max() - latitudes.min())
            lon_range = float(longitudes.max() - longitudes.min())
            coordinate_signal = lat_range <= 0.01 and lon_range <= 0.01
            coordinate_ambiguous = lat_range > 0.05 or lon_range > 0.05

        name_signal = len(normalized_names) == 1
        should_collapse = coordinate_signal or name_signal

        if should_collapse:
            for code in codes:
                mapping[code] = str(candidate_id)
            if len(normalized_names) > 1:
                ambiguities.append(
                    f"physical_site_id={candidate_id}: collapsed {len(codes)} product/site codes with multiple parsed names: {joined_unique(names)}"
                )
            if len(normalized_countries) > 1:
                ambiguities.append(
                    f"physical_site_id={candidate_id}: collapsed {len(codes)} product/site codes with multiple parsed country/location values: {joined_unique(countries)}"
                )
            if coordinate_ambiguous:
                ambiguities.append(
                    f"physical_site_id={candidate_id}: collapsed {len(codes)} product/site codes with coordinate spread >0.05 degrees"
                )
        else:
            for code in codes:
                mapping[code] = code
            ambiguities.append(
                f"candidate_physical_site_id={candidate_id}: not collapsed because parsed names/coordinates did not support a shared physical location; codes={joined_unique(codes)}"
            )

    return mapping, ambiguities


def write_matched_site_diagnostics(
    tables_dir: Path,
    matches: pd.DataFrame,
    matched_products: pd.DataFrame,
    site_key: str,
) -> Path:
    diagnostics_csv = tables_dir / "matched_site_diagnostics.csv"
    output_columns = [
        "physical_site_id",
        "root_site_code_guess",
        "site_status",
        "site_warnings",
        "has_valid_coordinates",
        "has_rwl",
        "has_crn",
        "n_rwl_products",
        "n_crn_products",
        "n_metadata_files",
        "n_other_files",
        "temporal_warnings",
    ]

    if site_key not in matches.columns:
        pd.DataFrame(columns=output_columns).to_csv(diagnostics_csv, index=False, encoding="utf-8")
        return diagnostics_csv

    matched_site_rows = matches.drop(columns="geometry", errors="ignore").copy()
    mapping, _ = build_physical_site_id_map(matched_site_rows, site_key)
    matched_site_rows["_site_key_clean"] = matched_site_rows[site_key].map(lower_clean_text)
    matched_site_rows["physical_site_id"] = matched_site_rows["_site_key_clean"].map(mapping)

    products = matched_products.copy()
    if site_key in products.columns:
        products["_site_key_clean"] = products[site_key].map(lower_clean_text)
    else:
        products["_site_key_clean"] = pd.Series(dtype=object)

    site_status_column = first_existing_column(matched_site_rows, ["site_status", "site_status_site"])
    site_warnings_column = first_existing_column(matched_site_rows, ["site_warnings", "site_warnings_site"])
    has_rwl_column = first_existing_column(matched_site_rows, ["has_rwl", "has_rwl_site"])
    has_crn_column = first_existing_column(matched_site_rows, ["has_crn", "has_crn_site"])
    n_rwl_column = first_existing_column(matched_site_rows, ["n_rwl_products", "n_rwl_products_site"])
    n_crn_column = first_existing_column(matched_site_rows, ["n_crn_products", "n_crn_products_site"])
    n_metadata_column = first_existing_column(matched_site_rows, ["n_metadata_files", "n_metadata_files_site"])
    n_other_column = first_existing_column(matched_site_rows, ["n_other_files", "n_other_files_site"])
    first_year_column = first_existing_column(matched_site_rows, ["first_year_min", "first_year_min_site"])
    last_year_column = first_existing_column(matched_site_rows, ["last_year_max", "last_year_max_site"])
    latitude_column = first_existing_column(matched_site_rows, ["latitude", "latitude_site"])
    longitude_column = first_existing_column(matched_site_rows, ["longitude", "longitude_site"])

    diagnostic_rows = []
    for root_site_code, group in matched_site_rows.groupby("_site_key_clean", dropna=True):
        product_group = products.loc[products["_site_key_clean"].eq(root_site_code)].copy()
        physical_site_id = first_clean_value(group["physical_site_id"]) or infer_site_code_base(root_site_code)

        if latitude_column and longitude_column:
            latitudes = pd.to_numeric(group[latitude_column], errors="coerce")
            longitudes = pd.to_numeric(group[longitude_column], errors="coerce")
            has_valid_coordinates = bool(latitudes.notna().any() and longitudes.notna().any())
        else:
            has_valid_coordinates = True

        n_rwl_products = (
            int(numeric_max(group[n_rwl_column]) or 0)
            if n_rwl_column
            else count_product_type(product_group, "rwl")
        )
        n_crn_products = (
            int(numeric_max(group[n_crn_column]) or 0)
            if n_crn_column
            else count_product_type(product_group, "crn")
        )
        n_metadata_files = (
            int(numeric_max(group[n_metadata_column]) or 0)
            if n_metadata_column
            else count_truthy(product_group["has_metadata"])
            if "has_metadata" in product_group.columns
            else 0
        )
        n_other_files = (
            int(numeric_max(group[n_other_column]) or 0)
            if n_other_column
            else count_other_products(product_group)
        )
        first_year = (
            numeric_min(group[first_year_column])
            if first_year_column
            else numeric_min(product_group["first_year"])
            if "first_year" in product_group.columns
            else None
        )
        last_year = (
            numeric_max(group[last_year_column])
            if last_year_column
            else numeric_max(product_group["last_year"])
            if "last_year" in product_group.columns
            else None
        )

        diagnostic_rows.append(
            {
                "physical_site_id": physical_site_id,
                "root_site_code_guess": root_site_code,
                "site_status": first_clean_value(group[site_status_column]) if site_status_column else "",
                "site_warnings": first_clean_value(group[site_warnings_column]) if site_warnings_column else "",
                "has_valid_coordinates": has_valid_coordinates,
                "has_rwl": bool_any(group[has_rwl_column]) if has_rwl_column else n_rwl_products > 0,
                "has_crn": bool_any(group[has_crn_column]) if has_crn_column else n_crn_products > 0,
                "n_rwl_products": n_rwl_products,
                "n_crn_products": n_crn_products,
                "n_metadata_files": n_metadata_files,
                "n_other_files": n_other_files,
                "temporal_warnings": temporal_warnings_for_values(first_year, last_year),
            }
        )

    diagnostics = pd.DataFrame(diagnostic_rows)
    for column in output_columns:
        if column not in diagnostics.columns:
            diagnostics[column] = pd.NA
    diagnostics = diagnostics[output_columns].sort_values(["physical_site_id", "root_site_code_guess"])
    diagnostics.to_csv(diagnostics_csv, index=False, encoding="utf-8")
    return diagnostics_csv


def write_physical_site_outputs(
    tables_dir: Path,
    matches: pd.DataFrame,
    matched_products: pd.DataFrame,
    site_key: str,
    country_lookup=None,
) -> dict[str, object]:
    physical_sites_csv = tables_dir / "matched_physical_sites.csv"
    physical_summary_csv = tables_dir / "matched_physical_site_summary.csv"

    if site_key not in matches.columns:
        empty_sites = pd.DataFrame(
            columns=[
                "physical_site_id",
                "representative_site_name",
                "country",
                "country_source",
                "latitude",
                "longitude",
                "n_product_records",
                "n_rwl_products",
                "n_crn_products",
                "has_rwl",
                "has_crn",
                "first_year_min",
                "last_year_max",
                "min_distance_to_feature_km",
                "source_scope",
                "source_name",
                "temporal_warnings",
            ]
        )
        empty_sites.to_csv(physical_sites_csv, index=False, encoding="utf-8")
        pd.DataFrame(
            [
                {"metric": "total matched product records", "value": int(len(matched_products))},
                {"metric": "total matched physical locations", "value": 0},
                {"metric": "physical locations with RWL", "value": 0},
                {"metric": "physical locations with CRN", "value": 0},
                {"metric": "aggregation ambiguities", "value": 1},
                {"metric": "physical locations with temporal warnings", "value": 0},
            ]
        ).to_csv(physical_summary_csv, index=False, encoding="utf-8")
        return {
            "total_matched_product_records": int(len(matched_products)),
            "total_matched_physical_locations": 0,
            "physical_locations_with_rwl": 0,
            "physical_locations_with_crn": 0,
            "physical_site_country_counts": "",
            "physical_site_country_source_counts": "",
            "physical_site_temporal_warning_count": 0,
            "physical_site_ambiguities": f"missing required site key column: {site_key}",
            "physical_site_ambiguity_count": 1,
        }

    matched_site_rows = matches.drop(columns="geometry", errors="ignore").copy()
    mapping, ambiguities = build_physical_site_id_map(matched_site_rows, site_key)

    matched_site_rows["_site_key_clean"] = matched_site_rows[site_key].map(lower_clean_text)
    matched_site_rows["physical_site_id"] = matched_site_rows["_site_key_clean"].map(mapping)
    missing_physical_ids = int(matched_site_rows["physical_site_id"].isna().sum())
    if missing_physical_ids:
        ambiguities.append(f"{missing_physical_ids} matched site-feature rows could not be assigned a physical_site_id")

    products = matched_products.copy()
    if site_key in products.columns:
        products["_site_key_clean"] = products[site_key].map(lower_clean_text)
        products["physical_site_id"] = products["_site_key_clean"].map(mapping)
        fallback_mask = products["physical_site_id"].isna() & products["_site_key_clean"].notna()
        if fallback_mask.any():
            products.loc[fallback_mask, "physical_site_id"] = products.loc[fallback_mask, "_site_key_clean"].map(infer_site_code_base)
            ambiguities.append(
                f"{int(fallback_mask.sum())} matched product rows used regex fallback because no matched-site mapping was available"
            )
    else:
        products["physical_site_id"] = pd.Series(dtype=object)
        ambiguities.append(f"matched products lack required site key column: {site_key}")

    metadata_country_column = first_existing_column(
        matched_site_rows,
        ["country", "country_site", "metadata_country", "metadata_country_site"],
    )
    parsed_location_column = first_existing_column(matched_site_rows, ["location", "location_site"])
    name_column = first_existing_column(matched_site_rows, ["representative_site_name", "site_name", "site_name_site"])
    latitude_column = first_existing_column(matched_site_rows, ["latitude", "latitude_site"])
    longitude_column = first_existing_column(matched_site_rows, ["longitude", "longitude_site"])
    first_year_column = first_existing_column(matched_site_rows, ["first_year_min", "first_year_min_site"])
    last_year_column = first_existing_column(matched_site_rows, ["last_year_max", "last_year_max_site"])
    distance_column = first_existing_column(matched_site_rows, ["distance_to_feature_km", "min_distance_to_feature_km"])
    source_scope_column = first_existing_column(matched_site_rows, ["source_scope", "source_scope_feature"])
    source_name_column = first_existing_column(matched_site_rows, ["source_name", "source_name_feature"])

    physical_rows = []
    valid_site_rows = matched_site_rows[matched_site_rows["physical_site_id"].notna()].copy()
    for physical_site_id, group in valid_site_rows.groupby("physical_site_id", dropna=True):
        product_group = products.loc[products["physical_site_id"].eq(physical_site_id)].copy()
        source_scopes = unique_clean_values(group[source_scope_column]) if source_scope_column else []
        source_names = unique_clean_values(group[source_name_column]) if source_name_column else []
        metadata_countries = unique_clean_values(group[metadata_country_column]) if metadata_country_column else []
        parsed_locations = unique_clean_values(group[parsed_location_column]) if parsed_location_column else []
        names = unique_clean_values(group[name_column]) if name_column else []

        normalized_country_values = {
            value
            for value in (normalize_country_text(value) for value in metadata_countries + parsed_locations)
            if value is not None
        }
        if len(normalized_country_values) > 1:
            ambiguities.append(
                f"physical_site_id={physical_site_id}: multiple normalized country/location values in matched rows: {joined_unique(sorted(normalized_country_values))}"
            )
        if len({value.lower() for value in names}) > 1:
            ambiguities.append(
                f"physical_site_id={physical_site_id}: multiple representative names in matched rows: {joined_unique(names)}"
            )

        lat_value = numeric_min(group[latitude_column]) if latitude_column else None
        lon_value = numeric_min(group[longitude_column]) if longitude_column else None
        if latitude_column and longitude_column:
            latitudes = pd.to_numeric(group[latitude_column], errors="coerce")
            longitudes = pd.to_numeric(group[longitude_column], errors="coerce")
            if not latitudes.dropna().empty and not longitudes.dropna().empty:
                if float(latitudes.max() - latitudes.min()) > 0.05 or float(longitudes.max() - longitudes.min()) > 0.05:
                    ambiguities.append(
                        f"physical_site_id={physical_site_id}: coordinate spread exceeds 0.05 degrees in matched rows"
                    )

        n_product_records = int(product_group["product_id"].nunique()) if "product_id" in product_group.columns else int(len(product_group))
        n_rwl_products = count_product_type(product_group, "rwl")
        n_crn_products = count_product_type(product_group, "crn")
        country, country_source = resolve_country_for_group(
            group,
            metadata_country_column,
            parsed_location_column,
            latitude_column,
            longitude_column,
            country_lookup,
        )
        first_year = (
            numeric_min(group[first_year_column])
            if first_year_column
            else numeric_min(product_group["first_year"])
            if "first_year" in product_group.columns
            else None
        )
        last_year = (
            numeric_max(group[last_year_column])
            if last_year_column
            else numeric_max(product_group["last_year"])
            if "last_year" in product_group.columns
            else None
        )
        temporal_warnings = temporal_warnings_for_values(first_year, last_year)
        if temporal_warnings:
            ambiguities.append(f"physical_site_id={physical_site_id}: {temporal_warnings}")

        physical_rows.append(
            {
                "physical_site_id": physical_site_id,
                "representative_site_name": first_clean_value(group[name_column]) if name_column else "",
                "country": country,
                "country_source": country_source,
                "latitude": lat_value,
                "longitude": lon_value,
                "n_product_records": n_product_records,
                "n_rwl_products": n_rwl_products,
                "n_crn_products": n_crn_products,
                "has_rwl": bool(n_rwl_products > 0),
                "has_crn": bool(n_crn_products > 0),
                "first_year_min": first_year,
                "last_year_max": last_year,
                "min_distance_to_feature_km": numeric_min(group[distance_column]) if distance_column else None,
                "source_scope": joined_unique(source_scopes),
                "source_name": joined_unique(source_names),
                "temporal_warnings": temporal_warnings,
            }
        )

    physical_sites = pd.DataFrame(physical_rows).sort_values("physical_site_id") if physical_rows else pd.DataFrame()
    output_columns = [
        "physical_site_id",
        "representative_site_name",
        "country",
        "country_source",
        "latitude",
        "longitude",
        "n_product_records",
        "n_rwl_products",
        "n_crn_products",
        "has_rwl",
        "has_crn",
        "first_year_min",
        "last_year_max",
        "min_distance_to_feature_km",
        "source_scope",
        "source_name",
        "temporal_warnings",
    ]
    for column in output_columns:
        if column not in physical_sites.columns:
            physical_sites[column] = pd.NA
    physical_sites = physical_sites[output_columns]
    physical_sites["country"] = physical_sites["country"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    physical_sites["country_source"] = physical_sites["country_source"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    physical_sites.to_csv(physical_sites_csv, index=False, encoding="utf-8")

    country_counts = (
        physical_sites["country"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown").value_counts().sort_index()
        if not physical_sites.empty and "country" in physical_sites.columns
        else pd.Series(dtype=int)
    )
    country_source_counts = (
        physical_sites["country_source"].fillna("unknown").astype(str).str.strip().replace("", "unknown").value_counts().sort_index()
        if not physical_sites.empty and "country_source" in physical_sites.columns
        else pd.Series(dtype=int)
    )
    temporal_warning_count = (
        int(physical_sites["temporal_warnings"].fillna("").astype(str).str.strip().ne("").sum())
        if not physical_sites.empty and "temporal_warnings" in physical_sites.columns
        else 0
    )

    summary_rows = [
        {"metric": "total matched product records", "value": int(len(matched_products))},
        {"metric": "total matched physical locations", "value": int(len(physical_sites))},
        {"metric": "physical locations with RWL", "value": int(physical_sites["has_rwl"].fillna(False).astype(bool).sum())},
        {"metric": "physical locations with CRN", "value": int(physical_sites["has_crn"].fillna(False).astype(bool).sum())},
        {"metric": "aggregation ambiguities", "value": int(len(ambiguities))},
        {"metric": "physical locations with temporal warnings", "value": temporal_warning_count},
    ]
    summary_rows.extend(
        {"metric": f"physical locations in {country}", "value": int(count)}
        for country, count in country_counts.items()
    )
    summary_rows.extend(
        {"metric": f"country source {source}", "value": int(count)}
        for source, count in country_source_counts.items()
    )
    pd.DataFrame(summary_rows).to_csv(physical_summary_csv, index=False, encoding="utf-8")

    return {
        "total_matched_product_records": int(len(matched_products)),
        "total_matched_physical_locations": int(len(physical_sites)),
        "physical_locations_with_rwl": int(physical_sites["has_rwl"].fillna(False).astype(bool).sum()),
        "physical_locations_with_crn": int(physical_sites["has_crn"].fillna(False).astype(bool).sum()),
        "physical_site_country_counts": "; ".join(f"{country}: {count}" for country, count in country_counts.items()),
        "physical_site_country_source_counts": "; ".join(f"{source}: {count}" for source, count in country_source_counts.items()),
        "physical_site_temporal_warning_count": temporal_warning_count,
        "physical_site_ambiguities": joined_unique(ambiguities, limit=20),
        "physical_site_ambiguity_count": int(len(ambiguities)),
    }


def write_text_summary(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "Treequake Phase C Spatial Match Summary",
        "",
        f"ITRDB rows read: {summary['itrdb_rows_read']}",
        f"Coordinate-bearing rows: {summary['coordinate_bearing_rows']}",
        f"Rows lacking coordinates: {summary['rows_lacking_coordinates']}",
        f"Spatial features read: {summary['spatial_features_read']}",
        f"Buffer distance: {summary['buffer_km']} km",
        f"Total site-feature matches: {summary['total_site_feature_matches']}",
        f"Unique matched sites: {summary['unique_matched_sites']}",
        f"Countries represented: {summary['countries_represented']}",
        f"Earliest first year: {summary['earliest_first_year']}",
        f"Latest last year: {summary['latest_last_year']}",
        f"Number with standard RWL: {summary['number_with_standard_rwl']}",
        f"Number with CRN only: {summary['number_with_crn_only']}",
    ]
    if "total_matched_physical_locations" in summary:
        lines.extend(
            [
                "",
                "Physical-site aggregation",
                f"Total matched product records: {summary['total_matched_product_records']}",
                f"Total matched physical locations: {summary['total_matched_physical_locations']}",
                f"Physical locations with RWL: {summary['physical_locations_with_rwl']}",
                f"Physical locations with CRN: {summary['physical_locations_with_crn']}",
                f"Physical locations by country: {summary['physical_site_country_counts']}",
                f"Country source counts: {summary['physical_site_country_source_counts']}",
                f"Physical locations with temporal warnings: {summary['physical_site_temporal_warning_count']}",
                f"Aggregation ambiguities: {summary['physical_site_ambiguity_count']}",
                "",
                "Reporting note: matched physical locations are the preferred scientific reporting unit.",
                "Matched product records are provided to help users locate available ITRDB files.",
                "Treequake preserves matched records with missing species, country, elevation, RWL/CRN, or metadata/encoding issues and flags quality for later filtering.",
            ]
        )
        if summary.get("physical_site_ambiguities"):
            lines.extend(["Aggregation ambiguity details:", str(summary["physical_site_ambiguities"])])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    buffer_label = f"{args.buffer_km:g}".replace(".", "p")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = (
        project_root
        / "outputs"
        / "runs"
        / f"{timestamp}_spatial_match_buffer{buffer_label}km"
    )
    if run_dir.exists():
        counter = 2
        base_run_dir = run_dir
        while run_dir.exists():
            run_dir = base_run_dir.with_name(f"{base_run_dir.name}_{counter}")
            counter += 1
    tables_dir = run_dir / "tables"
    gis_dir = run_dir / "gis"
    tables_dir.mkdir(parents=True, exist_ok=True)
    gis_dir.mkdir(parents=True, exist_ok=True)

    features_path = args.spatial_features or project_root / "catalogs" / "normalized" / "spatial_features.gpkg"
    sites_path = args.itrdb_sites or project_root / "catalogs" / "normalized" / "itrdb_sites.gpkg"
    products_path = args.itrdb_products or project_root / "outputs" / "inventories" / "itrdb_products.csv"

    for path in [features_path, sites_path, products_path]:
        if not path.exists():
            raise SystemExit(f"Required input not found: {path}")

    gpd, CRS = require_geospatial_stack()

    print("Reading spatial features...")
    features, features_layer = read_first_layer(gpd, features_path, ["spatial_features", "features"])
    print("Reading ITRDB sites...")
    sites, sites_layer = read_first_layer(gpd, sites_path, ["itrdb_sites", "itrdb_sites_points"])
    print("Reading ITRDB products...")
    products = pd.read_csv(products_path, encoding="utf-8", low_memory=False)
    country_lookup = load_country_lookup(gpd, project_root)

    if sites.crs is None:
        raise SystemExit("ITRDB sites GeoPackage has no CRS.")
    if features.crs is None:
        raise SystemExit("Spatial features GeoPackage has no CRS.")

    sites = sites[sites.geometry.notna() & ~sites.geometry.is_empty].copy().to_crs(WGS84)
    features = features[features.geometry.notna() & ~features.geometry.is_empty].copy()
    analysis_polygons = build_analysis_polygons(gpd, CRS, features, args.buffer_km, args.polygon_buffer)

    if analysis_polygons.empty:
        matches = sites.iloc[0:0].copy()
    else:
        print("Running spatial join...")
        matches = gpd.sjoin(
            sites,
            analysis_polygons,
            how="inner",
            predicate="within",
            lsuffix="site",
            rsuffix="feature",
        )
        matches = matches.drop(columns=["index_feature"], errors="ignore")

    site_key = "root_site_code_guess"
    unique_site_codes = set(matches[site_key].dropna().astype(str)) if site_key in matches else set()
    matched_products = products[
        products[site_key].astype(str).isin(unique_site_codes)
    ].copy() if site_key in products else products.iloc[0:0].copy()

    matched_sites_csv = tables_dir / "matched_sites.csv"
    file_index_csv = tables_dir / "matched_sites_file_index.csv"
    site_diagnostics_csv = tables_dir / "matched_site_diagnostics.csv"
    physical_sites_csv = tables_dir / "matched_physical_sites.csv"
    physical_summary_csv = tables_dir / "matched_physical_site_summary.csv"
    summary_csv = tables_dir / "match_summary.csv"
    matched_sites_gpkg = gis_dir / "matched_sites.gpkg"
    analysis_polygons_gpkg = gis_dir / "analysis_polygons.gpkg"

    matches.drop(columns="geometry", errors="ignore").to_csv(matched_sites_csv, index=False, encoding="utf-8")
    matched_products.to_csv(file_index_csv, index=False, encoding="utf-8")
    write_matched_site_diagnostics(
        tables_dir=tables_dir,
        matches=matches,
        matched_products=matched_products,
        site_key=site_key,
    )
    physical_site_summary = write_physical_site_outputs(
        tables_dir=tables_dir,
        matches=matches,
        matched_products=matched_products,
        site_key=site_key,
        country_lookup=country_lookup,
    )
    analysis_polygons.to_file(analysis_polygons_gpkg, layer="analysis_polygons", driver="GPKG", index=False)
    matches.to_file(matched_sites_gpkg, layer="matched_sites", driver="GPKG", index=False)

    country_column = first_existing_column(matches, ["country", "country_site", "location", "location_site"])
    country_series = (
        matches[country_column].dropna().astype(str).str.strip()
        if country_column
        else pd.Series(dtype=object)
    )
    country_series = country_series[country_series != ""]
    country_counts = country_series.value_counts().sort_index()
    countries = list(country_counts.index)

    first_year_column = first_existing_column(matches, ["first_year_min", "first_year_min_site"])
    last_year_column = first_existing_column(matches, ["last_year_max", "last_year_max_site"])
    has_rwl_column = first_existing_column(matches, ["has_rwl", "has_rwl_site"])
    has_crn_column = first_existing_column(matches, ["has_crn", "has_crn_site"])

    if first_year_column:
        earliest_first_year = pd.to_numeric(matches[first_year_column], errors="coerce").min()
    else:
        earliest_first_year = pd.NA
    if last_year_column:
        latest_last_year = pd.to_numeric(matches[last_year_column], errors="coerce").max()
    else:
        latest_last_year = pd.NA

    rwl_sites = set(
        matches.loc[
            matches.get(has_rwl_column, pd.Series(False, index=matches.index)).fillna(False).astype(bool),
            site_key,
        ]
        .dropna()
        .astype(str)
    ) if site_key in matches and has_rwl_column else set()
    crn_sites = set(
        matches.loc[
            matches.get(has_crn_column, pd.Series(False, index=matches.index)).fillna(False).astype(bool),
            site_key,
        ]
        .dropna()
        .astype(str)
    ) if site_key in matches and has_crn_column else set()

    summary = {
        "itrdb_rows_read": int(len(sites) + 0),
        "coordinate_bearing_rows": int(len(sites)),
        "rows_lacking_coordinates": None,
        "spatial_features_read": int(len(features)),
        "features_layer": features_layer,
        "sites_layer": sites_layer,
        "buffer_km": args.buffer_km,
        "polygon_buffer": bool(args.polygon_buffer),
        "total_site_feature_matches": int(len(matches)),
        "unique_matched_sites": int(len(unique_site_codes)),
        "countries_represented": len(countries),
        "country_values": "; ".join(countries),
        "country_counts": "; ".join(f"{country}: {count}" for country, count in country_counts.items()),
        "earliest_first_year": None if pd.isna(earliest_first_year) else int(earliest_first_year),
        "latest_last_year": None if pd.isna(latest_last_year) else int(latest_last_year),
        "number_with_standard_rwl": len(rwl_sites),
        "number_with_crn_only": len(crn_sites - rwl_sites),
    }
    summary.update(physical_site_summary)

    all_rows_layer = None
    if "itrdb_sites_all_rows" in list_layers(sites_path):
        all_sites = gpd.read_file(sites_path, layer="itrdb_sites_all_rows")
        all_rows_layer = "itrdb_sites_all_rows"
        summary["itrdb_rows_read"] = int(len(all_sites))
        summary["rows_lacking_coordinates"] = int(len(all_sites) - len(sites))
    else:
        summary["rows_lacking_coordinates"] = 0

    pd.DataFrame([summary]).to_csv(summary_csv, index=False, encoding="utf-8")
    write_text_summary(run_dir / "summary.txt", summary)

    config = {
        "project_root": str(project_root),
        "spatial_features": str(features_path),
        "itrdb_sites": str(sites_path),
        "itrdb_products": str(products_path),
        "buffer_km": args.buffer_km,
        "polygon_buffer": bool(args.polygon_buffer),
        "features_layer": features_layer,
        "sites_layer": sites_layer,
        "all_rows_layer": all_rows_layer,
        "country_lookup_loaded": bool(country_lookup),
        "matched_site_diagnostics": str(site_diagnostics_csv),
        "matched_physical_sites": str(physical_sites_csv),
        "matched_physical_site_summary": str(physical_summary_csv),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config_used.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(f"Output folder: {run_dir}")
    print(f"ITRDB rows read: {summary['itrdb_rows_read']}")
    print(f"Coordinate-bearing rows: {summary['coordinate_bearing_rows']}")
    print(f"Rows lacking coordinates: {summary['rows_lacking_coordinates']}")
    print(f"Spatial features read: {summary['spatial_features_read']}")
    print(f"Buffer distance: {summary['buffer_km']} km")
    print(f"Total site-feature matches: {summary['total_site_feature_matches']}")
    print(f"Unique matched sites: {summary['unique_matched_sites']}")
    print(f"Countries represented: {summary['countries_represented']}")
    print(f"Matched physical locations: {summary['total_matched_physical_locations']}")
    print(f"Physical locations with RWL: {summary['physical_locations_with_rwl']}")
    print(f"Physical locations with CRN: {summary['physical_locations_with_crn']}")
    print(f"Physical locations with temporal warnings: {summary['physical_site_temporal_warning_count']}")
    print(f"Country source counts: {summary['physical_site_country_source_counts']}")
    print(f"Physical-site aggregation ambiguities: {summary['physical_site_ambiguity_count']}")


if __name__ == "__main__":
    main()
