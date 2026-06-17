#!/usr/bin/env python3
"""Build product-level and site-level inventories for locally downloaded ITRDB data.

This script reads the preserved raw NOAA/ITRDB archive layout produced by the
TREEQUAKE obtainer and writes reproducible CSV inventories plus a parse log.
It does not download data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PRODUCT_COLUMNS = [
    "product_id",
    "site_code_raw",
    "root_site_code_guess",
    "product_type",
    "variable_guess",
    "filename",
    "extension",
    "local_path",
    "has_metadata",
    "metadata_path",
    "first_year",
    "last_year",
    "species_code",
    "species_name",
    "site_name",
    "location",
    "latitude",
    "longitude",
    "elevation_m",
    "investigator",
    "study_name",
    "noaa_landing_page",
    "json_metadata_url",
    "parse_status",
    "parse_warnings",
]

SITE_COLUMNS = [
    "root_site_code_guess",
    "representative_site_name",
    "latitude",
    "longitude",
    "elevation_m",
    "location",
    "species_codes",
    "first_year_min",
    "last_year_max",
    "has_rwl",
    "has_crn",
    "n_rwl_products",
    "n_crn_products",
    "n_metadata_files",
    "n_other_files",
    "site_status",
    "site_warnings",
]

LOG_COLUMNS = ["level", "site_code_raw", "product_type", "filename", "message"]

METADATA_EXTENSIONS = {".txt", ".json", ".xml", ".html", ".htm", ".md"}
TEXT_EXTENSIONS = {".txt", ".rwl", ".crn", ".json", ".xml", ".html", ".htm", ".md"}
YEAR_RE = re.compile(r"(?<!\d)([12]\d{3}|0[5-9]\d{2})(?!\d)")
FIELD_RE = re.compile(r"^\s*#?\s*([A-Za-z][A-Za-z0-9_ /()-]{1,80})\s*[:=]\s*(.*?)\s*$")
SIGNED_COORD_RE = re.compile(r"([+-]\d{4,5})([+-]\d{5})")
UNSIGNED_LAT_SIGNED_LON_RE = re.compile(r"(?<!\d)(\d{4,5})([+-]\d{5})(?!\d)")

MEASUREMENT_TYPE_CODES = {
    "": "total_ring_width",
    "b": "blue_intensity",
    "ba": "basal_area_increment",
    "bm": "basal_area_mass_increment",
    "c": "average_cell_wall_thickness",
    "d": "total_ring_density",
    "e": "earlywood_width",
    "f": "average_microfibril_angle",
    "g": "earlywood_tracheid_diameter",
    "h": "tracheid_diameter",
    "i": "earlywood_density",
    "k": "latewood_tracheid_diameter",
    "l": "latewood_width",
    "n": "minimum_density",
    "p": "latewood_percent",
    "t": "latewood_density",
    "x": "maximum_density",
}
CHRONOLOGY_TYPE_CODES = {
    "": "standard_chronology",
    "a": "arstan_chronology",
    "p": "low_pass_filter_chronology",
    "r": "residual_chronology",
    "w": "re_whitened_residual_chronology",
}
MEASUREMENT_SUFFIXES = sorted((code for code in MEASUREMENT_TYPE_CODES if code), key=len, reverse=True)
CHRONOLOGY_SUFFIXES = sorted((code for code in CHRONOLOGY_TYPE_CODES if code), key=len, reverse=True)


FIELD_ALIASES = {
    "site_name": "site_name",
    "sitename": "site_name",
    "location": "location",
    "northernmost_latitude": "north_lat",
    "southernmost_latitude": "south_lat",
    "easternmost_longitude": "east_lon",
    "westernmost_longitude": "west_lon",
    "elevation_m": "elevation_m",
    "elevation": "elevation_m",
    "collection_name": "collection_name",
    "first_year": "first_year",
    "last_year": "last_year",
    "species_name": "species_name",
    "species_code": "species_code",
    "investigators": "investigator",
    "investigator": "investigator",
    "study_name": "study_name",
    "noaa_landing_page": "noaa_landing_page",
    "study_level_json_metadata": "json_metadata_url",
}


@dataclass
class MetadataRecord:
    path: Path | None = None
    values: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ProductRecord:
    row: dict[str, str]
    warnings: list[str] = field(default_factory=list)


@dataclass
class FilenameCodes:
    stem: str
    root_guess: str
    measurement_code: str = ""
    chronology_code: str = ""
    measurement_type: str = "total_ring_width"
    chronology_type: str = ""
    warnings: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ITRDB product and site inventories from local raw files."
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="TREEQUAKE project root. Paths are resolved relative to this directory.",
    )
    return parser.parse_args()


def safe_read_text(path: Path, max_bytes: int | None = None) -> str:
    with path.open("rb") as handle:
        data = handle.read(max_bytes) if max_bytes else handle.read()
    return data.decode("utf-8", errors="replace")


def norm_key(key: str) -> str:
    key = key.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    match = YEAR_RE.search(value)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def average_present(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def fmt_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8f}".rstrip("0").rstrip(".")
    return str(value)


def product_type_for(path: Path) -> str:
    ext = path.suffix.lower()
    parts = {part.lower() for part in path.parts}
    if ext == ".rwl" or "measurements" in parts:
        return "rwl" if ext == ".rwl" else ("metadata" if ext in METADATA_EXTENSIONS else "other")
    if ext == ".crn" or "chronologies" in parts:
        return "crn" if ext == ".crn" else ("metadata" if ext in METADATA_EXTENSIONS else "other")
    if "metadata" in parts or ext in METADATA_EXTENSIONS:
        return "metadata"
    return "other"


def variable_guess_for(product_type: str, filename_codes: FilenameCodes | None = None) -> str:
    if product_type == "rwl":
        measurement_type = filename_codes.measurement_type if filename_codes else "total_ring_width"
        return f"tree_or_core_level_measurements:{measurement_type}"
    if product_type == "crn":
        if filename_codes:
            parts = [f"site_level_chronology:{filename_codes.chronology_type or 'standard_chronology'}"]
            if filename_codes.measurement_type != "total_ring_width":
                parts.append(f"measurement_basis:{filename_codes.measurement_type}")
            return ";".join(parts)
        return "site_level_chronology:standard_chronology"
    if product_type == "metadata":
        return "metadata"
    return ""


def product_stem(path: Path) -> str:
    stem = path.stem
    if stem.lower().endswith("-noaa"):
        return stem[:-5]
    return stem


def parse_filename_codes(path: Path, product_type: str, fallback_site_code: str) -> FilenameCodes:
    stem = product_stem(path)
    root = stem or fallback_site_code
    lower = root.lower()
    warnings: list[str] = []
    measurement_code = ""
    chronology_code = ""

    if product_type == "crn":
        for code in CHRONOLOGY_SUFFIXES:
            if len(lower) > len(code) and lower.endswith(code):
                chronology_code = code
                root = root[: -len(code)]
                lower = lower[: -len(code)]
                warnings.append(
                    f"interpreted filename suffix '{code}' as {CHRONOLOGY_TYPE_CODES[code]}"
                )
                break
        for code in MEASUREMENT_SUFFIXES:
            if (
                len(lower) > len(code)
                and lower.endswith(code)
                and (chronology_code or any(ch.isdigit() for ch in lower[: -len(code)]))
            ):
                measurement_code = code
                root = root[: -len(code)]
                lower = lower[: -len(code)]
                warnings.append(
                    f"interpreted filename suffix '{code}' as {MEASUREMENT_TYPE_CODES[code]}"
                )
                break
    elif product_type == "rwl":
        for code in MEASUREMENT_SUFFIXES:
            if len(lower) > len(code) and lower.endswith(code) and any(ch.isdigit() for ch in lower[: -len(code)]):
                measurement_code = code
                root = root[: -len(code)]
                warnings.append(
                    f"interpreted filename suffix '{code}' as {MEASUREMENT_TYPE_CODES[code]}"
                )
                break

    if not root:
        root = fallback_site_code
        warnings.append("could not infer root site code from filename; used folder name")

    return FilenameCodes(
        stem=stem,
        root_guess=root,
        measurement_code=measurement_code,
        chronology_code=chronology_code,
        measurement_type=MEASUREMENT_TYPE_CODES.get(measurement_code, "total_ring_width"),
        chronology_type=CHRONOLOGY_TYPE_CODES.get(chronology_code, ""),
        warnings=warnings,
    )


def root_site_code_guess(site_code: str, path: Path | None, product_type: str, warnings: list[str]) -> str:
    clean = site_code.strip()
    if path and product_type in {"rwl", "crn"}:
        codes = parse_filename_codes(path, product_type, clean)
        warnings.extend(codes.warnings)
        if codes.root_guess and codes.root_guess.lower() != clean.lower():
            warnings.append(
                f"root_site_code_guess used filename-derived root '{codes.root_guess}' while preserving folder/product code '{clean}'"
            )
        return codes.root_guess or clean
    return clean


def product_id_for(root: str, product_type: str, path: Path) -> str:
    token = f"{root}|{product_type}|{path.as_posix()}"
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
    return f"{root}_{product_type}_{path.stem}_{digest}"


def site_code_from_path(raw_root: Path, path: Path) -> str:
    rel = path.relative_to(raw_root)
    if len(rel.parts) >= 1:
        return rel.parts[0]
    return path.stem


def parse_metadata_file(path: Path) -> MetadataRecord:
    record = MetadataRecord(path=path)
    try:
        text = safe_read_text(path, max_bytes=2_000_000)
    except OSError as exc:
        record.warnings.append(f"metadata read failed: {exc}")
        return record

    for line in text.splitlines():
        match = FIELD_RE.match(line)
        if not match:
            continue
        key, value = match.groups()
        alias = FIELD_ALIASES.get(norm_key(key))
        if alias and value:
            record.values.setdefault(alias, value.strip())

    north = parse_float(record.values.get("north_lat"))
    south = parse_float(record.values.get("south_lat"))
    east = parse_float(record.values.get("east_lon"))
    west = parse_float(record.values.get("west_lon"))
    latitude = average_present([north, south])
    longitude = average_present([east, west])
    if latitude is not None:
        record.values["latitude"] = fmt_number(latitude)
    if longitude is not None:
        record.values["longitude"] = fmt_number(longitude)

    elevation = parse_float(record.values.get("elevation_m"))
    if elevation is not None:
        record.values["elevation_m"] = fmt_number(elevation)

    first_year = parse_int(record.values.get("first_year"))
    last_year = parse_int(record.values.get("last_year"))
    if first_year is not None:
        record.values["first_year"] = str(first_year)
    if last_year is not None:
        record.values["last_year"] = str(last_year)

    if not record.values:
        record.warnings.append("no recognized NOAA metadata fields found")
    return record


def parse_ddmm_token(token: str, is_longitude: bool) -> float | None:
    token = token.strip()
    if not token:
        return None
    sign = -1 if token.startswith("-") else 1
    digits = token.lstrip("+-")
    if not digits.isdigit() or len(digits) < 4:
        return None
    degree_digits = 3 if is_longitude else 2
    if len(digits) == 5 and not is_longitude:
        degree_digits = 3
    if len(digits) <= degree_digits:
        return None
    degrees = int(digits[:degree_digits])
    minutes = int(digits[degree_digits:])
    if minutes >= 60:
        return None
    return sign * (degrees + minutes / 60.0)


def parse_tucson_lat_lon(value: str) -> tuple[float | None, float | None]:
    def valid_pair(lat_value: float | None, lon_value: float | None) -> tuple[float | None, float | None]:
        if lat_value is None or lon_value is None:
            return None, None
        if not (-90 <= lat_value <= 90 and -180 <= lon_value <= 180):
            return None, None
        return lat_value, lon_value

    compact = re.sub(r"\s+", "", value or "")
    match = SIGNED_COORD_RE.search(compact)
    if match:
        return valid_pair(parse_ddmm_token(match.group(1), False), parse_ddmm_token(match.group(2), True))
    match = UNSIGNED_LAT_SIGNED_LON_RE.search(compact)
    if match:
        return valid_pair(parse_ddmm_token(match.group(1), False), parse_ddmm_token(match.group(2), True))

    tokens = re.findall(r"[+-]?\d{4,5}", compact)
    if len(tokens) >= 2:
        return valid_pair(parse_ddmm_token(tokens[0], False), parse_ddmm_token(tokens[1], True))
    return None, None


def read_initial_lines(path: Path, max_lines: int = 12) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    try:
        text = safe_read_text(path, max_bytes=20_000)
    except OSError as exc:
        return [], [f"product header read failed: {exc}"]
    lines = [line.rstrip("\n\r") for line in text.splitlines() if line.strip()]
    return lines[:max_lines], warnings


def parse_tucson_product_header(path: Path) -> MetadataRecord:
    record = MetadataRecord(path=None)
    if path.suffix.lower() not in {".rwl", ".crn"}:
        return record

    lines, warnings = read_initial_lines(path)
    record.warnings.extend(warnings)
    if len(lines) < 3:
        record.warnings.append("fewer than 3 nonblank lines available for Tucson header parsing")
        return record

    header_lines: dict[str, str] = {}
    for line in lines[:8]:
        marker = line[6:9].strip() if len(line) >= 9 else ""
        if marker in {"1", "2", "3"}:
            header_lines.setdefault(marker, line)
    if len(header_lines) < 3:
        header_lines = {"1": lines[0], "2": lines[1], "3": lines[2]}
        record.warnings.append("Tucson header line markers not found; parsed first three nonblank lines")

    line1 = header_lines.get("1", "")
    line2 = header_lines.get("2", "")
    line3 = header_lines.get("3", "")

    def slice_text(line: str, start: int, end: int) -> str:
        return line[start:end].strip()

    values: dict[str, str] = {}
    if slice_text(line1, 9, 61):
        values["site_name"] = slice_text(line1, 9, 61)
    if slice_text(line1, 61, 65):
        values["species_code"] = slice_text(line1, 61, 65)
    if slice_text(line2, 9, 22):
        values["location"] = slice_text(line2, 9, 22)
    if slice_text(line2, 22, 30):
        values["species_name"] = slice_text(line2, 22, 30)

    elevation = parse_float(slice_text(line2, 40, 45))
    if elevation is not None:
        values["elevation_m"] = fmt_number(elevation)

    lat, lon = parse_tucson_lat_lon(slice_text(line2, 47, 57))
    if lat is None or lon is None:
        lat, lon = parse_tucson_lat_lon(line2)
    if lat is not None:
        values["latitude"] = fmt_number(lat)
    if lon is not None:
        values["longitude"] = fmt_number(lon)

    years = [int(match.group(1)) for match in YEAR_RE.finditer(slice_text(line2, 67, 80))]
    if len(years) < 2:
        years = [int(match.group(1)) for match in YEAR_RE.finditer(line2)]
        if len(years) > 2:
            years = years[-2:]
    if len(years) >= 2:
        values["first_year"] = str(years[0])
        values["last_year"] = str(years[1])
    elif len(years) == 1:
        values["first_year"] = str(years[0])

    if slice_text(line3, 9, 72):
        values["investigator"] = slice_text(line3, 9, 72)

    if not values:
        record.warnings.append("no recognized Tucson header fields found")
    record.values = values
    return record


def merge_missing(primary: dict[str, str], fallback: dict[str, str]) -> dict[str, str]:
    merged = primary.copy()
    for key, value in fallback.items():
        if value and not merged.get(key):
            merged[key] = value
    return merged


def discover_files(raw_root: Path) -> list[Path]:
    if not raw_root.exists():
        return []
    return sorted(path for path in raw_root.rglob("*") if path.is_file())


def build_metadata_index(raw_root: Path, files: list[Path]) -> dict[str, list[MetadataRecord]]:
    index: dict[str, list[MetadataRecord]] = defaultdict(list)
    for path in files:
        if product_type_for(path) != "metadata":
            continue
        site_code = site_code_from_path(raw_root, path)
        index[site_code].append(parse_metadata_file(path))
    return index


def choose_metadata(
    site_code: str,
    product_path: Path,
    metadata_index: dict[str, list[MetadataRecord]],
) -> MetadataRecord:
    candidates = metadata_index.get(site_code, [])
    if not candidates:
        return MetadataRecord()
    stem = product_path.stem.lower()
    for candidate in candidates:
        if candidate.path and candidate.path.stem.lower() == stem:
            return candidate
    for candidate in candidates:
        if candidate.path and (stem in candidate.path.stem.lower() or candidate.path.stem.lower() in stem):
            return candidate
    return candidates[0]


def scan_years_from_product(path: Path) -> tuple[int | None, int | None, list[str]]:
    warnings: list[str] = []
    if path.suffix.lower() not in {".rwl", ".crn"}:
        return None, None, warnings
    try:
        text = safe_read_text(path, max_bytes=1_500_000)
    except OSError as exc:
        return None, None, [f"product read failed: {exc}"]
    years = [int(match.group(1)) for match in YEAR_RE.finditer(text)]
    plausible = [year for year in years if 500 <= year <= 2026]
    if not plausible:
        warnings.append("no plausible year range found in product text")
        return None, None, warnings
    return min(plausible), max(plausible), warnings


def build_product_rows(raw_root: Path, files: list[Path]) -> tuple[list[ProductRecord], list[dict[str, str]], str]:
    metadata_index = build_metadata_index(raw_root, files)
    product_records: list[ProductRecord] = []
    log_rows: list[dict[str, str]] = []
    inspected_note = observed_structure_note(raw_root, files)

    for path in files:
        site_code = site_code_from_path(raw_root, path)
        warnings: list[str] = []
        product_type = product_type_for(path)
        filename_codes = parse_filename_codes(path, product_type, site_code)
        root_guess = root_site_code_guess(site_code, path, product_type, warnings)
        metadata = choose_metadata(site_code, path, metadata_index)
        product_header = parse_tucson_product_header(path)
        values = merge_missing(metadata.values, product_header.values)
        warnings.extend(metadata.warnings)
        warnings.extend(product_header.warnings)
        warnings.extend(filename_codes.warnings)

        product_first, product_last, product_warnings = scan_years_from_product(path)
        warnings.extend(product_warnings)
        first_year = parse_int(values.get("first_year")) or product_first
        last_year = parse_int(values.get("last_year")) or product_last

        if product_type in {"rwl", "crn"} and not metadata.path:
            warnings.append("no metadata file associated with product")
        if not values.get("latitude") or not values.get("longitude"):
            warnings.append("missing coordinates; unusable for spatial matching until resolved")
        if first_year is None or last_year is None:
            warnings.append("missing year range")
        if product_type == "metadata" and not values:
            warnings.append("metadata file did not yield recognized fields")

        parse_status = "ok" if not warnings else "warning"
        row = {
            "product_id": product_id_for(root_guess, product_type, path.relative_to(raw_root)),
            "site_code_raw": site_code,
            "root_site_code_guess": root_guess,
            "product_type": product_type,
            "variable_guess": variable_guess_for(product_type, filename_codes),
            "filename": path.name,
            "extension": path.suffix.lower().lstrip("."),
            "local_path": path.as_posix(),
            "has_metadata": "true" if metadata.path else "false",
            "metadata_path": metadata.path.as_posix() if metadata.path else "",
            "first_year": str(first_year) if first_year is not None else "",
            "last_year": str(last_year) if last_year is not None else "",
            "species_code": values.get("species_code", ""),
            "species_name": values.get("species_name", ""),
            "site_name": values.get("site_name", "") or values.get("collection_name", ""),
            "location": values.get("location", ""),
            "latitude": values.get("latitude", ""),
            "longitude": values.get("longitude", ""),
            "elevation_m": values.get("elevation_m", ""),
            "investigator": values.get("investigator", ""),
            "study_name": values.get("study_name", ""),
            "noaa_landing_page": values.get("noaa_landing_page", ""),
            "json_metadata_url": values.get("json_metadata_url", ""),
            "parse_status": parse_status,
            "parse_warnings": "; ".join(dict.fromkeys(warnings)),
        }
        record = ProductRecord(row=row, warnings=warnings)
        product_records.append(record)

        if warnings:
            for warning in dict.fromkeys(warnings):
                log_rows.append(
                    {
                        "level": "warning",
                        "site_code_raw": site_code,
                        "product_type": product_type,
                        "filename": path.name,
                        "message": warning,
                    }
                )

    known_sites = {site_code_from_path(raw_root, path) for path in files}
    if raw_root.exists():
        for child in sorted(path for path in raw_root.iterdir() if path.is_dir()):
            site_code = child.name
            if site_code not in known_sites:
                warnings = ["folder contains no files under recognized raw-data root"]
                root_guess = root_site_code_guess(site_code, None, "other", warnings)
                row = empty_folder_product_row(child, site_code, root_guess, warnings)
                product_records.append(ProductRecord(row=row, warnings=warnings))
                for warning in warnings:
                    log_rows.append(
                        {
                            "level": "warning",
                            "site_code_raw": site_code,
                            "product_type": "other",
                            "filename": "",
                            "message": warning,
                        }
                    )
    else:
        log_rows.append(
            {
                "level": "error",
                "site_code_raw": "",
                "product_type": "",
                "filename": "",
                "message": f"raw ITRDB directory not found: {raw_root}",
            }
        )

    return product_records, log_rows, inspected_note


def empty_folder_product_row(path: Path, site_code: str, root_guess: str, warnings: list[str]) -> dict[str, str]:
    return {
        "product_id": product_id_for(root_guess, "other", Path(site_code)),
        "site_code_raw": site_code,
        "root_site_code_guess": root_guess,
        "product_type": "other",
        "variable_guess": "",
        "filename": "",
        "extension": "",
        "local_path": path.as_posix(),
        "has_metadata": "false",
        "metadata_path": "",
        "first_year": "",
        "last_year": "",
        "species_code": "",
        "species_name": "",
        "site_name": "",
        "location": "",
        "latitude": "",
        "longitude": "",
        "elevation_m": "",
        "investigator": "",
        "study_name": "",
        "noaa_landing_page": "",
        "json_metadata_url": "",
        "parse_status": "warning",
        "parse_warnings": "; ".join(warnings),
    }


def first_nonempty(rows: list[dict[str, str]], key: str) -> str:
    for row in rows:
        value = row.get(key, "")
        if value:
            return value
    return ""


def min_int(rows: list[dict[str, str]], key: str) -> str:
    values = [int(row[key]) for row in rows if row.get(key, "").isdigit()]
    return str(min(values)) if values else ""


def max_int(rows: list[dict[str, str]], key: str) -> str:
    values = [int(row[key]) for row in rows if row.get(key, "").isdigit()]
    return str(max(values)) if values else ""


def build_site_rows(product_records: list[ProductRecord]) -> list[dict[str, str]]:
    grouped: dict[str, list[ProductRecord]] = defaultdict(list)
    for record in product_records:
        grouped[record.row["root_site_code_guess"]].append(record)

    site_rows: list[dict[str, str]] = []
    for root_guess, records in sorted(grouped.items()):
        rows = [record.row for record in records]
        type_counts = Counter(row["product_type"] for row in rows)
        has_rwl = type_counts["rwl"] > 0
        has_crn = type_counts["crn"] > 0
        if has_rwl and has_crn:
            site_status = "rwl_and_crn"
        elif has_rwl:
            site_status = "rwl_only"
        elif has_crn:
            site_status = "crn_only"
        elif type_counts["metadata"] > 0 and type_counts["other"] == 0:
            site_status = "metadata_only"
        elif type_counts["other"] > 0 or rows:
            site_status = "other_only"
        else:
            site_status = "broken_or_empty"

        warnings = []
        for record in records:
            warnings.extend(record.warnings)
        if not has_rwl and not has_crn:
            warnings.append("site has neither RWL nor CRN product")
        if not first_nonempty(rows, "latitude") or not first_nonempty(rows, "longitude"):
            warnings.append("site missing coordinates")

        species_codes = sorted({row["species_code"] for row in rows if row.get("species_code")})
        site_rows.append(
            {
                "root_site_code_guess": root_guess,
                "representative_site_name": first_nonempty(rows, "site_name"),
                "latitude": first_nonempty(rows, "latitude"),
                "longitude": first_nonempty(rows, "longitude"),
                "elevation_m": first_nonempty(rows, "elevation_m"),
                "location": first_nonempty(rows, "location"),
                "species_codes": "|".join(species_codes),
                "first_year_min": min_int(rows, "first_year"),
                "last_year_max": max_int(rows, "last_year"),
                "has_rwl": "true" if has_rwl else "false",
                "has_crn": "true" if has_crn else "false",
                "n_rwl_products": str(type_counts["rwl"]),
                "n_crn_products": str(type_counts["crn"]),
                "n_metadata_files": str(type_counts["metadata"]),
                "n_other_files": str(type_counts["other"]),
                "site_status": site_status,
                "site_warnings": "; ".join(dict.fromkeys(warnings)),
            }
        )
    return site_rows


def observed_structure_note(raw_root: Path, files: list[Path]) -> str:
    metadata_files = [path for path in files if product_type_for(path) == "metadata"]
    chronology_files = [path for path in files if product_type_for(path) == "crn"]
    measurement_files = [path for path in files if product_type_for(path) == "rwl"]

    lines = [
        "Observed NOAA/local archive structure note:",
        f"- Raw root inspected: {raw_root}",
        f"- Metadata files available for inspection: {len(metadata_files)}",
        f"- Chronology (.crn) files available for inspection: {len(chronology_files)}",
        f"- Measurement (.rwl) files available for inspection: {len(measurement_files)}",
    ]
    if not files:
        lines.append("- No local raw files were available in this run, so the required 20/20/sample inspection could not be completed.")
        return "\n".join(lines)

    sample_paths = {
        "metadata": metadata_files[:20],
        "chronologies": chronology_files[:20],
        "measurements": measurement_files[:20],
    }
    for label, paths in sample_paths.items():
        lines.append(f"- Sample inspected {label}: {len(paths)}")
        for path in paths[:5]:
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            lines.append(f"  - {path.relative_to(raw_root)} ({size} bytes)")

    classes_by_folder = Counter()
    for path in files:
        rel = path.relative_to(raw_root)
        if len(rel.parts) >= 2:
            classes_by_folder[rel.parts[1]] += 1
    if classes_by_folder:
        lines.append("- Second-level folder/file-class counts observed: " + ", ".join(f"{k}={v}" for k, v in classes_by_folder.most_common()))
    return "\n".join(lines)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    raw_root: Path,
    product_rows: list[dict[str, str]],
    site_rows: list[dict[str, str]],
    log_rows: list[dict[str, str]],
    observed_note: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    product_counts = Counter(row["product_type"] for row in product_rows)
    site_counts = Counter(row["site_status"] for row in site_rows)
    folders_scanned = sum(1 for item in raw_root.iterdir() if item.is_dir()) if raw_root.exists() else 0
    coordinate_success = sum(1 for row in site_rows if row["latitude"] and row["longitude"])
    missing_coordinates = len(site_rows) - coordinate_success
    neither = sum(1 for row in site_rows if row["has_rwl"] == "false" and row["has_crn"] == "false")
    warning_counts = Counter(row["message"] for row in log_rows)

    lines = [
        observed_note,
        "",
        "Inventory summary:",
        f"total folders scanned: {folders_scanned}",
        f"total products: {len(product_rows)}",
        f"total RWL products: {product_counts['rwl']}",
        f"total CRN products: {product_counts['crn']}",
        f"total metadata files: {product_counts['metadata']}",
        f"total site rows: {len(site_rows)}",
        f"sites with RWL and CRN: {site_counts['rwl_and_crn']}",
        f"sites with RWL only: {site_counts['rwl_only']}",
        f"sites with CRN only: {site_counts['crn_only']}",
        f"sites with neither RWL nor CRN: {neither}",
        f"sites with valid coordinates: {coordinate_success}",
        f"sites missing coordinates: {missing_coordinates}",
        "",
        "sample 20 site rows:",
    ]
    for row in site_rows[:20]:
        lines.append(
            f"- {row['root_site_code_guess']}: status={row['site_status']}, "
            f"rwl={row['n_rwl_products']}, crn={row['n_crn_products']}, "
            f"coords=({row['latitude']}, {row['longitude']}), name={row['representative_site_name']}"
        )
    lines.append("")
    lines.append("sample 20 products:")
    for row in product_rows[:20]:
        lines.append(
            f"- {row['product_id']}: type={row['product_type']}, file={row['filename']}, "
            f"site={row['site_code_raw']}, years={row['first_year']}-{row['last_year']}, status={row['parse_status']}"
        )
    lines.append("")
    lines.append("top 20 parse warnings:")
    if warning_counts:
        for warning, count in warning_counts.most_common(20):
            lines.append(f"- {count}: {warning}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    raw_root = project_root / "data" / "raw" / "itrdb"
    inventory_dir = project_root / "outputs" / "inventories"
    log_dir = project_root / "outputs" / "logs"

    files = discover_files(raw_root)
    product_records, log_rows, observed_note = build_product_rows(raw_root, files)
    product_rows = [record.row for record in product_records]
    site_rows = build_site_rows(product_records)

    products_path = inventory_dir / "itrdb_products.csv"
    sites_path = inventory_dir / "itrdb_sites.csv"
    log_path = log_dir / "itrdb_inventory_parse_log.csv"
    summary_path = log_dir / "itrdb_inventory_summary.txt"

    write_csv(products_path, PRODUCT_COLUMNS, product_rows)
    write_csv(sites_path, SITE_COLUMNS, site_rows)
    write_csv(log_path, LOG_COLUMNS, log_rows)
    write_summary(summary_path, raw_root, product_rows, site_rows, log_rows, observed_note)

    print(f"Wrote {len(product_rows)} products to {products_path}")
    print(f"Wrote {len(site_rows)} sites to {sites_path}")
    print(f"Wrote {len(log_rows)} log rows to {log_path}")
    print(f"Wrote summary to {summary_path}")

    if not raw_root.exists():
        print(f"ERROR: raw ITRDB directory not found: {raw_root}", file=sys.stderr)
        return 2
    if len(product_rows) <= 100 or len(site_rows) <= 100:
        print(
            "WARNING: inventory row counts are below pass criteria; this may reflect an incomplete current download.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
