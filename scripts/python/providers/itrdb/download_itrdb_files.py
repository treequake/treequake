#!/usr/bin/env python3
"""Discover and optionally download NOAA/ITRDB tree-ring files.

The default mode is safe discovery only. Use --download for real file downloads.
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen


BASE_URLS = {
    "measurements": "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/",
    "chronologies": "https://www.ncei.noaa.gov/pub/data/paleo/treering/chronologies/",
}

DATA_EXTENSIONS = {
    ".rwl",
    ".crn",
    ".txt",
    ".csv",
    ".json",
    ".xml",
    ".readme",
    ".doc",
    ".docx",
    ".pdf",
    ".zip",
}

METADATA_EXTENSIONS = {".txt", ".csv", ".json", ".xml", ".readme", ".doc", ".docx", ".pdf"}
ROOT_DEBUG_FILES = {
    "measurements": "itrdb_debug_root_listing_measurements.html",
    "chronologies": "itrdb_debug_root_listing_chronologies.html",
}
DEFAULT_USER_AGENT = "TREEQUAKE ITRDB Obtainer/0.2 (+https://www.ncei.noaa.gov/)"


@dataclass(frozen=True)
class FetchResult:
    url: str
    status: int | None
    body: bytes
    content_type: str
    error: str = ""


@dataclass(frozen=True)
class DiscoveredFile:
    url: str
    base_class: str


class LinkEntry(NamedTuple):
    href: str
    text: str


class LinkParser(HTMLParser):
    """Extract links from simple Apache-style directory listings."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[LinkEntry] = []
        self._current_href: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self._current_href = html.unescape(value.strip())
                self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._current_href is None:
            return
        text = html.unescape("".join(self._current_text).strip())
        self.links.append(LinkEntry(self._current_href, text))
        self._current_href = None
        self._current_text = []


def normalize_dir_url(url: str) -> str:
    clean, _fragment = urldefrag(url.strip())
    return clean if clean.endswith("/") else f"{clean}/"


def normalize_file_url(url: str) -> str:
    clean, _fragment = urldefrag(url.strip())
    return clean


def fetch_url(url: str, timeout: int, user_agent: str, retries: int, pause: float) -> FetchResult:
    last_error = ""
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": user_agent})
            with urlopen(request, timeout=timeout) as response:
                body = response.read()
                return FetchResult(
                    url=response.geturl(),
                    status=getattr(response, "status", None),
                    body=body,
                    content_type=response.headers.get("content-type", ""),
                )
        except HTTPError as exc:
            body = exc.read()
            return FetchResult(
                url=url,
                status=exc.code,
                body=body,
                content_type=exc.headers.get("content-type", "") if exc.headers else "",
                error=f"HTTP {exc.code}: {exc.reason}",
            )
        except (TimeoutError, URLError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(pause * (attempt + 1))

    return FetchResult(url=url, status=None, body=b"", content_type="", error=last_error)


def extract_links(base_url: str, body: bytes) -> list[str]:
    text = body.decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(text)

    links: list[str] = []
    for href, link_text in parser.links:
        if not href or href.startswith("#") or href.startswith("?"):
            continue
        if href in {".", "./", "..", "../"} or link_text.lower() == "parent directory":
            continue
        absolute = normalize_file_url(urljoin(base_url, href))
        name = Path(unquote(urlparse(absolute).path)).name
        if name in {"", ".", ".."}:
            continue
        if link_text.endswith("/") and not urlparse(absolute).path.endswith("/"):
            absolute = normalize_dir_url(absolute)
        links.append(absolute)

    return sorted(set(links))


def extension_from_url(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name.lower()
    if name.endswith(".readme"):
        return ".readme"
    return Path(name).suffix.lower()


def is_data_file(url: str) -> bool:
    return extension_from_url(url) in DATA_EXTENSIONS


def is_directory_link(url: str) -> bool:
    return urlparse(url).path.endswith("/")


def inside_base(url: str, base_url: str) -> bool:
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    return (
        parsed_url.scheme == parsed_base.scheme
        and parsed_url.netloc == parsed_base.netloc
        and parsed_url.path.startswith(parsed_base.path)
    )


def path_parts_under_base(url: str, base_url: str) -> list[str]:
    path = unquote(urlparse(url).path)
    base_path = unquote(urlparse(base_url).path)
    if not path.startswith(base_path):
        return []
    rel = path[len(base_path) :].strip("/")
    return [part for part in rel.split("/") if part]


def site_code_from_filename(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0].lower()
    for suffix in ("-rwl-noaa", "-crn-noaa", "-noaa"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem or "unknown"


def classify_file(url: str, base_class: str) -> str:
    ext = extension_from_url(url)
    if ext == ".rwl":
        return "measurements"
    if ext == ".crn":
        return "chronologies"
    if ext in METADATA_EXTENSIONS:
        return "metadata"
    if base_class in {"measurements", "chronologies"} and ext in DATA_EXTENSIONS:
        return base_class
    return "other"


def region_country_guess(url: str, base_url: str) -> tuple[str, str]:
    parts = path_parts_under_base(url, base_url)
    if not parts:
        return "", ""
    region = parts[0] if len(parts) >= 2 else ""
    country = parts[1] if len(parts) >= 3 else ""
    return region, country


def crawl_base(
    base_class: str,
    base_url: str,
    log_dir: Path,
    max_depth: int,
    timeout: int,
    retries: int,
    pause: float,
    user_agent: str,
) -> tuple[list[DiscoveredFile], list[dict[str, str]]]:
    base_url = normalize_dir_url(base_url)
    queue: deque[tuple[str, int]] = deque([(base_url, 0)])
    visited: set[str] = set()
    discovered: set[str] = set()
    files: list[DiscoveredFile] = []
    directory_rows: list[dict[str, str]] = []

    while queue:
        directory_url, depth = queue.popleft()
        directory_url = normalize_dir_url(directory_url)
        if directory_url in visited or depth > max_depth:
            continue
        if not inside_base(directory_url, base_url):
            continue

        visited.add(directory_url)
        result = fetch_url(directory_url, timeout, user_agent, retries, pause)
        is_root = directory_url == base_url
        if is_root:
            debug_path = log_dir / ROOT_DEBUG_FILES[base_class]
            debug_path.write_bytes(result.body)
            (debug_path.with_suffix(debug_path.suffix + ".headers")).write_text(
                f"url: {result.url}\nstatus: {result.status}\n"
                f"content-type: {result.content_type}\nerror: {result.error}\n",
                encoding="utf-8",
            )

        row = {
            "directory_url": directory_url,
            "base_class": base_class,
            "depth": str(depth),
            "status": "ok" if result.status and 200 <= result.status < 400 and not result.error else "failed",
            "http_status": "" if result.status is None else str(result.status),
            "error": result.error,
            "link_count": "0",
        }

        if row["status"] != "ok":
            directory_rows.append(row)
            continue

        links = extract_links(directory_url, result.body)
        row["link_count"] = str(len(links))
        directory_rows.append(row)

        for link in links:
            if not inside_base(link, base_url):
                continue
            if is_directory_link(link):
                child = normalize_dir_url(link)
                if child not in visited:
                    queue.append((child, depth + 1))
            elif is_data_file(link):
                file_url = normalize_file_url(link)
                if file_url not in discovered:
                    discovered.add(file_url)
                    files.append(DiscoveredFile(file_url, base_class))

        if pause:
            time.sleep(pause)

    return sorted(files, key=lambda item: item.url), directory_rows


def local_path_for(url: str, base_class: str, data_root: Path) -> Path:
    filename = Path(unquote(urlparse(url).path)).name
    site_code = site_code_from_filename(filename)
    file_class = classify_file(url, base_class)
    return data_root / site_code / file_class / filename


def download_file(
    url: str,
    destination: Path,
    overwrite: bool,
    timeout: int,
    retries: int,
    pause: float,
    user_agent: str,
) -> tuple[str, int]:
    if destination.exists() and not overwrite:
        return "exists", destination.stat().st_size

    result = fetch_url(url, timeout, user_agent, retries, pause)
    if result.error or not result.status or not (200 <= result.status < 400):
        return f"failed: {result.error or result.status}", 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(result.body)
    return "downloaded", len(result.body)


def write_csv(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validation_lines(
    inventory_rows: list[dict[str, str]],
    directory_rows: list[dict[str, str]],
    sample_count: int = 20,
) -> tuple[bool, list[str]]:
    ext_counts = Counter(row["extension"] for row in inventory_rows)
    class_counts = Counter(row["file_class"] for row in inventory_rows)
    crawled_region_dirs = {
        row["directory_url"]
        for row in directory_rows
        if row["status"] == "ok" and int(row["depth"] or "0") >= 1
    }
    failures = [row for row in directory_rows if row["status"] != "ok"]

    checks = {
        "at least 100 total files discovered": len(inventory_rows) >= 100,
        "at least 50 .rwl files discovered": ext_counts[".rwl"] >= 50,
        "at least 50 .crn files discovered": ext_counts[".crn"] >= 50,
        "at least 5 regional or country-like subdirectories crawled": len(crawled_region_dirs) >= 5,
    }
    passed = all(checks.values())

    lines = [
        "ITRDB dry-run validation report",
        "================================",
        f"total directories visited: {len(directory_rows)}",
        f"successful directories visited: {sum(1 for row in directory_rows if row['status'] == 'ok')}",
        f"failed directory requests: {len(failures)}",
        f"total files discovered: {len(inventory_rows)}",
        "",
        "counts by extension:",
    ]
    lines.extend(f"  {ext or '[none]'}: {count}" for ext, count in sorted(ext_counts.items()))
    lines.extend(["", "counts by file class:"])
    lines.extend(f"  {file_class}: {count}" for file_class, count in sorted(class_counts.items()))
    lines.extend(
        [
            "",
            f".rwl count: {ext_counts['.rwl']}",
            f".crn count: {ext_counts['.crn']}",
            f"regional/country-like subdirectories crawled: {len(crawled_region_dirs)}",
            "",
            "threshold checks:",
        ]
    )
    lines.extend(f"  {'PASS' if ok else 'FAIL'} - {label}" for label, ok in checks.items())

    if failures:
        lines.extend(["", "failed request samples:"])
        for row in failures[:10]:
            lines.append(
                f"  {row['directory_url']} status={row['http_status'] or '[none]'} error={row['error'] or '[none]'}"
            )

    lines.extend(["", "sample discovered URLs:"])
    lines.extend(f"  {row['url']}" for row in inventory_rows[:sample_count])
    lines.extend(["", f"VALIDATION: {'PASS' if passed else 'FAIL'}"])
    return passed, lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, help="TREEQUAKE project root")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="discover and report only")
    parser.add_argument("--download", action="store_true", help="perform real downloads")
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--pause", type=float, default=0.1, help="seconds between polite requests")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run and args.download:
        print("[ERROR] Choose either --dry-run or --download, not both.", file=sys.stderr)
        return 2

    dry_run = not args.download
    project_root = Path(args.project_root).resolve()
    data_root = project_root / "data" / "raw" / "itrdb"
    inventory_path = project_root / "outputs" / "inventories" / "itrdb_file_inventory.csv"
    download_log_path = project_root / "outputs" / "logs" / "itrdb_download_log.csv"
    directory_log_path = project_root / "outputs" / "logs" / "itrdb_crawled_directories.csv"
    validation_path = project_root / "outputs" / "logs" / "itrdb_validation_report.txt"
    log_dir = project_root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    all_files: list[DiscoveredFile] = []
    directory_rows: list[dict[str, str]] = []

    for base_class, base_url in BASE_URLS.items():
        print(f"[INFO] Crawling {base_url}")
        files, dirs = crawl_base(
            base_class=base_class,
            base_url=base_url,
            log_dir=log_dir,
            max_depth=args.max_depth,
            timeout=args.timeout,
            retries=args.retries,
            pause=args.pause,
            user_agent=args.user_agent,
        )
        print(f"[INFO] Found {len(files)} files and visited {len(dirs)} directories under {base_url}")
        all_files.extend(files)
        directory_rows.extend(dirs)

    seen_files: set[str] = set()
    inventory_rows: list[dict[str, str]] = []
    download_rows: list[dict[str, str]] = []

    for item in sorted(all_files, key=lambda found: found.url):
        if item.url in seen_files:
            continue
        seen_files.add(item.url)
        base_url = BASE_URLS[item.base_class]
        filename = Path(unquote(urlparse(item.url).path)).name
        ext = extension_from_url(item.url)
        site_code = site_code_from_filename(filename)
        file_class = classify_file(item.url, item.base_class)
        region, country = region_country_guess(item.url, base_url)
        destination = local_path_for(item.url, item.base_class, data_root)

        if dry_run:
            status = "dry_run_not_downloaded"
            bytes_written = 0
        else:
            status, bytes_written = download_file(
                item.url,
                destination,
                overwrite=args.overwrite,
                timeout=args.timeout,
                retries=args.retries,
                pause=args.pause,
                user_agent=args.user_agent,
            )

        inventory_rows.append(
            {
                "site_code": site_code,
                "file_class": file_class,
                "filename": filename,
                "url": item.url,
                "local_path": str(destination),
                "status": status,
                "extension": ext,
                "region_guess": region,
                "country_guess": country,
            }
        )
        download_rows.append(
            {
                "url": item.url,
                "local_path": str(destination),
                "status": status,
                "bytes_written": str(bytes_written),
            }
        )

    write_csv(
        inventory_path,
        inventory_rows,
        [
            "site_code",
            "file_class",
            "filename",
            "url",
            "local_path",
            "status",
            "extension",
            "region_guess",
            "country_guess",
        ],
    )
    write_csv(
        directory_log_path,
        directory_rows,
        ["directory_url", "base_class", "depth", "status", "http_status", "error", "link_count"],
    )
    write_csv(download_log_path, download_rows, ["url", "local_path", "status", "bytes_written"])

    passed, report_lines = validation_lines(inventory_rows, directory_rows)
    validation_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[INFO] Total files discovered: {len(inventory_rows)}")
    print(f"[INFO] Inventory: {inventory_path}")
    print(f"[INFO] Directory log: {directory_log_path}")
    print(f"[INFO] Download log: {download_log_path}")
    print(f"[INFO] Validation report: {validation_path}")
    if dry_run:
        print("[DRY RUN] No files downloaded.")

    if dry_run and not passed:
        print("[ERROR] Dry-run validation failed. See validation report.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
