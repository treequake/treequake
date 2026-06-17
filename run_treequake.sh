#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$PROJECT_ROOT/config/defaults.yml"

echo "TREEQUAKE"
echo "Project root: $PROJECT_ROOT"
echo "Config:       $CONFIG_FILE"
echo

echo "This is the scaffold launcher."
echo "Next module to implement: ITRDB downloader and inventory builder."
