#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/if/Ricerca/treequake"

mkdir -p "$PROJECT_ROOT"/{config,scripts/python,scripts/r,data/raw,data/processed,outputs/{inventories,matches,tests,figures,logs},docs}

cat > "$PROJECT_ROOT/README.md" <<'EOF'
# TREEQUAKE

TREEQUAKE is an open-science workflow for screening possible earthquake-associated growth suppressions in tree-ring datasets.

It combines earthquake catalogs, tree-ring archives, spatial matching, dendrochronological processing, and reproducible reporting.

## Status

Early research prototype.

## Core idea

1. Download earthquake catalogs.
2. Download and inventory ITRDB tree-ring datasets.
3. Match earthquake events to nearby tree-ring sites.
4. Test whether growth suppression begins near earthquake years.
5. Log all successes, failures, exclusions, and candidate signals.

## Philosophy

TREEQUAKE is a screening framework, not a causal proof machine. Results identify candidate site-event pairs for further dendroecological, geological, and historical investigation.
EOF

cat > "$PROJECT_ROOT/LICENSE" <<'EOF'
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, subject to the standard MIT License terms.
EOF

cat > "$PROJECT_ROOT/CITATION.cff" <<'EOF'
cff-version: 1.2.0
title: TREEQUAKE
message: "If you use TREEQUAKE, please cite the software and the source datasets used in your analysis."
type: software
authors:
  - family-names: Taylor
    given-names: EJ
version: 0.1.0
date-released: 2026-01-01
license: MIT
EOF

cat > "$PROJECT_ROOT/config/defaults.yml" <<'EOF'
project:
  name: treequake
  version: 0.1.0

earthquakes:
  catalog: usgs_comcat
  min_magnitude: 5.5
  start_year: 1000
  end_year: 2026
  radii_km: [25, 50, 75, 100, 150, 200]

tree_rings:
  provider: itrdb
  download_all_associated_files: true
  analyze_rwl_only_initially: true
  min_trees_required: 5
  pre_window_years: 10
  post_window_years: 10

suppression:
  methods:
    - proportion_70
    - empirical_ratio
    - onset_year
  proportion_threshold: 0.70
  empirical_p_threshold: 0.05
  onset_allowed_lag_years: [0, 1]

plots:
  show_in_rstudio_pane: true
  save_png: true
  individual_raw_tree_panels: true
  individual_rwi_tree_panels: true
  master_chronology: true

reproducibility:
  preserve_raw_downloads: true
  log_failures: true
  overwrite_existing_downloads: false
EOF

cat > "$PROJECT_ROOT/run_treequake.sh" <<'EOF'
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
EOF

chmod +x "$PROJECT_ROOT/run_treequake.sh"

echo "Created TREEQUAKE scaffold at:"
echo "$PROJECT_ROOT"
echo
echo "Run:"
echo "bash $PROJECT_ROOT/run_treequake.sh"
