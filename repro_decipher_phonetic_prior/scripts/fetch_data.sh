#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data_external

# Official machine-readable sources mentioned by the paper appendix/context.
# These are optional mirrors because many source files are already vendored in third_party/ancient-scripts-datasets.

download_if_missing() {
  local url="$1"
  local out="$2"
  if [ -f "$out" ]; then
    echo "[cached] $out"
    return
  fi
  echo "[download] $url -> $out"
  curl -L --fail --retry 3 "$url" -o "$out"
}

# Wiktionary dump (raw source for descendant-tree extraction).
download_if_missing \
  "https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2" \
  "data_external/enwiktionary-latest-pages-articles.xml.bz2"

# Bible corpus mirrors referenced in appendix (already vendored in dataset repo, cached here for completeness).
download_if_missing \
  "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/Hebrew.xml" \
  "data_external/Hebrew.xml"

download_if_missing \
  "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/Latin.xml" \
  "data_external/Latin.xml"

# Generate checksum manifest for all external cache files.
(
  cd data_external
  find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 shasum -a 256 > checksums.sha256
)

echo "Wrote data_external/checksums.sha256"

# Hard requirements for full-paper regeneration that are not programmatically downloadable as ready TSV.
missing=0
for required in \
  "data_external/wiktionary_descendants_pg.tsv" \
  "data_external/wiktionary_descendants_on.tsv" \
  "data_external/wiktionary_descendants_oe.tsv" \
  "data_external/rodriguez_ramos_2014_personal_names.tsv"
do
  if [ ! -f "$required" ]; then
    echo "[missing] $required"
    missing=1
  fi
done

if [ "$missing" -eq 1 ]; then
  cat <<'MSG'

Some appendix assets are still missing in machine-readable form.
Manual preparation is required for:
- Wiktionary descendant-tree extracts (PG/ON/OE) as TSV.
- Iberian personal-name correspondences extracted from Rodriguez Ramos (2014) as TSV.

Expected file names:
- data_external/wiktionary_descendants_pg.tsv
- data_external/wiktionary_descendants_on.tsv
- data_external/wiktionary_descendants_oe.tsv
- data_external/rodriguez_ramos_2014_personal_names.tsv

Set ALLOW_INCOMPLETE=1 to continue without these files.
MSG

  if [ "${ALLOW_INCOMPLETE:-0}" != "1" ]; then
    exit 2
  fi
fi

echo "External data fetch step complete."
