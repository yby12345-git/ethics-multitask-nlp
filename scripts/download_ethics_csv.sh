#!/usr/bin/env bash

# Download the ETHICS benchmark dataset from Hugging Face.
# The raw CSV files are stored under data/ethics_raw/.
# This script downloads the training and test splits for all five subtasks.

set -euo pipefail

BASE_URL="https://huggingface.co/datasets/hendrycks/ethics/resolve/main/data"
SUBSETS=(
  "commonsense"
  "deontology"
  "justice"
  "utilitarianism"
  "virtue"
)
SPLITS=(
  "train"
  "test"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/data/ethics_raw"

mkdir -p "${OUTPUT_ROOT}"

if ! command -v wget >/dev/null 2>&1; then
  echo "Error: wget is not installed or is not available in PATH." >&2
  exit 1
fi

echo "Starting ETHICS dataset download."
echo "Output directory: ${OUTPUT_ROOT}"

for subset in "${SUBSETS[@]}"; do
  subset_dir="${OUTPUT_ROOT}/${subset}"
  mkdir -p "${subset_dir}"

  echo
  echo "Processing subset: ${subset}"

  for split in "${SPLITS[@]}"; do
    url="${BASE_URL}/${subset}/${split}.csv"
    output_file="${subset_dir}/${split}.csv"
    temporary_file="${output_file}.part"

    echo "Downloading ${url}"
    echo "Saving to ${output_file}"

    rm -f "${temporary_file}"

    wget \
      --show-progress \
      --tries=3 \
      --timeout=30 \
      --retry-connrefused \
      --output-document="${temporary_file}" \
      "${url}"

    if [[ ! -s "${temporary_file}" ]]; then
      echo "Error: downloaded file is empty: ${temporary_file}" >&2
      rm -f "${temporary_file}"
      exit 1
    fi

    mv "${temporary_file}" "${output_file}"
  done
done

echo
echo "ETHICS dataset download completed successfully."
echo "Downloaded files are available under:"
echo "  ${OUTPUT_ROOT}"
