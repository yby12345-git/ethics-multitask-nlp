#!/usr/bin/env bash
set -e

BASE_URL="https://huggingface.co/datasets/hendrycks/ethics/resolve/main/data"
SUBSETS=("commonsense" "deontology" "justice" "utilitarianism" "virtue")
SPLITS=("train" "test")

mkdir -p data/ethics_raw

for subset in "${SUBSETS[@]}"; do
  mkdir -p "data/ethics_raw/${subset}"
  for split in "${SPLITS[@]}"; do
    url="${BASE_URL}/${subset}/${split}.csv"
    out="data/ethics_raw/${subset}/${split}.csv"
    echo "Downloading ${url} -> ${out}"
    wget --show-progress --tries=3 --timeout=30 -O "${out}" "${url}"
  done
done

echo "All CSV files downloaded to data/ethics_raw/"
