#!/bin/bash
# Search and Download Non-Violent Datasets from Kaggle
# Uses actual verified Kaggle datasets

set -e

echo "=========================================="
echo "KAGGLE NON-VIOLENT DATASET SEARCH"
echo "=========================================="
echo ""

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials not found!"
    exit 1
fi

echo "✅ Kaggle credentials found"
echo ""

# Create output directory
mkdir -p /workspace/datasets/nonviolent_kaggle
cd /workspace/datasets/nonviolent_kaggle

echo "🔍 Searching Kaggle for non-violent activity datasets..."
echo ""

# Search for human activity datasets
kaggle datasets list -s "human activity recognition" --sort-by hotness | head -20

echo ""
echo "=========================================="
echo "MANUAL DOWNLOAD INSTRUCTIONS"
echo "=========================================="
echo ""
echo "To download a dataset from the list above:"
echo "1. Copy the dataset name (format: username/dataset-name)"
echo "2. Run: kaggle datasets download -d username/dataset-name"
echo "3. Unzip: unzip dataset-name.zip"
echo ""
echo "Example:"
echo "  kaggle datasets download -d username/dataset-name"
echo "  unzip dataset-name.zip -d dataset_folder"
echo ""
echo "=========================================="
