#!/bin/bash
# Search for Additional Violent Datasets on Kaggle
# Run this on your RunPod server to find more violent datasets

set -e

echo "=========================================="
echo "KAGGLE VIOLENT DATASET SEARCH"
echo "=========================================="
echo ""

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials not found!"
    exit 1
fi

echo "✅ Kaggle credentials found"
echo ""

echo "🔍 Searching Kaggle for additional violent datasets..."
echo ""

# Search for fight detection datasets
echo "=== FIGHT DETECTION DATASETS ==="
kaggle datasets list -s "fight detection" --sort-by hotness | head -20
echo ""

# Search for violence detection datasets
echo "=== VIOLENCE DETECTION DATASETS ==="
kaggle datasets list -s "violence detection" --sort-by hotness | head -20
echo ""

# Search for assault datasets
echo "=== ASSAULT/CRIME DATASETS ==="
kaggle datasets list -s "assault detection" --sort-by hotness | head -20
echo ""

# Search for combat sports
echo "=== COMBAT SPORTS DATASETS ==="
kaggle datasets list -s "combat sports" --sort-by hotness | head -20
echo ""

# Search for UFC/MMA
echo "=== UFC/MMA DATASETS ==="
kaggle datasets list -s "UFC MMA" --sort-by hotness | head -20
echo ""

# Search for surveillance violence
echo "=== SURVEILLANCE VIOLENCE DATASETS ==="
kaggle datasets list -s "surveillance violence" --sort-by hotness | head -20
echo ""

echo "=========================================="
echo "SEARCH COMPLETE!"
echo "=========================================="
echo ""
echo "📋 NEXT STEPS:"
echo "1. Review the dataset names above"
echo "2. Verify they contain violent content (check dataset descriptions on Kaggle website)"
echo "3. Add confirmed datasets to download_additional_violent.sh"
echo ""
