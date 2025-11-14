#!/bin/bash

echo "================================================================"
echo "COPYING EXISTING VGG19 FEATURES TO ENSEMBLE CACHE"
echo "================================================================"
echo ""
echo "This will save 2-3 hours of feature extraction!"
echo ""

# Create directories
mkdir -p /workspace/ensemble_cache/vgg19_bilstm
mkdir -p /workspace/ensemble_cache/vgg19_bigru
mkdir -p /workspace/ensemble_cache/vgg19_attention

echo "Creating cache directories..."

# Copy features for Model 1 (vgg19_bilstm)
echo "Copying features for vgg19_bilstm..."
cp /workspace/feature_cache/train_features.npy /workspace/ensemble_cache/vgg19_bilstm/
cp /workspace/feature_cache/train_labels.npy /workspace/ensemble_cache/vgg19_bilstm/
cp /workspace/feature_cache/val_features.npy /workspace/ensemble_cache/vgg19_bilstm/
cp /workspace/feature_cache/val_labels.npy /workspace/ensemble_cache/vgg19_bilstm/
cp /workspace/feature_cache/test_features.npy /workspace/ensemble_cache/vgg19_bilstm/
cp /workspace/feature_cache/test_labels.npy /workspace/ensemble_cache/vgg19_bilstm/

# Copy to Model 2 (vgg19_bigru) - same VGG19 features
echo "Copying features for vgg19_bigru..."
cp /workspace/feature_cache/train_features.npy /workspace/ensemble_cache/vgg19_bigru/
cp /workspace/feature_cache/train_labels.npy /workspace/ensemble_cache/vgg19_bigru/
cp /workspace/feature_cache/val_features.npy /workspace/ensemble_cache/vgg19_bigru/
cp /workspace/feature_cache/val_labels.npy /workspace/ensemble_cache/vgg19_bigru/
cp /workspace/feature_cache/test_features.npy /workspace/ensemble_cache/vgg19_bigru/
cp /workspace/feature_cache/test_labels.npy /workspace/ensemble_cache/vgg19_bigru/

# Copy to Model 3 (vgg19_attention) - same VGG19 features
echo "Copying features for vgg19_attention..."
cp /workspace/feature_cache/train_features.npy /workspace/ensemble_cache/vgg19_attention/
cp /workspace/feature_cache/train_labels.npy /workspace/ensemble_cache/vgg19_attention/
cp /workspace/feature_cache/val_features.npy /workspace/ensemble_cache/vgg19_attention/
cp /workspace/feature_cache/val_labels.npy /workspace/ensemble_cache/vgg19_attention/
cp /workspace/feature_cache/test_features.npy /workspace/ensemble_cache/vgg19_attention/
cp /workspace/feature_cache/test_labels.npy /workspace/ensemble_cache/vgg19_attention/

echo ""
echo "================================================================"
echo "✅ FEATURES COPIED SUCCESSFULLY!"
echo "================================================================"
echo ""
echo "All 3 models now have cached VGG19 features."
echo "Feature extraction will be skipped during training."
echo ""
echo "Now you can start training:"
echo "bash /home/admin/Desktop/NexaraVision/TRAIN_ENSEMBLE_92_PERCENT.sh"
echo ""
echo "Training will start immediately with Model 1!"
echo "================================================================"
