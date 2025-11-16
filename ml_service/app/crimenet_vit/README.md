# CrimeNet Vision Transformer Implementation Guide

## Overview

CrimeNet ViT achieves 99% accuracy on violence detection by using Vision Transformers instead of traditional CNNs.

## Architecture

```
Video (16 frames)
    ↓
Frame Patches (196 patches per frame, 16x16 each)
    ↓
Patch Embeddings + Position Embeddings
    ↓
Transformer Encoder (12 layers)
    ├─ Multi-Head Self-Attention
    ├─ Layer Normalization
    └─ Feed-Forward Network
    ↓
Temporal Transformer (across frames)
    ↓
Classification Head → Violence Score (0-1)
```

## Installation

```bash
# Install dependencies
pip install transformers==4.36.0
pip install timm==0.9.12
pip install einops==0.7.0
pip install torch==2.1.0 torchvision==0.16.0
```

## Implementation Options

### Option 1: Use Pre-trained ViT and Fine-tune

**Fastest approach** - Uses pre-trained weights from ImageNet:

```python
from transformers import ViTForImageClassification
import torch

# Load pre-trained ViT
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=2,  # violence vs non-violence
    ignore_mismatched_sizes=True
)

# Fine-tune on your violence dataset
# Training code in train_crimenet_vit.py
```

### Option 2: Build CrimeNet ViT from Scratch

**Best accuracy** - Implements full CrimeNet architecture:

```python
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

class CrimeNetViT(nn.Module):
    def __init__(self, num_frames=16, num_classes=2):
        super().__init__()
        self.num_frames = num_frames

        # Spatial ViT (per-frame)
        self.spatial_vit = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_classes=0  # Feature extraction only
        )

        # Temporal Transformer (across frames)
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=3072
            ),
            num_layers=4
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, frames, channels, height, width)
        B, T, C, H, W = x.shape

        # Extract spatial features from each frame
        x = x.view(B * T, C, H, W)
        spatial_features = self.spatial_vit(x)  # (B*T, 768)

        # Reshape for temporal modeling
        spatial_features = spatial_features.view(B, T, -1)  # (B, T, 768)

        # Temporal attention across frames
        temporal_features = self.temporal_transformer(
            spatial_features.transpose(0, 1)  # (T, B, 768)
        )

        # Global average pooling over time
        pooled = temporal_features.mean(dim=0)  # (B, 768)

        # Classification
        logits = self.classifier(pooled)  # (B, 2)

        return logits
```

### Option 3: Ensemble with VGG19 (Recommended)

**Best transition path** - Combines strengths of both:

```python
class EnsembleModel(nn.Module):
    def __init__(self, vgg19_model, vit_model):
        super().__init__()
        self.vgg19 = vgg19_model  # Your existing model
        self.vit = vit_model      # New CrimeNet ViT

        # Learnable ensemble weights
        self.alpha = nn.Parameter(torch.tensor(0.5))  # VGG19 weight

    def forward(self, x):
        # Get predictions from both models
        vgg_pred = self.vgg19(x)
        vit_pred = self.vit(x)

        # Weighted combination
        ensemble_pred = self.alpha * vgg_pred + (1 - self.alpha) * vit_pred

        return ensemble_pred, vgg_pred, vit_pred
```

## Training Strategy

### Step 1: Prepare Dataset

```python
# Your existing dataset structure should work
violence_detection_mvp/
├── organized_dataset/
│   ├── train/
│   │   ├── violence/
│   │   └── non_violence/
│   ├── val/
│   └── test/
```

### Step 2: Training Script

See `train_crimenet_vit.py` for complete training code.

**Key hyperparameters**:
- Learning rate: 1e-4 (with warmup)
- Batch size: 8 (with gradient accumulation for effective batch of 32)
- Frames per video: 16
- Image size: 224x224
- Epochs: 50-100
- Optimizer: AdamW with weight decay

### Step 3: Fine-tuning Tips

1. **Use pre-trained weights**: Start from ImageNet pre-training
2. **Freeze early layers**: Only train last 4 transformer blocks initially
3. **Gradual unfreezing**: Unfreeze more layers after 10 epochs
4. **Data augmentation**: Random crop, flip, color jitter
5. **Class weights**: Handle imbalanced dataset

## Expected Results

| Model | Accuracy | False Positives | Inference Speed |
|-------|----------|-----------------|-----------------|
| VGG19 + Bi-LSTM (current) | 87-90% | 15-20% | 30 FPS |
| CrimeNet ViT (replace) | 99% | 2.96% | 60 FPS |
| Ensemble (recommended) | 95-97% | 5-8% | 45 FPS |

## Integration with Existing System

### Update API Endpoint

```python
# In ml_service/app/main.py

@app.post("/api/detect/video-vit")
async def detect_violence_vit(file: UploadFile):
    # Load video
    video_frames = load_video(file, num_frames=16)

    # Preprocess
    frames_tensor = preprocess_for_vit(video_frames)

    # Inference with CrimeNet ViT
    with torch.no_grad():
        logits = crimenet_vit_model(frames_tensor)
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    return {
        "violence_probability": prob,
        "is_violence": prob > 0.85,
        "model": "CrimeNet ViT"
    }

@app.post("/api/detect/ensemble")
async def detect_violence_ensemble(file: UploadFile):
    # Get predictions from both models
    vgg_result = await detect_violence_vgg(file)
    vit_result = await detect_violence_vit(file)

    # Weighted ensemble
    ensemble_prob = 0.3 * vgg_result["violence_probability"] + \
                    0.5 * vit_result["violence_probability"] + \
                    0.2 * skeleton_result["violence_probability"]

    return {
        "violence_probability": ensemble_prob,
        "is_violence": ensemble_prob > 0.85,
        "model": "Ensemble (VGG + ViT + Skeleton)",
        "individual_predictions": {
            "vgg19": vgg_result["violence_probability"],
            "vit": vit_result["violence_probability"],
            "skeleton": skeleton_result["violence_probability"]
        }
    }
```

## Performance Optimization

### TensorRT Optimization (for production)

```bash
# Convert PyTorch model to ONNX
python export_to_onnx.py --model crimenet_vit --output model.onnx

# Convert ONNX to TensorRT
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

# Expected speedup: 3-5x faster inference
```

### Quantization (for edge devices)

```python
import torch.quantization

# Dynamic quantization (easiest)
quantized_model = torch.quantization.quantize_dynamic(
    crimenet_vit_model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Expected: 4x smaller model, 2x faster, <1% accuracy loss
```

## Next Steps

1. **Week 1**: Train CrimeNet ViT on your dataset (see `train_crimenet_vit.py`)
2. **Week 2**: Create ensemble model and A/B test
3. **Week 3**: Optimize with TensorRT and deploy

## Troubleshooting

**Out of Memory**:
- Reduce batch size to 4
- Use gradient accumulation
- Enable mixed precision training (FP16)

**Low Accuracy**:
- Check learning rate (try 5e-5 to 2e-4)
- Increase training epochs
- Add more data augmentation

**Slow Inference**:
- Use TensorRT optimization
- Batch multiple frames together
- Use FP16 precision

## References

- ViViT: A Video Vision Transformer (ICCV 2021)
- CrimeNet: 99% accuracy paper (2024)
- Transformers library documentation
