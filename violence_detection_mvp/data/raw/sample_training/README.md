# Sample Violence Detection Dataset

## Dataset Information
- **Purpose**: Training and testing the Violence Detection MVP
- **Type**: Synthetic videos for immediate development
- **Structure**: RWF-2000 compatible format

## Directory Structure
```
data/raw/sample_training/
├── train/
│   ├── Fight/          # Violence videos
│   └── NonFight/       # Non-violence videos
└── val/
    ├── Fight/          # Violence validation videos
    └── NonFight/       # Non-violence validation videos
```

## Video Specifications
- **Duration**: 5 seconds per video
- **FPS**: 30 frames per second
- **Resolution**: 640x480 pixels
- **Format**: MP4 (H.264)

## Usage
```bash
# Train with sample dataset
python3 run.py train --data-dir data/raw/sample_training

# Evaluate with sample dataset
python3 run.py evaluate --data-dir data/raw/sample_training
```

## Characteristics
- **Violence videos**: Red-dominant, erratic movements, impact flashes
- **Non-violence videos**: Blue/green-dominant, calm movements, predictable patterns

This synthetic dataset allows immediate testing of the violence detection pipeline while real datasets are being downloaded.
