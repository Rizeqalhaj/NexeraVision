#!/usr/bin/env python3
"""
Optimized Training Script for 2× RTX 5000 Ada Generation
64GB VRAM total - Aggressive batch sizes and multi-GPU training
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms, models
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import cv2
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - Optimized for 2× RTX 5000 Ada (64GB VRAM)
# ============================================================================

CONFIG = {
    # Multi-GPU settings
    'multi_gpu': True,
    'gpu_ids': [0, 1],  # Both RTX 5000 cards
    'distributed': True,  # Use DistributedDataParallel for better performance

    # Dataset paths
    'dataset_path': '/workspace/organized_dataset',  # From analyze_and_split_dataset.py
    'checkpoint_dir': '/workspace/checkpoints',
    'log_dir': '/workspace/logs',

    # Model architecture
    'backbone': 'vgg19',  # or 'resnet50', 'efficientnet_b0'
    'lstm_hidden': 128,
    'lstm_layers': 3,
    'attention_heads': 8,
    'dropout': 0.3,

    # Training - AGGRESSIVE for 64GB VRAM
    'batch_size': 32,  # Per GPU = 64 total (2 GPUs)
    'accumulation_steps': 4,  # Effective batch = 256
    'epochs': 50,
    'early_stopping_patience': 10,

    # Sequence settings
    'num_frames': 16,  # 16 frames per video
    'frame_size': (224, 224),

    # Optimizer
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'optimizer': 'adamw',  # AdamW better than Adam

    # Scheduler
    'scheduler': 'cosine',  # Cosine annealing with warmup
    'warmup_epochs': 5,

    # Mixed precision
    'mixed_precision': True,  # FP16 training (2x faster)

    # Data augmentation
    'augment_train': True,
    'augment_strength': 'medium',  # light, medium, heavy
}

# ============================================================================
# VIDEO DATASET
# ============================================================================

class VideoDataset(Dataset):
    """Video dataset for violence detection."""

    def __init__(self, video_paths, labels, num_frames=16, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def load_video(self, path):
        """Load video and extract frames."""
        cap = cv2.VideoCapture(str(path))
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames uniformly
        if total_frames > self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = list(range(total_frames))
            # Pad if not enough frames
            while len(indices) < self.num_frames:
                indices.append(indices[-1])

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()
        return frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            frames = self.load_video(video_path)

            # Apply transforms
            if self.transform:
                frames = [self.transform(frame) for frame in frames]

            # Stack to tensor: (T, C, H, W)
            frames = torch.stack(frames)

            return frames, label

        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Return zeros on error
            return torch.zeros(self.num_frames, 3, 224, 224), label


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ViolenceDetectionModel(nn.Module):
    """VGG19 + Bi-LSTM + Multi-Head Attention for violence detection."""

    def __init__(self, lstm_hidden=128, lstm_layers=3, attention_heads=8, dropout=0.3):
        super().__init__()

        # VGG19 backbone (pre-trained on ImageNet)
        vgg19 = models.vgg19(pretrained=True)
        self.features = vgg19.features  # CNN layers

        # Adaptive pooling to fixed size
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # FC layer from VGG19 (extract fc2 = 4096-dim features)
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=4096,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,  # bidirectional = 2x hidden
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, time, channels, height, width)
        batch_size, time_steps = x.size(0), x.size(1)

        # Reshape to process all frames: (batch * time, C, H, W)
        x = x.view(batch_size * time_steps, x.size(2), x.size(3), x.size(4))

        # CNN feature extraction
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)  # (batch * time, 4096)

        # Reshape back to sequence: (batch, time, features)
        x = x.view(batch_size, time_steps, -1)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, lstm_hidden * 2)

        # Multi-head attention (self-attention)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling over time
        pooled = torch.mean(attn_out, dim=1)  # (batch, lstm_hidden * 2)

        # Classification
        output = self.classifier(pooled)  # (batch, 1)

        return output.squeeze(1)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def get_transforms(augment=False, strength='medium'):
    """Get data transforms."""
    if augment:
        if strength == 'light':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        elif strength == 'medium':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:  # heavy
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def load_dataset(config):
    """Load train/val/test splits from physical folder structure."""
    dataset_path = Path(config['dataset_path'])

    splits = {}
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split

        if not split_dir.exists():
            print(f"❌ ERROR: {split_dir} not found")
            print(f"   Run create_physical_splits.py first!")
            sys.exit(1)

        violent_dir = split_dir / 'violent'
        nonviolent_dir = split_dir / 'nonviolent'

        if not violent_dir.exists() or not nonviolent_dir.exists():
            print(f"❌ ERROR: violent/ or nonviolent/ folders not found in {split_dir}")
            sys.exit(1)

        videos = []
        labels = []

        # Load violent videos (label = 1)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
        for ext in video_extensions:
            for video in violent_dir.glob(f'*{ext}'):
                videos.append(str(video))
                labels.append(1)

        # Load non-violent videos (label = 0)
        for ext in video_extensions:
            for video in nonviolent_dir.glob(f'*{ext}'):
                videos.append(str(video))
                labels.append(0)

        splits[split] = (videos, labels)
        print(f"✅ Loaded {split}: {len(videos)} videos ({labels.count(1)} violent, {labels.count(0)} non-violent)")

    return splits


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, accumulation_steps=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (videos, labels) in enumerate(pbar):
        videos = videos.to(device)
        labels = labels.float().to(device)

        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=CONFIG['mixed_precision']):
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # Scale loss for accumulation

        # Backward pass
        scaler.scale(loss).backward()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Metrics
        total_loss += loss.item() * accumulation_steps
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for videos, labels in pbar:
            videos = videos.to(device)
            labels = labels.float().to(device)

            with torch.cuda.amp.autocast(enabled=CONFIG['mixed_precision']):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

    return total_loss / len(dataloader), 100 * correct / total


def main():
    """Main training loop."""
    print("="*80)
    print("VIOLENCE DETECTION TRAINING - 2× RTX 5000 Ada Generation")
    print("="*80)
    print()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    print(f"🔧 GPUs available: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    print(f"\n🔧 Configuration:")
    print(f"   Batch size per GPU: {CONFIG['batch_size']}")
    print(f"   Total batch size: {CONFIG['batch_size'] * len(CONFIG['gpu_ids'])}")
    print(f"   Accumulation steps: {CONFIG['accumulation_steps']}")
    print(f"   Effective batch: {CONFIG['batch_size'] * len(CONFIG['gpu_ids']) * CONFIG['accumulation_steps']}")
    print(f"   Mixed precision: {CONFIG['mixed_precision']}")
    print()

    # Load dataset
    print("📁 Loading dataset...")
    splits = load_dataset(CONFIG)

    train_videos, train_labels = splits['train']
    val_videos, val_labels = splits['val']

    # Create datasets
    train_transform = get_transforms(
        augment=CONFIG['augment_train'],
        strength=CONFIG['augment_strength']
    )
    val_transform = get_transforms(augment=False)

    train_dataset = VideoDataset(
        train_videos, train_labels,
        num_frames=CONFIG['num_frames'],
        transform=train_transform
    )

    val_dataset = VideoDataset(
        val_videos, val_labels,
        num_frames=CONFIG['num_frames'],
        transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Create model
    print("🏗️  Building model...")
    model = ViolenceDetectionModel(
        lstm_hidden=CONFIG['lstm_hidden'],
        lstm_layers=CONFIG['lstm_layers'],
        attention_heads=CONFIG['attention_heads'],
        dropout=CONFIG['dropout']
    )

    # Multi-GPU setup
    if CONFIG['multi_gpu'] and torch.cuda.device_count() > 1:
        print(f"🚀 Using DataParallel with {len(CONFIG['gpu_ids'])} GPUs")
        model = nn.DataParallel(model, device_ids=CONFIG['gpu_ids'])

    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print()

    # Loss and optimizer
    criterion = nn.BCELoss()

    if CONFIG['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )

    # Learning rate scheduler
    if CONFIG['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG['mixed_precision'])

    # Training loop
    print("🚀 Starting training...")
    print("="*80)

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            accumulation_steps=CONFIG['accumulation_steps']
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        if CONFIG['scheduler'] == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint_path = Path(CONFIG['checkpoint_dir'])
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG
            }, checkpoint_path / 'best_model.pth')

            print(f"   ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{CONFIG['early_stopping_patience']}")

        # Early stopping
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()
