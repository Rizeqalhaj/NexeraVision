"""
Training script for CrimeNet Vision Transformer

Usage:
    python train_crimenet_vit.py --data_dir /path/to/organized_dataset --epochs 100

Expected performance:
    - Week 1: 92-95% accuracy
    - Week 2: 95-97% accuracy
    - Week 3: 97-99% accuracy with fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from model import CrimeNetViT, create_crimenet_vit


class ViolenceVideoDataset(Dataset):
    """
    Dataset for loading violence detection videos

    Directory structure:
        data_dir/
            train/
                violence/
                    video1.mp4
                    video2.mp4
                non_violence/
                    video3.mp4
                    video4.mp4
            val/
                violence/
                non_violence/
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        num_frames: int = 16,
        img_size: int = 224,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir) / split
        self.num_frames = num_frames
        self.img_size = img_size
        self.augment = augment and (split == 'train')

        # Get all video paths
        self.samples = []
        for label_idx, label_name in enumerate(['non_violence', 'violence']):
            label_dir = self.data_dir / label_name
            if not label_dir.exists():
                continue

            for video_file in label_dir.glob('*.mp4'):
                self.samples.append((str(video_file), label_idx))

        print(f"{split}: Found {len(self.samples)} videos "
              f"({sum(1 for _, l in self.samples if l == 1)} violence, "
              f"{sum(1 for _, l in self.samples if l == 0)} non-violence)")

        # Data augmentation for training
        if self.augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.samples)

    def load_video(self, video_path: str) -> np.ndarray:
        """Load video and extract frames"""
        cap = cv2.VideoCapture(video_path)

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames uniformly
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Repeat frames if video is too short
            indices = np.arange(total_frames)
            indices = np.pad(
                indices,
                (0, self.num_frames - len(indices)),
                mode='wrap'
            )

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        return np.array(frames)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        # Load video frames
        frames = self.load_video(video_path)  # (T, H, W, C)

        # Apply transforms to each frame
        transformed_frames = []
        for frame in frames:
            transformed_frames.append(self.transform(frame))

        # Stack frames
        video_tensor = torch.stack(transformed_frames)  # (T, C, H, W)

        return video_tensor, label


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler
) -> tuple:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for videos, labels in pbar:
        videos = videos.to(device)  # (B, T, C, H, W)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with torch.cuda.amp.autocast():
            logits, _ = model(videos)
            loss = criterion(logits, labels)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss / (pbar.n + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # For calculating precision, recall, F1
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    pbar = tqdm(dataloader, desc='Validation')
    for videos, labels in pbar:
        videos = videos.to(device)
        labels = labels.to(device)

        # Forward pass
        logits, _ = model(videos)
        loss = criterion(logits, labels)

        # Metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Precision/Recall metrics
        true_positives += ((predicted == 1) & (labels == 1)).sum().item()
        false_positives += ((predicted == 1) & (labels == 0)).sum().item()
        false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return avg_loss, accuracy, precision, recall, f1_score


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Create datasets
    train_dataset = ViolenceVideoDataset(
        args.data_dir,
        split='train',
        num_frames=args.num_frames,
        img_size=args.img_size,
        augment=True
    )

    val_dataset = ViolenceVideoDataset(
        args.data_dir,
        split='val',
        num_frames=args.num_frames,
        img_size=args.img_size,
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    print("\nCreating CrimeNet ViT model...")
    model = create_crimenet_vit(
        pretrained=args.pretrained,
        num_classes=2,
        num_frames=args.num_frames
    )
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Loss function with class weights (handle imbalanced data)
    class_counts = [
        sum(1 for _, l in train_dataset.samples if l == 0),
        sum(1 for _, l in train_dataset.samples if l == 1)
    ]
    class_weights = torch.tensor(
        [1.0 / c for c in class_counts],
        device=device
    )
    class_weights = class_weights / class_weights.sum()  # Normalize

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_val_acc = 0
    best_f1 = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # Validate
        val_loss, val_acc, precision, recall, f1_score = validate(
            model, val_loader, criterion, device
        )

        # Update learning rate
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(f1_score)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1_score:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': f1_score,
            }, os.path.join(args.save_dir, 'best_model_acc.pth'))
            print(f"✓ Saved best accuracy model: {val_acc:.2f}%")

        if f1_score > best_f1:
            best_f1 = f1_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': f1_score,
            }, os.path.join(args.save_dir, 'best_model_f1.pth'))
            print(f"✓ Saved best F1 model: {f1_score:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'best_val_acc': best_val_acc,
        'best_f1': best_f1
    }, os.path.join(args.save_dir, 'final_model.pth'))

    # Save training history
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Models saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CrimeNet ViT')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to organized_dataset directory')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames per video clip')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')

    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    # Model
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ImageNet pre-trained weights')

    # Output
    parser.add_argument('--save_dir', type=str,
                        default='checkpoints/crimenet_vit',
                        help='Directory to save models')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Train
    main(args)
