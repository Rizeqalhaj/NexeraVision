#!/usr/bin/env python3
"""
SIMPLE RESUME TRAINING - No complex config needed
Just loads your checkpoint and continues training
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

from data_preprocessing import VideoDataPreprocessor
from model_architecture import ViolenceDetectionModel

class VideoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_paths, labels, preprocessor, batch_size=8, shuffle=True, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.video_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_videos = []
        batch_labels = []

        for i in batch_indices:
            frames = self.preprocessor.extract_frames(self.video_paths[i])
            if self.augment:
                frames = self._augment_frames(frames)
            batch_videos.append(frames)
            batch_labels.append(self.labels[i])

        X = np.array(batch_videos)
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=2)
        return X, y

    def _augment_frames(self, frames):
        if np.random.random() > 0.5:
            frames = np.flip(frames, axis=2)
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * factor, 0, 1)
        return frames

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def find_checkpoint():
    """Find latest checkpoint"""
    checkpoint_dir = Path('/workspace/checkpoints')

    if not checkpoint_dir.exists():
        print("⚠️  No checkpoints directory")
        return None

    # Check for checkpoints
    finetuning_ckpt = checkpoint_dir / 'finetuning_best_model.keras'
    initial_ckpt = checkpoint_dir / 'initial_best_model.keras'

    if finetuning_ckpt.exists():
        log_file = Path('/workspace/logs/training/finetuning_training_log.csv')
        return {
            'path': str(finetuning_ckpt),
            'phase': 'finetuning',
            'log': str(log_file) if log_file.exists() else None
        }

    if initial_ckpt.exists():
        log_file = Path('/workspace/logs/training/initial_training_log.csv')
        return {
            'path': str(initial_ckpt),
            'phase': 'initial',
            'log': str(log_file) if log_file.exists() else None
        }

    return None


def get_last_epoch(log_path):
    """Get last completed epoch from CSV log"""
    if not log_path or not Path(log_path).exists():
        return 0

    try:
        df = pd.read_csv(log_path)
        return len(df)
    except:
        return 0


def main():
    print("=" * 80)
    print("SIMPLE RESUME TRAINING")
    print("=" * 80)

    # Check for checkpoint
    ckpt = find_checkpoint()

    if ckpt:
        print(f"\n🔄 CHECKPOINT FOUND!")
        print(f"   Path: {ckpt['path']}")
        print(f"   Phase: {ckpt['phase']}")

        last_epoch = get_last_epoch(ckpt['log'])
        print(f"   Last Epoch: {last_epoch}")

        # Load checkpoint
        print("\n📥 Loading checkpoint...")
        model = tf.keras.models.load_model(ckpt['path'])
        print("✅ Checkpoint loaded!")

        # Create model builder wrapper
        model_builder = ViolenceDetectionModel(
            frames_per_video=20,
            sequence_model='Bidirectional-GRU',
            gru_units=128,
            dense_layers=[256, 128, 64],
            dropout_rates=[0.5, 0.5, 0.5]
        )
        model_builder.model = model

    else:
        print("\n🆕 No checkpoint - starting fresh")

        # Build new model
        model_builder = ViolenceDetectionModel(
            frames_per_video=20,
            sequence_model='Bidirectional-GRU',
            gru_units=128,
            dense_layers=[256, 128, 64],
            dropout_rates=[0.5, 0.5, 0.5]
        )

        model = model_builder.build_model(trainable_backbone=False)
        model_builder.compile_model(learning_rate=0.0001)

        ckpt = {'phase': 'initial', 'log': None}
        last_epoch = 0

    # Load splits
    print("\n📂 Loading data splits...")
    splits_file = Path('/workspace/processed/splits.json')

    with open(splits_file) as f:
        splits_data = json.load(f)

    train_videos = splits_data['train']['videos']
    train_labels = splits_data['train']['labels']
    val_videos = splits_data['val']['videos']
    val_labels = splits_data['val']['labels']

    print(f"   Train: {len(train_videos)} videos")
    print(f"   Val: {len(val_videos)} videos")

    # Create preprocessor
    print("\n🔧 Setting up data pipeline...")
    preprocessor = VideoDataPreprocessor(
        dataset_dir="/workspace/datasets/tier1",
        frames_per_video=20
    )

    # Create generators
    train_gen = VideoDataGenerator(
        train_videos, train_labels, preprocessor,
        batch_size=8, shuffle=True, augment=True
    )

    val_gen = VideoDataGenerator(
        val_videos, val_labels, preprocessor,
        batch_size=8, shuffle=False, augment=False
    )

    # Setup callbacks
    print(f"\n📊 Setting up training ({ckpt['phase']} phase)...")

    checkpoint_path = f"/workspace/checkpoints/{ckpt['phase']}_best_model.keras"
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            f"/workspace/logs/training/{ckpt['phase']}_training_log.csv",
            append=True
        )
    ]

    # Determine training plan
    initial_epochs = 50
    finetune_epochs = 50
    total_epochs = initial_epochs + finetune_epochs

    if ckpt['phase'] == 'finetuning':
        # Already in fine-tuning
        if last_epoch >= total_epochs:
            print(f"\n✅ Training complete! ({last_epoch}/{total_epochs} epochs)")
            return

        print(f"\n🚀 Resuming fine-tuning from epoch {last_epoch}/{total_epochs}")
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=total_epochs,
            initial_epoch=last_epoch,
            callbacks=callbacks,
            verbose=1
        )

    else:
        # Initial phase
        if last_epoch >= initial_epochs:
            # Move to fine-tuning
            print(f"\n✅ Initial training complete!")
            print(f"\n🔧 Unfreezing backbone for fine-tuning...")
            model_builder.unfreeze_backbone(num_layers=-1)

            # Update callbacks for fine-tuning
            checkpoint_path = "/workspace/checkpoints/finetuning_best_model.keras"
            callbacks[0] = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks[3] = tf.keras.callbacks.CSVLogger(
                "/workspace/logs/training/finetuning_training_log.csv",
                append=True
            )

            print(f"\n🚀 Starting fine-tuning...")
            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=total_epochs,
                initial_epoch=initial_epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Continue initial training
            print(f"\n🚀 Resuming initial training from epoch {last_epoch}/{initial_epochs}")
            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=initial_epochs,
                initial_epoch=last_epoch,
                callbacks=callbacks,
                verbose=1
            )

            # Then fine-tuning
            print(f"\n✅ Initial training complete!")
            print(f"\n🔧 Unfreezing backbone for fine-tuning...")
            model_builder.unfreeze_backbone(num_layers=-1)

            # Update callbacks
            checkpoint_path = "/workspace/checkpoints/finetuning_best_model.keras"
            callbacks[0] = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks[3] = tf.keras.callbacks.CSVLogger(
                "/workspace/logs/training/finetuning_training_log.csv",
                append=True
            )

            print(f"\n🚀 Starting fine-tuning...")
            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=total_epochs,
                initial_epoch=initial_epochs,
                callbacks=callbacks,
                verbose=1
            )

    # Save final model
    final_path = "/workspace/models/final_model.keras"
    Path(final_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(final_path)

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print(f"   Final model: {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted - checkpoint saved, can resume!")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
