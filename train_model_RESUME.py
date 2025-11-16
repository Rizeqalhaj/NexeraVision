#!/usr/bin/env python3
"""
NexaraVision Training Script with RESUME capability
Automatically detects and loads checkpoints
"""

import os
import sys

# ⚠️ CRITICAL: Set environment variables BEFORE importing TensorFlow
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 (adjust if needed)
    print("🎮 Using GPU 0")

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

from model_architecture import ViolenceDetectionModel

def setup_gpu():
    """Configure GPU for optimal training"""
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print("=" * 80)
        print("🎮 GPU Configuration")
        print("=" * 80)
        print(f"Detected {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        print()

        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth enabled (dynamic allocation)")

            tf.config.set_visible_devices(gpus[0], 'GPU')
            print("✅ Using GPU 0 only (single-GPU mode)")

            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"✅ Logical GPUs: {len(logical_gpus)}")
            print("=" * 80)
            print()

        except RuntimeError as e:
            print(f"⚠️  GPU setup error: {e}")
            print("=" * 80)
            print()

setup_gpu()

class OptimizedDataGenerator(tf.keras.utils.Sequence):
    """Fast data generator using pre-extracted frames"""

    def __init__(self, video_ids, labels, frames_dir, batch_size=32, shuffle=True, augment=False):
        self.video_ids = video_ids
        self.labels = labels
        self.frames_dir = Path(frames_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.video_ids))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.video_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_videos = []
        batch_labels = []

        for i in batch_indices:
            frames_file = self.frames_dir / f"{self.video_ids[i]}.npy"

            try:
                frames = np.load(frames_file)

                if self.augment:
                    frames = self._augment_frames(frames)

                batch_videos.append(frames)
                batch_labels.append(self.labels[i])

            except Exception as e:
                print(f"⚠️  Error loading {frames_file}: {e}")
                batch_videos.append(np.zeros((20, 224, 224, 3), dtype=np.float32))
                batch_labels.append(self.labels[i])

        X = np.array(batch_videos)
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=2)

        return X, y

    def _augment_frames(self, frames):
        """Apply data augmentation"""
        if np.random.random() > 0.5:
            frames = np.flip(frames, axis=2)

        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * factor, 0, 1)

        return frames

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class ResumeTrainingPipeline:
    """Training pipeline with automatic checkpoint resumption"""

    def __init__(self, config_path="/workspace/training_config.json"):
        with open(config_path) as f:
            self.config = json.load(f)

        print("=" * 80)
        print("NexaraVision Training Pipeline with RESUME")
        print("=" * 80)
        print(f"Loaded config from: {config_path}\n")

        if self.config.get('training', {}).get('mixed_precision', False):
            print("🚀 Enabling Mixed Precision Training (FP16)")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        self.model_builder = None
        self.model = None
        self.resume_info = None

    def find_latest_checkpoint(self):
        """Find the most recent checkpoint"""
        checkpoint_dir = Path(self.config['paths']['models']) / 'checkpoints'

        if not checkpoint_dir.exists():
            return None

        # Check for checkpoints in priority order
        checkpoint_files = [
            checkpoint_dir / 'finetuning_best_model.keras',  # Most recent
            checkpoint_dir / 'initial_best_model.keras'       # Fallback
        ]

        for checkpoint_path in checkpoint_files:
            if checkpoint_path.exists():
                # Try to find corresponding log
                log_name = checkpoint_path.stem.replace('_best_model', '_training_log.csv')
                log_path = Path(self.config['paths']['logs']) / 'training' / log_name

                resume_info = {
                    'checkpoint_path': str(checkpoint_path),
                    'log_path': str(log_path) if log_path.exists() else None,
                    'phase': 'finetuning' if 'finetuning' in checkpoint_path.name else 'initial'
                }

                return resume_info

        return None

    def load_data(self):
        """Load video IDs and labels from splits"""

        print("\n" + "=" * 80)
        print("STEP 1: Loading Pre-Extracted Data")
        print("=" * 80)

        splits_file = Path("/workspace/processed/splits.json")

        if not splits_file.exists():
            raise FileNotFoundError(f"Splits file not found: {splits_file}")

        with open(splits_file) as f:
            splits_data = json.load(f)

        self.splits = {}
        video_id = 0

        for split_name in ['train', 'val', 'test']:
            split_count = len(splits_data[split_name]['videos'])
            video_ids = list(range(video_id, video_id + split_count))
            labels = splits_data[split_name]['labels']

            self.splits[split_name] = (video_ids, labels)
            video_id += split_count

            print(f"\n{split_name.capitalize()} Set: {len(video_ids):,} videos")
            print(f"  Violence: {sum(labels):,}")
            print(f"  Non-Violence: {len(labels) - sum(labels):,}")

        print("\n✅ Data loaded (using pre-extracted frames)")
        print("=" * 80)

        return self.splits

    def build_or_load_model(self):
        """Build new model or load from checkpoint"""

        print("\n" + "=" * 80)
        print("STEP 2: Model Building/Loading")
        print("=" * 80)

        # Check for existing checkpoint
        self.resume_info = self.find_latest_checkpoint()

        if self.resume_info:
            print(f"\n🔄 CHECKPOINT FOUND!")
            print(f"   Path: {self.resume_info['checkpoint_path']}")
            print(f"   Phase: {self.resume_info['phase']}")

            # Load checkpoint
            try:
                self.model = tf.keras.models.load_model(self.resume_info['checkpoint_path'])
                print("✅ Checkpoint loaded successfully!")

                # Get training progress from log
                if self.resume_info['log_path']:
                    df = pd.read_csv(self.resume_info['log_path'])
                    last_epoch = len(df)
                    last_acc = df['accuracy'].iloc[-1] if 'accuracy' in df.columns else 0
                    last_val_acc = df['val_accuracy'].iloc[-1] if 'val_accuracy' in df.columns else 0

                    print(f"\n📊 Training Progress:")
                    print(f"   Completed Epochs: {last_epoch}")
                    print(f"   Last Train Accuracy: {last_acc:.2%}")
                    print(f"   Last Val Accuracy: {last_val_acc:.2%}")

                    self.resume_info['last_epoch'] = last_epoch
                    self.resume_info['last_val_acc'] = last_val_acc
                else:
                    self.resume_info['last_epoch'] = 0

                # Create model builder wrapper for compatibility
                self.model_builder = ViolenceDetectionModel()
                self.model_builder.model = self.model

                print("\n✅ Ready to resume training!")

            except Exception as e:
                print(f"⚠️  Error loading checkpoint: {e}")
                print("   Building new model instead...")
                self.resume_info = None
                self._build_new_model()
        else:
            print("\n🆕 No checkpoint found - starting fresh training")
            self._build_new_model()

        print("=" * 80)
        return self.model

    def _build_new_model(self):
        """Build a new model from scratch"""
        self.model_builder = ViolenceDetectionModel(
            frames_per_video=self.config['training']['frames_per_video'],
            sequence_model=self.config['model']['sequence_model'],
            gru_units=self.config['model']['gru_units'],
            dense_layers=self.config['model']['dense_layers'],
            dropout_rates=self.config['model']['dropout']
        )

        self.model = self.model_builder.build_model(trainable_backbone=False)
        self.model_builder.compile_model(
            learning_rate=self.config['training']['learning_rate'],
            optimizer=self.config['training']['optimizer'],
            loss=self.config['training']['loss']
        )

        self.model_builder.print_summary()
        self.model_builder.count_parameters()

    def setup_callbacks(self, phase='initial'):
        """Setup training callbacks"""

        print("\n" + "=" * 80)
        print(f"Setting Up Callbacks ({phase})")
        print("=" * 80)

        callbacks = []

        checkpoint_path = f"{self.config['paths']['models']}/checkpoints/{phase}_best_model.keras"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        print(f"✅ ModelCheckpoint: {checkpoint_path}")

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        print(f"✅ EarlyStopping: patience={self.config['training']['early_stopping_patience']}")

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        print(f"✅ ReduceLROnPlateau: factor=0.5, patience=10")

        log_dir = f"{self.config['paths']['logs']}/training/{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        print(f"✅ TensorBoard: {log_dir}")

        csv_path = f"{self.config['paths']['logs']}/training/{phase}_training_log.csv"
        csv_logger = tf.keras.callbacks.CSVLogger(csv_path, append=True)  # APPEND mode for resume
        callbacks.append(csv_logger)
        print(f"✅ CSVLogger: {csv_path} (append mode)")

        print("=" * 80)

        return callbacks

    def train(self, initial_epochs=30, fine_tune_epochs=20):
        """Train model with resume capability"""

        if self.splits is None or self.model is None:
            raise ValueError("Must load data and build model first")

        train_ids, train_labels = self.splits['train']
        val_ids, val_labels = self.splits['val']

        # Create generators
        train_generator = OptimizedDataGenerator(
            train_ids, train_labels,
            frames_dir="/workspace/processed/frames",
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            augment=self.config['data']['augmentation']
        )

        val_generator = OptimizedDataGenerator(
            val_ids, val_labels,
            frames_dir="/workspace/processed/frames",
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            augment=False
        )

        # Determine where to resume from
        if self.resume_info:
            if self.resume_info['phase'] == 'finetuning':
                # Already in fine-tuning phase
                print("\n" + "=" * 80)
                print("🔄 RESUMING Fine-Tuning Phase")
                print("=" * 80)

                initial_epoch = self.resume_info['last_epoch']
                total_epochs = initial_epochs + fine_tune_epochs

                if initial_epoch >= total_epochs:
                    print(f"\n✅ Training already complete! ({initial_epoch}/{total_epochs} epochs)")
                    return None, None

                callbacks = self.setup_callbacks(phase='finetuning')

                print(f"\n🚀 Resuming from epoch {initial_epoch}/{total_epochs}...")
                history_finetune = self.model.fit(
                    train_generator,
                    validation_data=val_generator,
                    epochs=total_epochs,
                    initial_epoch=initial_epoch,
                    callbacks=callbacks,
                    verbose=1
                )

                return None, history_finetune

            else:
                # In initial phase
                print("\n" + "=" * 80)
                print("🔄 RESUMING Initial Training Phase")
                print("=" * 80)

                initial_epoch = self.resume_info['last_epoch']

                if initial_epoch >= initial_epochs:
                    print(f"\n✅ Initial training complete! Moving to fine-tuning...")
                    # Skip to fine-tuning
                    print("\n" + "=" * 80)
                    print("STEP 4: Fine-Tuning (Unfrozen Backbone)")
                    print("=" * 80)

                    self.model_builder.unfreeze_backbone(num_layers=-1)
                    callbacks = self.setup_callbacks(phase='finetuning')

                    print("\n🚀 Starting fine-tuning...")
                    history_finetune = self.model.fit(
                        train_generator,
                        validation_data=val_generator,
                        epochs=initial_epochs + fine_tune_epochs,
                        initial_epoch=initial_epochs,
                        callbacks=callbacks,
                        verbose=1
                    )

                    return None, history_finetune
                else:
                    # Continue initial training
                    callbacks = self.setup_callbacks(phase='initial')

                    print(f"\n🚀 Resuming from epoch {initial_epoch}/{initial_epochs}...")
                    history_initial = self.model.fit(
                        train_generator,
                        validation_data=val_generator,
                        epochs=initial_epochs,
                        initial_epoch=initial_epoch,
                        callbacks=callbacks,
                        verbose=1
                    )

                    # Then do fine-tuning
                    print("\n" + "=" * 80)
                    print("STEP 4: Fine-Tuning (Unfrozen Backbone)")
                    print("=" * 80)

                    self.model_builder.unfreeze_backbone(num_layers=-1)
                    callbacks = self.setup_callbacks(phase='finetuning')

                    print("\n🚀 Starting fine-tuning...")
                    history_finetune = self.model.fit(
                        train_generator,
                        validation_data=val_generator,
                        epochs=initial_epochs + fine_tune_epochs,
                        initial_epoch=initial_epochs,
                        callbacks=callbacks,
                        verbose=1
                    )

                    return history_initial, history_finetune

        else:
            # Fresh training (no checkpoint)
            print("\n" + "=" * 80)
            print("STEP 3: Initial Training")
            print("=" * 80)
            print(f"Epochs: {initial_epochs}")
            print(f"Batch Size: {self.config['training']['batch_size']}")
            print("=" * 80)

            callbacks = self.setup_callbacks(phase='initial')

            print("\n🚀 Starting training...")
            history_initial = self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=initial_epochs,
                callbacks=callbacks,
                verbose=1
            )

            print("\n✅ Initial training complete!")

            # Fine-tuning
            print("\n" + "=" * 80)
            print("STEP 4: Fine-Tuning (Unfrozen Backbone)")
            print("=" * 80)

            self.model_builder.unfreeze_backbone(num_layers=-1)
            callbacks = self.setup_callbacks(phase='finetuning')

            print("\n🚀 Starting fine-tuning...")
            history_finetune = self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=initial_epochs + fine_tune_epochs,
                initial_epoch=initial_epochs,
                callbacks=callbacks,
                verbose=1
            )

            print("\n✅ Fine-tuning complete!")

            # Save final model
            final_model_path = f"{self.config['paths']['models']}/saved_models/final_model.keras"
            self.model.save(final_model_path)
            print(f"\n✅ Final model saved: {final_model_path}")

            return history_initial, history_finetune

    def evaluate(self):
        """Evaluate on test set"""

        print("\n" + "=" * 80)
        print("STEP 5: Evaluation on Test Set")
        print("=" * 80)

        test_ids, test_labels = self.splits['test']

        test_generator = OptimizedDataGenerator(
            test_ids, test_labels,
            frames_dir="/workspace/processed/frames",
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            augment=False
        )

        print("\n📊 Evaluating model...")
        results = self.model.evaluate(test_generator, verbose=1)

        print("\n" + "=" * 80)
        print("Test Set Results")
        print("=" * 80)

        metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
        for metric, value in zip(metrics, results):
            print(f"{metric.capitalize():12}: {value:.4f}")

        precision, recall = results[2], results[3]
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{'F1-Score':12}: {f1_score:.4f}")

        print("=" * 80)

        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'test_size': len(test_ids),
            'metrics': {
                'loss': float(results[0]),
                'accuracy': float(results[1]),
                'precision': float(results[2]),
                'recall': float(results[3]),
                'auc': float(results[4]),
                'f1_score': float(f1_score)
            }
        }

        results_path = f"{self.config['paths']['logs']}/evaluation/test_results.json"
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n✅ Results saved: {results_path}")

        return results_dict


def main():
    """Main training function with resume capability"""

    print("=" * 80)
    print("NexaraVision Training Pipeline with AUTO-RESUME")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        pipeline = ResumeTrainingPipeline()
        pipeline.load_data()
        pipeline.build_or_load_model()

        history = pipeline.train(
            initial_epochs=30,
            fine_tune_epochs=20
        )

        results = pipeline.evaluate()

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Final Test Accuracy: {results['metrics']['accuracy']:.2%}")
        print(f"Final Test Precision: {results['metrics']['precision']:.2%}")
        print(f"Final Test Recall: {results['metrics']['recall']:.2%}")
        print(f"Final Test F1-Score: {results['metrics']['f1_score']:.2%}")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("✅ Checkpoint saved - you can resume by running this script again!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
