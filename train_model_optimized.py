#!/usr/bin/env python3
"""
NexaraVision Training Script (OPTIMIZED)
Uses pre-extracted frames for 10x faster training
"""

import os
import sys

# ⚠️ CRITICAL: Set environment variables BEFORE importing TensorFlow
# This forces single GPU mode and prevents OOM errors
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use only GPU 1 (GPU 0 is occupied)
    print("🎮 Forced CUDA_VISIBLE_DEVICES=1 (using GPU 1, GPU 0 is occupied)")

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Dynamic memory allocation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Better memory management

# Now import TensorFlow (will only see GPU 0)
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from model_architecture import ViolenceDetectionModel

# GPU Configuration for Multi-GPU Systems
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
            # Enable memory growth (prevents TensorFlow from allocating all VRAM)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth enabled (dynamic allocation)")

            # Use only first GPU to avoid multi-GPU configuration issues
            # For distributed training, comment out this line
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

# Setup GPU before importing model
setup_gpu()

class OptimizedDataGenerator(tf.keras.utils.Sequence):
    """Fast data generator using pre-extracted frames"""

    def __init__(self, video_ids, labels, frames_dir, batch_size=32, shuffle=True, augment=False):
        """
        Initialize optimized generator

        Args:
            video_ids: List of video IDs
            labels: List of labels
            frames_dir: Directory containing pre-extracted .npy files
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augment: Whether to apply augmentation
        """
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
            # Load pre-extracted frames (FAST!)
            frames_file = self.frames_dir / f"{self.video_ids[i]}.npy"

            try:
                frames = np.load(frames_file)

                # Apply augmentation if enabled
                if self.augment:
                    frames = self._augment_frames(frames)

                batch_videos.append(frames)
                batch_labels.append(self.labels[i])

            except Exception as e:
                print(f"⚠️  Error loading {frames_file}: {e}")
                # Add black frames as fallback
                batch_videos.append(np.zeros((20, 224, 224, 3), dtype=np.float32))
                batch_labels.append(self.labels[i])

        X = np.array(batch_videos)
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=2)

        return X, y

    def _augment_frames(self, frames):
        """Apply data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            frames = np.flip(frames, axis=2)

        # Random brightness adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * factor, 0, 1)

        return frames

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class OptimizedTrainingPipeline:
    """Optimized training pipeline using pre-extracted frames"""

    def __init__(self, config_path="/workspace/training_config.json"):
        with open(config_path) as f:
            self.config = json.load(f)

        print("=" * 80)
        print("NexaraVision OPTIMIZED Training Pipeline")
        print("=" * 80)
        print(f"Loaded config from: {config_path}\n")

        # Enable mixed precision if configured
        if self.config.get('training', {}).get('mixed_precision', False):
            print("🚀 Enabling Mixed Precision Training (FP16)")
            print("   Memory usage reduced by ~40%\n")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        self.model_builder = None
        self.model = None

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

        # Create video ID mappings
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

    def build_model(self):
        """Build and compile model"""

        print("\n" + "=" * 80)
        print("STEP 2: Model Building")
        print("=" * 80)

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
        self.model_builder.save_architecture(
            self.config['paths']['models'] + '/architecture_config.json'
        )

        return self.model

    def setup_callbacks(self, phase='initial'):
        """Setup training callbacks"""

        print("\n" + "=" * 80)
        print("Setting Up Callbacks")
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
        csv_logger = tf.keras.callbacks.CSVLogger(csv_path)
        callbacks.append(csv_logger)
        print(f"✅ CSVLogger: {csv_path}")

        print("=" * 80)

        return callbacks

    def train(self, initial_epochs=30, fine_tune_epochs=20):
        """Train model with optimized data loading"""

        if self.splits is None or self.model is None:
            raise ValueError("Must load data and build model first")

        train_ids, train_labels = self.splits['train']
        val_ids, val_labels = self.splits['val']

        print("\n" + "=" * 80)
        print("STEP 3: Initial Training (OPTIMIZED)")
        print("=" * 80)
        print(f"Epochs: {initial_epochs}")
        print(f"Batch Size: {self.config['training']['batch_size']}")
        print(f"Data Loading: PRE-EXTRACTED FRAMES (10x faster!)")
        print("=" * 80)

        # Create optimized generators
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

        callbacks = self.setup_callbacks(phase='initial')

        print("\n🚀 Starting optimized training...")
        print("   (Much faster than before!)\n")

        history_initial = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=initial_epochs,
            callbacks=callbacks,
            verbose=1
        )

        print("\n✅ Initial training complete!")

        # Fine-tuning phase
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
    """Main optimized training function"""

    print("=" * 80)
    print("NexaraVision OPTIMIZED Training Pipeline")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        pipeline = OptimizedTrainingPipeline()
        pipeline.load_data()
        pipeline.build_model()

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
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
