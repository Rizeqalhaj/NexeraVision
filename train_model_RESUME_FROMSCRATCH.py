#!/usr/bin/env python3
"""
NexaraVision Training Script with RESUME capability
Compatible with train_model.py (on-the-fly frame extraction)
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
import os
import pandas as pd

from data_preprocessing import VideoDataPreprocessor
from model_architecture import ViolenceDetectionModel

class VideoDataGenerator(tf.keras.utils.Sequence):
    """Data generator for video sequences"""

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
            # Extract frames
            frames = self.preprocessor.extract_frames(self.video_paths[i])

            # Apply augmentation if enabled
            if self.augment:
                frames = self._augment_frames(frames)

            batch_videos.append(frames)
            batch_labels.append(self.labels[i])

        # Convert to numpy arrays
        X = np.array(batch_videos)
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=2)

        return X, y

    def _augment_frames(self, frames):
        """Apply data augmentation to frames"""
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


class ResumeTrainingPipeline:
    """Training pipeline with automatic checkpoint resumption"""

    def __init__(self, config_path="/workspace/training_config.json"):
        """Initialize training pipeline"""

        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)

        print("=" * 80)
        print("NexaraVision Training Pipeline with AUTO-RESUME")
        print("=" * 80)
        print(f"Loaded config from: {config_path}\n")

        self.preprocessor = None
        self.model_builder = None
        self.model = None
        self.splits = None
        self.resume_info = None

    def find_latest_checkpoint(self):
        """Find the most recent checkpoint"""
        # Use explicit vast.ai checkpoint path
        checkpoint_dir = Path('/workspace/checkpoints')

        if not checkpoint_dir.exists():
            print(f"⚠️  Checkpoint directory not found: {checkpoint_dir}")
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
                log_path = Path('/workspace/logs/training') / log_name

                resume_info = {
                    'checkpoint_path': str(checkpoint_path),
                    'log_path': str(log_path) if log_path.exists() else None,
                    'phase': 'finetuning' if 'finetuning' in checkpoint_path.name else 'initial'
                }

                return resume_info

        return None

    def prepare_data(self):
        """Prepare and split data"""

        print("\n" + "=" * 80)
        print("STEP 1: Data Preparation")
        print("=" * 80)

        # Initialize preprocessor
        self.preprocessor = VideoDataPreprocessor(
            dataset_dir=self.config['paths']['datasets'],
            frames_per_video=self.config['training']['frames_per_video'],
            train_split=self.config['data']['train_split'],
            val_split=self.config['data']['val_split'],
            test_split=self.config['data']['test_split']
        )

        # Check if splits already exist
        splits_path = Path(self.config['paths']['processed']) / 'splits.json'

        if splits_path.exists():
            print(f"\n✅ Loading existing splits from: {splits_path}")
            self.splits = self.preprocessor.load_splits(str(splits_path))
            print("✅ Splits loaded successfully!")
        else:
            print("\n🆕 Creating new splits...")
            # Scan datasets
            total_videos = self.preprocessor.scan_datasets()

            if total_videos == 0:
                raise ValueError("No videos found!")

            # Create splits
            self.splits = self.preprocessor.create_splits()

            # Save splits
            self.preprocessor.save_splits(self.splits, str(splits_path))

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
                self.model_builder = ViolenceDetectionModel(
                    frames_per_video=self.config['training']['frames_per_video'],
                    sequence_model=self.config['model']['sequence_model'],
                    gru_units=self.config['model']['gru_units'],
                    dense_layers=self.config['model']['dense_layers'],
                    dropout_rates=self.config['model']['dropout']
                )
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
        # Initialize model builder
        self.model_builder = ViolenceDetectionModel(
            frames_per_video=self.config['training']['frames_per_video'],
            sequence_model=self.config['model']['sequence_model'],
            gru_units=self.config['model']['gru_units'],
            dense_layers=self.config['model']['dense_layers'],
            dropout_rates=self.config['model']['dropout']
        )

        # Build model (frozen backbone for transfer learning)
        self.model = self.model_builder.build_model(trainable_backbone=False)

        # Compile model
        self.model_builder.compile_model(
            learning_rate=self.config['training']['learning_rate'],
            optimizer=self.config['training']['optimizer'],
            loss=self.config['training']['loss']
        )

        # Print summary
        self.model_builder.print_summary()
        self.model_builder.count_parameters()

        # Save architecture - use vast.ai path
        arch_path = Path('/workspace/models/architecture_config.json')
        arch_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_builder.save_architecture(str(arch_path))

    def setup_callbacks(self, phase='initial'):
        """Setup training callbacks"""

        print("\n" + "=" * 80)
        print(f"Setting Up Callbacks ({phase})")
        print("=" * 80)

        callbacks = []

        # Model checkpoint - use vast.ai path
        checkpoint_path = f"/workspace/checkpoints/{phase}_best_model.keras"
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        print(f"✅ ModelCheckpoint: {checkpoint_path}")

        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        print(f"✅ EarlyStopping: patience={self.config['training']['early_stopping_patience']}")

        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        print(f"✅ ReduceLROnPlateau: factor=0.5, patience=10")

        # TensorBoard - use vast.ai path
        log_dir = f"/workspace/logs/training/{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        print(f"✅ TensorBoard: {log_dir}")

        # CSV Logger - use vast.ai path
        csv_path = f"/workspace/logs/training/{phase}_training_log.csv"
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

        csv_logger = tf.keras.callbacks.CSVLogger(csv_path, append=True)  # APPEND for resume
        callbacks.append(csv_logger)
        print(f"✅ CSVLogger: {csv_path} (append mode)")

        print("=" * 80)

        return callbacks

    def train(self, initial_epochs=50, fine_tune_epochs=50):
        """
        Train model with resume capability

        Args:
            initial_epochs: Epochs for initial training (frozen backbone)
            fine_tune_epochs: Epochs for fine-tuning (unfrozen backbone)
        """

        if self.splits is None or self.model is None:
            raise ValueError("Must prepare data and build model first")

        # Unpack splits
        X_train, y_train, _ = self.splits['train']
        X_val, y_val, _ = self.splits['val']

        # Create data generators
        train_generator = VideoDataGenerator(
            X_train, y_train,
            self.preprocessor,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            augment=self.config['data']['augmentation']
        )

        val_generator = VideoDataGenerator(
            X_val, y_val,
            self.preprocessor,
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

                    # Save final model - use vast.ai path
                    final_model_path = "/workspace/models/saved_models/final_model.keras"
                    Path(final_model_path).parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(final_model_path)
                    print(f"\n✅ Final model saved: {final_model_path}")

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

                    # Save final model - use vast.ai path
                    final_model_path = "/workspace/models/saved_models/final_model.keras"
                    Path(final_model_path).parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(final_model_path)
                    print(f"\n✅ Final model saved: {final_model_path}")

                    return history_initial, history_finetune

        else:
            # Fresh training (no checkpoint)
            print("\n" + "=" * 80)
            print("STEP 3: Initial Training (Transfer Learning)")
            print("=" * 80)
            print(f"Epochs: {initial_epochs}")
            print(f"Batch Size: {self.config['training']['batch_size']}")
            print(f"Backbone: FROZEN (transfer learning)")
            print("=" * 80)

            callbacks = self.setup_callbacks(phase='initial')

            print("\n🚀 Starting initial training...")
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
            print(f"Epochs: {fine_tune_epochs}")
            print(f"Learning Rate: 1e-5 (reduced for fine-tuning)")
            print(f"Backbone: UNFROZEN (fine-tuning all layers)")
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

            # Save final model - use vast.ai path
            final_model_path = "/workspace/models/saved_models/final_model.keras"
            Path(final_model_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(final_model_path)
            print(f"\n✅ Final model saved: {final_model_path}")

            return history_initial, history_finetune

    def evaluate(self):
        """Evaluate model on test set"""

        if self.model is None or self.splits is None:
            raise ValueError("Must build and train model first")

        print("\n" + "=" * 80)
        print("STEP 5: Evaluation on Test Set")
        print("=" * 80)

        X_test, y_test, _ = self.splits['test']

        # Create test generator
        test_generator = VideoDataGenerator(
            X_test, y_test,
            self.preprocessor,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            augment=False
        )

        # Evaluate
        print("\n📊 Evaluating model...")
        results = self.model.evaluate(test_generator, verbose=1)

        # Print results
        print("\n" + "=" * 80)
        print("Test Set Results")
        print("=" * 80)

        metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
        for metric, value in zip(metrics, results):
            print(f"{metric.capitalize():12}: {value:.4f}")

        # Calculate F1 score
        precision, recall = results[2], results[3]
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{'F1-Score':12}: {f1_score:.4f}")

        print("=" * 80)

        # Save results
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'test_size': len(X_test),
            'metrics': {
                'loss': float(results[0]),
                'accuracy': float(results[1]),
                'precision': float(results[2]),
                'recall': float(results[3]),
                'auc': float(results[4]),
                'f1_score': float(f1_score)
            }
        }

        results_path = "/workspace/logs/evaluation/test_results.json"
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
        pipeline.prepare_data()
        pipeline.build_or_load_model()

        history = pipeline.train(
            initial_epochs=50,
            fine_tune_epochs=50
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
