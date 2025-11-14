#!/usr/bin/env python3
"""
NexaraVision Training Script
Main training pipeline with callbacks and monitoring
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

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


class TrainingPipeline:
    """Complete training pipeline"""

    def __init__(self, config_path="/workspace/training_config.json"):
        """Initialize training pipeline"""

        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)

        print("=" * 80)
        print("NexaraVision Training Pipeline")
        print("=" * 80)
        print(f"Loaded config from: {config_path}\n")

        self.preprocessor = None
        self.model_builder = None
        self.model = None
        self.splits = None

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

        # Scan datasets
        total_videos = self.preprocessor.scan_datasets()

        if total_videos == 0:
            raise ValueError("No videos found!")

        # Create splits
        self.splits = self.preprocessor.create_splits()

        # Save splits
        self.preprocessor.save_splits(
            self.splits,
            self.config['paths']['processed'] + '/splits.json'
        )

        return self.splits

    def build_model(self):
        """Build and compile model"""

        print("\n" + "=" * 80)
        print("STEP 2: Model Building")
        print("=" * 80)

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

        # Save architecture
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

        # Model checkpoint
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

        # TensorBoard
        log_dir = f"{self.config['paths']['logs']}/training/{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        print(f"✅ TensorBoard: {log_dir}")

        # CSV Logger
        csv_path = f"{self.config['paths']['logs']}/training/{phase}_training_log.csv"
        csv_logger = tf.keras.callbacks.CSVLogger(csv_path)
        callbacks.append(csv_logger)
        print(f"✅ CSVLogger: {csv_path}")

        print("=" * 80)

        return callbacks

    def train(self, initial_epochs=50, fine_tune_epochs=50):
        """
        Train model with transfer learning

        Args:
            initial_epochs: Epochs for initial training (frozen backbone)
            fine_tune_epochs: Epochs for fine-tuning (unfrozen backbone)
        """

        if self.splits is None or self.model is None:
            raise ValueError("Must prepare data and build model first")

        # Unpack splits
        X_train, y_train, _ = self.splits['train']
        X_val, y_val, _ = self.splits['val']

        # ===================================================================
        # PHASE 1: Initial Training (Frozen Backbone)
        # ===================================================================

        print("\n" + "=" * 80)
        print("STEP 3: Initial Training (Transfer Learning)")
        print("=" * 80)
        print(f"Epochs: {initial_epochs}")
        print(f"Batch Size: {self.config['training']['batch_size']}")
        print(f"Backbone: FROZEN (transfer learning)")
        print("=" * 80)

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

        # Setup callbacks
        callbacks = self.setup_callbacks(phase='initial')

        # Train
        print("\n🚀 Starting initial training...")
        history_initial = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=initial_epochs,
            callbacks=callbacks,
            verbose=1
        )

        print("\n✅ Initial training complete!")

        # ===================================================================
        # PHASE 2: Fine-Tuning (Unfrozen Backbone)
        # ===================================================================

        print("\n" + "=" * 80)
        print("STEP 4: Fine-Tuning (Unfrozen Backbone)")
        print("=" * 80)
        print(f"Epochs: {fine_tune_epochs}")
        print(f"Learning Rate: 1e-5 (reduced for fine-tuning)")
        print(f"Backbone: UNFROZEN (fine-tuning last layers)")
        print("=" * 80)

        # Unfreeze backbone
        self.model_builder.unfreeze_backbone(num_layers=-1)

        # Setup callbacks for fine-tuning
        callbacks = self.setup_callbacks(phase='finetuning')

        # Continue training
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

        results_path = f"{self.config['paths']['logs']}/evaluation/test_results.json"
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n✅ Results saved: {results_path}")

        return results_dict


def main():
    """Main training function"""

    print("=" * 80)
    print("NexaraVision Training Pipeline")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        # Initialize pipeline
        pipeline = TrainingPipeline()

        # Prepare data
        pipeline.prepare_data()

        # Build model
        pipeline.build_model()

        # Train model
        history = pipeline.train(
            initial_epochs=30,  # Start with 30 epochs
            fine_tune_epochs=20  # Then 20 epochs of fine-tuning
        )

        # Evaluate
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
