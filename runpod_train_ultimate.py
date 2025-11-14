#!/usr/bin/env python3
"""
ULTIMATE Training for 93-97% Accuracy
- 10× data augmentation
- Better feature extraction (EfficientNetB4)
- Bidirectional LSTM + Attention
- 1000 epochs with cosine annealing
- Advanced regularization
"""

import os
import sys

# GPU Configuration
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from datetime import datetime

# Enable mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# GPU Setup
print("=" * 80)
print("ULTIMATE TRAINING FOR 93-97% ACCURACY")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ {len(gpus)} Physical GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

print("=" * 80)

# Import configuration
from src.config import SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES

# ULTIMATE HYPERPARAMETERS
BATCH_SIZE = 128
EPOCHS = 1000  # More epochs for convergence
LEARNING_RATE = 0.0001
AUGMENTATION_FACTOR = 10  # 10× augmentation!

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_ultimate_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

GLOBAL_BATCH_SIZE = BATCH_SIZE * len(gpus) if gpus else BATCH_SIZE


def extract_features_efficientnet(
    data_dir: str,
    subset: str = 'train',
    augmentation_factor: int = AUGMENTATION_FACTOR
):
    """
    Extract features using EfficientNetB4 (better than VGG19).
    """
    from tensorflow.keras.applications import EfficientNetB4
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import cv2

    logger.info(f"Extracting features for {subset} with EfficientNetB4...")
    logger.info(f"Augmentation factor: {augmentation_factor}x")

    # Initialize EfficientNetB4
    with tf.device('/GPU:0'):
        base_model = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
            pooling='avg'  # Global average pooling
        )
        feature_extractor = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.output
        )

    logger.info("✅ EfficientNetB4 feature extractor loaded")

    # ULTIMATE Data Augmentation
    if subset == 'train':
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            zoom_range=0.25,
            brightness_range=[0.5, 1.5],
            fill_mode='nearest',
            shear_range=0.2,
            channel_shift_range=30.0
        )
    else:
        datagen = ImageDataGenerator()

    # Get video paths
    data_path = Path(data_dir) / subset
    violence_videos = sorted(list((data_path / 'Fight').glob('*.avi')))
    nonviolence_videos = sorted(list((data_path / 'NonFight').glob('*.avi')))

    all_videos = violence_videos + nonviolence_videos
    labels = [1] * len(violence_videos) + [0] * len(nonviolence_videos)

    logger.info(f"Processing {len(all_videos)} videos")
    if subset == 'train':
        logger.info(f"Will create {len(all_videos) * augmentation_factor} total samples")

    features_list = []
    labels_list = []

    for idx, (video_path, label) in enumerate(zip(all_videos, labels)):
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                cap.release()
                continue

            frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            if len(frames) != SEQUENCE_LENGTH:
                continue

            frames = np.array(frames, dtype=np.uint8)

            # Create multiple augmented versions
            num_versions = augmentation_factor if subset == 'train' else 1

            for aug_idx in range(num_versions):
                if subset == 'train' and aug_idx > 0:
                    augmented_frames = []
                    for frame in frames:
                        frame_reshaped = frame.reshape((1,) + frame.shape)
                        aug_iter = datagen.flow(frame_reshaped, batch_size=1)
                        augmented_frame = next(aug_iter)[0].astype('uint8')
                        augmented_frames.append(augmented_frame)
                    current_frames = np.array(augmented_frames)
                else:
                    current_frames = frames.copy()

                # Preprocess for EfficientNet
                frames_preprocessed = preprocess_input(current_frames.astype('float32'))

                # Extract features
                with tf.device('/GPU:0'):
                    video_features = feature_extractor.predict(
                        frames_preprocessed,
                        batch_size=SEQUENCE_LENGTH,
                        verbose=0
                    )

                features_list.append(video_features)
                labels_list.append(label)

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(all_videos)} videos")

        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {e}")
            continue

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)

    logger.info(f"✅ Feature extraction complete:")
    logger.info(f"   X.shape = {X.shape}")
    logger.info(f"   y.shape = {y.shape}")

    return X, y


def build_ultimate_model(input_shape, num_classes=2):
    """
    Build ULTIMATE model: Bidirectional LSTM + Attention.
    """
    logger.info(f"Building ULTIMATE model with input_shape={input_shape}")

    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Bidirectional LSTM layers
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        name='bidirectional_lstm_1'
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        name='bidirectional_lstm_2'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Attention mechanism
    attention = tf.keras.layers.Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(256)(attention)  # 256 = 128*2 (bidirectional)
    attention = tf.keras.layers.Permute([2, 1])(attention)

    # Apply attention
    sent_representation = tf.keras.layers.Multiply()([x, attention])
    sent_representation = tf.keras.layers.Lambda(lambda xin: tf.keras.backend.sum(xin, axis=1))(sent_representation)

    # Dense layers
    x = tf.keras.layers.Dense(512, activation='relu')(sent_representation)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Cosine decay learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * 100,  # Approximate steps per epoch
        alpha=0.1
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("✅ ULTIMATE model built successfully")
    return model


def train_ultimate(
    data_dir: str,
    output_dir: str = './models_ultimate',
    augmentation_factor: int = AUGMENTATION_FACTOR
):
    """
    ULTIMATE training for 93-97% accuracy.
    """
    logger.info("=" * 80)
    logger.info("ULTIMATE TRAINING - TARGET: 93-97% ACCURACY")
    logger.info("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # PHASE 1: Extract training features
    logger.info("\nPHASE 1: FEATURE EXTRACTION - TRAINING SET (EfficientNetB4 + 10× augmentation)")
    X_train, y_train = extract_features_efficientnet(
        data_dir=data_dir,
        subset='train',
        augmentation_factor=augmentation_factor
    )

    logger.info(f"Training data: X_train={X_train.shape}, y_train={y_train.shape}")

    # PHASE 2: Extract validation features
    logger.info("\nPHASE 2: FEATURE EXTRACTION - VALIDATION SET")
    X_val, y_val = extract_features_efficientnet(
        data_dir=data_dir,
        subset='val',
        augmentation_factor=1
    )

    logger.info(f"Validation data: X_val={X_val.shape}, y_val={y_val.shape}")

    # PHASE 3: Build and train model
    logger.info("\nPHASE 3: ULTIMATE MODEL TRAINING")

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"✅ Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_ultimate_model(input_shape, num_classes=NUM_CLASSES)
        model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_path / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=1e-8,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,  # Very patient
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(output_path / 'training_log.csv')
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_path / 'tensorboard_logs'),
            histogram_freq=1
        )
    ]

    # Train
    logger.info("\nStarting ULTIMATE training:")
    logger.info(f"  Epochs: {EPOCHS}")
    logger.info(f"  Batch size: {GLOBAL_BATCH_SIZE}")
    logger.info(f"  Training samples: {X_train.shape[0]}")
    logger.info(f"  Validation samples: {X_val.shape[0]}")
    logger.info(f"  Augmentation: {augmentation_factor}×")
    logger.info(f"  Target accuracy: 93-97%")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=GLOBAL_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(str(output_path / 'final_model.h5'))
    logger.info(f"✅ Model saved to {output_path}")

    # Final evaluation
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 80)

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    logger.info(f"Training Accuracy:   {train_acc:.4f}")
    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    logger.info(f"Improvement over baseline: +{(val_acc - 0.7575) * 100:.2f}%")

    if val_acc >= 0.93:
        logger.info("🎉 TARGET ACHIEVED: 93%+ accuracy!")
    elif val_acc >= 0.90:
        logger.info("✅ EXCELLENT: 90%+ accuracy achieved!")
    elif val_acc >= 0.85:
        logger.info("✅ GREAT: 85%+ accuracy achieved!")

    return history, model


def main():
    """Main entry point"""

    DATA_DIR = "/workspace/data/combined"  # Will update after combining all datasets
    OUTPUT_DIR = "./models_ultimate"

    logger.info("=" * 80)
    logger.info("ULTIMATE VIOLENCE DETECTION TRAINING")
    logger.info("=" * 80)
    logger.info(f"Dataset: {DATA_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Augmentation: {AUGMENTATION_FACTOR}×")
    logger.info(f"Target: 93-97% accuracy")
    logger.info("=" * 80)

    if not Path(DATA_DIR).exists():
        logger.error(f"Dataset not found at: {DATA_DIR}")
        sys.exit(1)

    try:
        history, model = train_ultimate(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            augmentation_factor=AUGMENTATION_FACTOR
        )

        logger.info("\n" + "=" * 80)
        logger.info("✅ ULTIMATE TRAINING COMPLETE!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
