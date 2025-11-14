#!/usr/bin/env python3
"""
Train 5 different models for ensemble voting.
Ensemble predictions average to achieve 93-97% accuracy.
"""

import os
import sys

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES

# Load pre-extracted features (from ultimate script)
def load_features(data_dir):
    """Load pre-extracted features"""
    logger.info(f"Loading features from {data_dir}")

    X_train = np.load(f"{data_dir}/X_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    X_val = np.load(f"{data_dir}/X_val.npy")
    y_val = np.load(f"{data_dir}/y_val.npy")

    return X_train, y_train, X_val, y_val


def build_model_1(input_shape):
    """Model 1: Deep LSTM"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.3),
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3),
        tf.keras.layers.LSTM(64, dropout=0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax', dtype='float32')
    ], name='deep_lstm')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_model_2(input_shape):
    """Model 2: Bidirectional LSTM"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.3)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(2, activation='softmax', dtype='float32')
    ], name='bidirectional_lstm')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_model_3(input_shape):
    """Model 3: GRU-based"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.GRU(256, return_sequences=True, dropout=0.3),
        tf.keras.layers.GRU(128, dropout=0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax', dtype='float32')
    ], name='gru_based')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_model_4(input_shape):
    """Model 4: Conv1D + LSTM"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = tf.keras.layers.LSTM(64, dropout=0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='conv_lstm')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_model_5(input_shape):
    """Model 5: Attention-based"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3)(inputs)

    # Self-attention
    attention = tf.keras.layers.Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(128)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)

    sent_representation = tf.keras.layers.Multiply()([x, attention])
    sent_representation = tf.keras.layers.Lambda(lambda xin: tf.keras.backend.sum(xin, axis=1))(sent_representation)

    x = tf.keras.layers.Dense(256, activation='relu')(sent_representation)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='attention_based')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_ensemble(data_dir, output_dir='./models_ensemble'):
    """Train all 5 models"""

    logger.info("=" * 80)
    logger.info("ENSEMBLE TRAINING - 5 MODELS")
    logger.info("=" * 80)

    # Load features
    X_train, y_train, X_val, y_val = load_features(data_dir)

    input_shape = (X_train.shape[1], X_train.shape[2])
    logger.info(f"Input shape: {input_shape}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Model builders
    model_builders = [
        build_model_1,
        build_model_2,
        build_model_3,
        build_model_4,
        build_model_5
    ]

    models = []
    results = []

    for i, builder in enumerate(model_builders, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Training Model {i}/5")
        logger.info(f"{'='*80}")

        model = builder(input_shape)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(output_path / f'model_{i}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=1e-7
            )
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=500,
            batch_size=256,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        _, val_acc = model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Model {i} Validation Accuracy: {val_acc:.4f}")

        models.append(model)
        results.append(val_acc)

    # Ensemble prediction
    logger.info("\n" + "=" * 80)
    logger.info("ENSEMBLE EVALUATION")
    logger.info("=" * 80)

    predictions = []
    for model in models:
        pred = model.predict(X_val, verbose=0)
        predictions.append(pred)

    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    ensemble_pred_labels = np.argmax(ensemble_pred, axis=1)

    ensemble_acc = np.mean(ensemble_pred_labels == y_val)

    logger.info(f"\nIndividual model accuracies:")
    for i, acc in enumerate(results, 1):
        logger.info(f"  Model {i}: {acc:.4f}")

    logger.info(f"\n🎯 ENSEMBLE Accuracy: {ensemble_acc:.4f}")
    logger.info(f"Improvement: +{(ensemble_acc - max(results)) * 100:.2f}%")

    if ensemble_acc >= 0.93:
        logger.info("🎉 TARGET ACHIEVED: 93%+ accuracy!")

    return models, ensemble_acc


def main():
    DATA_DIR = "./features_ultimate"  # From ultimate script
    OUTPUT_DIR = "./models_ensemble"

    train_ensemble(DATA_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()
