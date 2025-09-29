"""
Model architecture module for Violence Detection MVP.
Implements LSTM with Attention mechanism for video sequence classification.
"""

import numpy as np
import random
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization,
    Activation, Layer
)
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from .config import Config

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seeds(seed: int = 1) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


class AttentionLayer(Layer):
    """
    Custom attention layer for focusing on important frames in video sequences.
    """

    def __init__(self, **kwargs):
        """Initialize the attention layer."""
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build the attention layer."""
        # Create a trainable weight variable for attention
        self.attention_dense = Dense(1, use_bias=False)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Apply attention mechanism.

        Args:
            inputs: Input tensor with shape (batch_size, time_steps, features)

        Returns:
            Context vector with shape (batch_size, features)
        """
        # Calculate attention scores
        attention_scores = self.attention_dense(inputs)  # (batch_size, time_steps, 1)

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # (batch_size, time_steps, 1)

        # Calculate weighted sum (context vector)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)  # (batch_size, features)

        return context_vector

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return (input_shape[0], input_shape[2])


class ViolenceDetectionModel:
    """
    Violence detection model with LSTM and Attention mechanism.
    """

    def __init__(self, config: Config = Config):
        """Initialize the model with configuration."""
        self.config = config
        self.model: Optional[Model] = None
        self.is_compiled: bool = False

        # Set seeds for reproducibility
        set_seeds(1)

    def build_model(self) -> Model:
        """
        Build the LSTM-Attention model for violence detection.

        Returns:
            Compiled Keras model
        """
        logger.info("Building LSTM-Attention model...")

        # Input layer
        input_seq = Input(
            shape=(self.config.N_CHUNKS, self.config.CHUNK_SIZE),
            name='video_input'
        )

        # First LSTM layer
        x = LSTM(
            self.config.RNN_SIZE,
            return_sequences=True,
            name='lstm_1'
        )(input_seq)
        x = BatchNormalization(name='bn_1')(x)
        x = Dropout(self.config.DROPOUT_RATE, name='dropout_1')(x)

        # Second LSTM layer
        x = LSTM(
            self.config.RNN_SIZE,
            return_sequences=True,
            name='lstm_2'
        )(x)
        x = BatchNormalization(name='bn_2')(x)
        x = Dropout(self.config.DROPOUT_RATE, name='dropout_2')(x)

        # Third LSTM layer
        x = LSTM(
            self.config.RNN_SIZE,
            return_sequences=True,
            name='lstm_3'
        )(x)
        x = BatchNormalization(name='bn_3')(x)
        x = Dropout(self.config.DROPOUT_RATE, name='dropout_3')(x)

        # Attention mechanism
        attention_output = AttentionLayer(name='attention')(x)

        # Dense layers after attention
        x = Dense(256, name='dense_1')(attention_output)
        x = BatchNormalization(name='bn_4')(x)
        x = Activation('relu', name='relu_1')(x)
        x = Dropout(self.config.DROPOUT_RATE, name='dropout_4')(x)

        x = Dense(128, name='dense_2')(x)
        x = BatchNormalization(name='bn_5')(x)
        x = Activation('relu', name='relu_2')(x)
        x = Dropout(self.config.DROPOUT_RATE, name='dropout_5')(x)

        x = Dense(64, name='dense_3')(x)
        x = Activation('relu', name='relu_3')(x)
        x = Dropout(self.config.DROPOUT_RATE, name='dropout_6')(x)

        # Output layer
        output = Dense(
            self.config.NUM_CLASSES,
            activation='softmax',
            name='output'
        )(x)

        # Create model
        model = Model(inputs=input_seq, outputs=output, name='violence_detection_model')

        logger.info("Model architecture built successfully")
        return model

    def compile_model(self, model: Model) -> Model:
        """
        Compile the model with optimizer and loss function.

        Args:
            model: Keras model to compile

        Returns:
            Compiled model
        """
        optimizer = Adam(learning_rate=self.config.LEARNING_RATE)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        self.is_compiled = True
        logger.info(f"Model compiled with learning rate: {self.config.LEARNING_RATE}")

        return model

    def create_model(self) -> Model:
        """
        Create and compile the complete model.

        Returns:
            Compiled Keras model ready for training
        """
        model = self.build_model()
        model = self.compile_model(model)
        self.model = model

        return model

    def get_model_summary(self) -> str:
        """
        Get a string representation of the model summary.

        Returns:
            Model summary as string
        """
        if self.model is None:
            self.create_model()

        # Capture model summary
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)

    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")

        self.model.save(filepath)
        logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: str) -> Model:
        """
        Load a model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded Keras model
        """
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        self.is_compiled = True
        logger.info(f"Model loaded from: {filepath}")

        return self.model


class ModelBuilder:
    """
    Factory class for building different model architectures.
    """

    @staticmethod
    def build_simple_lstm(config: Config) -> Model:
        """
        Build a simple LSTM model without attention.

        Args:
            config: Configuration object

        Returns:
            Simple LSTM model
        """
        model = Sequential([
            Input(shape=(config.N_CHUNKS, config.CHUNK_SIZE)),
            LSTM(config.RNN_SIZE, return_sequences=True),
            Dropout(config.DROPOUT_RATE),
            LSTM(config.RNN_SIZE),
            Dropout(config.DROPOUT_RATE),
            Dense(128, activation='relu'),
            Dropout(config.DROPOUT_RATE),
            Dense(config.NUM_CLASSES, activation='softmax')
        ], name='simple_lstm_model')

        return model

    @staticmethod
    def build_bidirectional_lstm(config: Config) -> Model:
        """
        Build a bidirectional LSTM model.

        Args:
            config: Configuration object

        Returns:
            Bidirectional LSTM model
        """
        from tensorflow.keras.layers import Bidirectional

        input_seq = Input(shape=(config.N_CHUNKS, config.CHUNK_SIZE))

        x = Bidirectional(LSTM(config.RNN_SIZE, return_sequences=True))(input_seq)
        x = Dropout(config.DROPOUT_RATE)(x)
        x = Bidirectional(LSTM(config.RNN_SIZE))(x)
        x = Dropout(config.DROPOUT_RATE)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(config.DROPOUT_RATE)(x)
        output = Dense(config.NUM_CLASSES, activation='softmax')(x)

        model = Model(inputs=input_seq, outputs=output, name='bidirectional_lstm_model')
        return model

    @staticmethod
    def build_gru_attention(config: Config) -> Model:
        """
        Build a GRU model with attention mechanism.

        Args:
            config: Configuration object

        Returns:
            GRU-Attention model
        """
        from tensorflow.keras.layers import GRU

        input_seq = Input(shape=(config.N_CHUNKS, config.CHUNK_SIZE))

        x = GRU(config.RNN_SIZE, return_sequences=True)(input_seq)
        x = BatchNormalization()(x)
        x = Dropout(config.DROPOUT_RATE)(x)

        x = GRU(config.RNN_SIZE, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(config.DROPOUT_RATE)(x)

        # Attention
        attention_output = AttentionLayer()(x)

        x = Dense(128, activation='relu')(attention_output)
        x = Dropout(config.DROPOUT_RATE)(x)
        output = Dense(config.NUM_CLASSES, activation='softmax')(x)

        model = Model(inputs=input_seq, outputs=output, name='gru_attention_model')
        return model


def create_callbacks(config: Config) -> list:
    """
    Create training callbacks.

    Args:
        config: Configuration object

    Returns:
        List of Keras callbacks
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            verbose=1,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.get_model_path('checkpoint')),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    return callbacks


def get_model_metrics(model: Model) -> dict:
    """
    Get model architecture metrics.

    Args:
        model: Keras model

    Returns:
        Dictionary containing model metrics
    """
    total_params = model.count_params()
    trainable_params = sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([K.count_params(w) for w in model.non_trainable_weights])

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Approximate size in MB
        'num_layers': len(model.layers)
    }


def validate_model_architecture(config: Config) -> dict:
    """
    Validate the model architecture with dummy data.

    Args:
        config: Configuration object

    Returns:
        Dictionary containing validation results
    """
    try:
        # Create model
        model_builder = ViolenceDetectionModel(config)
        model = model_builder.create_model()

        # Create dummy input
        dummy_input = np.random.random((1, config.N_CHUNKS, config.CHUNK_SIZE))

        # Test forward pass
        output = model.predict(dummy_input, verbose=0)

        # Get metrics
        metrics = get_model_metrics(model)

        return {
            'success': True,
            'output_shape': output.shape,
            'output_sum': float(np.sum(output)),
            'metrics': metrics
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }