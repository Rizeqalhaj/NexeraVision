#!/usr/bin/env python3
"""
NexaraVision Model Architecture
ResNet50V2 + Bidirectional LSTM/GRU for violence detection
Based on research achieving 96-100% accuracy
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import ResNet50V2
import json

class ViolenceDetectionModel:
    """ResNet50V2 + Bi-LSTM model for violence detection"""

    def __init__(self,
                 frames_per_video=20,
                 img_size=(224, 224),
                 sequence_model='GRU',
                 gru_units=128,
                 dense_layers=[256, 128, 64],
                 dropout_rates=[0.5, 0.5, 0.5],
                 num_classes=2):
        """
        Initialize model architecture

        Args:
            frames_per_video: Number of frames in each video sequence
            img_size: Input image size (height, width)
            sequence_model: 'GRU' or 'LSTM'
            gru_units: Number of GRU/LSTM units
            dense_layers: List of dense layer sizes
            dropout_rates: List of dropout rates for each dense layer
            num_classes: Number of output classes (2 for violence/non-violence)
        """
        self.frames_per_video = frames_per_video
        self.img_size = img_size
        self.sequence_model_type = sequence_model
        self.gru_units = gru_units
        self.dense_layers = dense_layers
        self.dropout_rates = dropout_rates
        self.num_classes = num_classes

        self.model = None

    def build_model(self, trainable_backbone=False):
        """
        Build complete model architecture

        Args:
            trainable_backbone: Whether ResNet50V2 backbone is trainable

        Returns:
            Keras Model
        """
        print("=" * 80)
        print("Building Model Architecture")
        print("=" * 80)
        print(f"\nBackbone: ResNet50V2 (ImageNet pretrained)")
        print(f"Sequence Model: Bidirectional-{self.sequence_model_type}")
        print(f"Temporal Units: {self.gru_units}")
        print(f"Dense Layers: {self.dense_layers}")
        print(f"Dropout: {self.dropout_rates}")
        print(f"Output Classes: {self.num_classes}")
        print()

        # Input: (batch, frames, height, width, channels)
        input_shape = (self.frames_per_video, *self.img_size, 3)
        inputs = layers.Input(shape=input_shape, name='video_input')

        # ===================================================================
        # SPATIAL FEATURE EXTRACTION (ResNet50V2)
        # ===================================================================

        # Load pretrained ResNet50V2 (without top classification layer)
        backbone = ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.img_size, 3),
            pooling='avg'  # Global average pooling
        )

        # Freeze backbone for transfer learning
        backbone.trainable = trainable_backbone

        # Apply backbone to each frame using TimeDistributed
        # This processes all frames independently
        x = layers.TimeDistributed(
            backbone,
            name='resnet50v2_features'
        )(inputs)

        print(f"✅ ResNet50V2 backbone loaded")
        print(f"   Output shape: (batch, {self.frames_per_video}, 2048)")
        print(f"   Trainable: {trainable_backbone}")

        # ===================================================================
        # TEMPORAL MODELING (Bidirectional GRU/LSTM)
        # ===================================================================

        if self.sequence_model_type == 'GRU':
            x = layers.Bidirectional(
                layers.GRU(
                    self.gru_units,
                    return_sequences=False,
                    dropout=0.2,
                    recurrent_dropout=0.2
                ),
                name='bidirectional_gru'
            )(x)
        else:  # LSTM
            x = layers.Bidirectional(
                layers.LSTM(
                    self.gru_units,
                    return_sequences=False,
                    dropout=0.2,
                    recurrent_dropout=0.2
                ),
                name='bidirectional_lstm'
            )(x)

        print(f"✅ Bidirectional-{self.sequence_model_type} added")
        print(f"   Output shape: (batch, {self.gru_units * 2})")

        # ===================================================================
        # CLASSIFICATION HEAD
        # ===================================================================

        # Dense layers with dropout
        for i, (units, dropout) in enumerate(zip(self.dense_layers, self.dropout_rates)):
            x = layers.Dense(
                units,
                activation='relu',
                name=f'dense_{i+1}'
            )(x)

            x = layers.Dropout(dropout, name=f'dropout_{i+1}')(x)

            print(f"✅ Dense layer {i+1}: {units} units, dropout={dropout}")

        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)

        print(f"✅ Output layer: {self.num_classes} classes (softmax)")

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='ViolenceDetectionModel')

        self.model = model

        print("\n" + "=" * 80)
        print("✅ Model Built Successfully!")
        print("=" * 80)

        return model

    def compile_model(self,
                     learning_rate=0.0005,
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=None):
        """
        Compile model with optimizer and loss

        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            loss: Loss function
            metrics: List of metrics to track
        """
        if self.model is None:
            raise ValueError("Model must be built before compiling")

        if metrics is None:
            metrics = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]

        # Create optimizer
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )

        print("\n" + "=" * 80)
        print("Model Compiled")
        print("=" * 80)
        print(f"Optimizer: {optimizer}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Loss: {loss}")
        print(f"Metrics: {[m if isinstance(m, str) else m.name for m in metrics]}")
        print("=" * 80)

    def print_summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model must be built first")

        print("\n" + "=" * 80)
        print("MODEL SUMMARY")
        print("=" * 80)
        self.model.summary()
        print("=" * 80)

    def count_parameters(self):
        """Count trainable and non-trainable parameters"""
        if self.model is None:
            raise ValueError("Model must be built first")

        trainable = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable = sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        total = trainable + non_trainable

        print("\n" + "=" * 80)
        print("MODEL PARAMETERS")
        print("=" * 80)
        print(f"Trainable:     {trainable:,}")
        print(f"Non-trainable: {non_trainable:,}")
        print(f"Total:         {total:,}")
        print("=" * 80)

        return trainable, non_trainable, total

    def unfreeze_backbone(self, num_layers=-1):
        """
        Unfreeze ResNet50V2 backbone for fine-tuning

        Args:
            num_layers: Number of layers to unfreeze (-1 for all)
        """
        if self.model is None:
            raise ValueError("Model must be built first")

        # Find ResNet50V2 backbone
        backbone = None
        for layer in self.model.layers:
            if 'resnet50v2' in layer.name.lower():
                if hasattr(layer, 'layer'):
                    backbone = layer.layer  # Extract from TimeDistributed
                    break

        if backbone is None:
            print("⚠️  ResNet50V2 backbone not found")
            return

        if num_layers == -1:
            # Unfreeze all layers
            backbone.trainable = True
            print(f"\n✅ Unfroze all ResNet50V2 layers")
        else:
            # Unfreeze last N layers
            backbone.trainable = True
            for layer in backbone.layers[:-num_layers]:
                layer.trainable = False

            print(f"\n✅ Unfroze last {num_layers} ResNet50V2 layers")

        # Recompile model with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )

        print("✅ Model recompiled with learning rate: 1e-5 (fine-tuning)")

        # Show new parameter counts
        self.count_parameters()

    def save_architecture(self, filepath="/workspace/models/architecture_config.json"):
        """Save model architecture configuration"""
        config = {
            'frames_per_video': self.frames_per_video,
            'img_size': self.img_size,
            'sequence_model': self.sequence_model_type,
            'gru_units': self.gru_units,
            'dense_layers': self.dense_layers,
            'dropout_rates': self.dropout_rates,
            'num_classes': self.num_classes
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✅ Architecture config saved to: {filepath}")


def main():
    """Test model architecture"""

    print("=" * 80)
    print("NexaraVision Model Architecture Test")
    print("=" * 80)

    # Create model
    model_builder = ViolenceDetectionModel(
        frames_per_video=20,
        img_size=(224, 224),
        sequence_model='GRU',
        gru_units=128,
        dense_layers=[256, 128, 64],
        dropout_rates=[0.5, 0.5, 0.5]
    )

    # Build model
    model = model_builder.build_model(trainable_backbone=False)

    # Compile model
    model_builder.compile_model(learning_rate=0.0005)

    # Print summary
    model_builder.print_summary()

    # Count parameters
    model_builder.count_parameters()

    # Save architecture config
    model_builder.save_architecture()

    # Test forward pass
    print("\n" + "=" * 80)
    print("Testing Forward Pass")
    print("=" * 80)

    # Create dummy input
    batch_size = 2
    dummy_input = tf.random.normal((batch_size, 20, 224, 224, 3))

    print(f"\nInput shape: {dummy_input.shape}")
    output = model(dummy_input, training=False)
    print(f"Output shape: {output.shape}")
    print(f"Output (sample 1): {output[0].numpy()}")
    print(f"Predictions: {tf.argmax(output, axis=1).numpy()}")

    print("\n" + "=" * 80)
    print("✅ Model Architecture Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
