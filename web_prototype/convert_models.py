#!/usr/bin/env python3
"""
Model Converter: Load old Keras 2.3 models and re-save in TensorFlow 2.15 format
Uses intermediate TensorFlow version for compatibility
"""
import os
import sys
import json
import tempfile
import shutil
import h5py
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from pathlib import Path

# Custom AttentionLayer (required for loading)
class AttentionLayer(Layer):
    """Custom attention layer for video sequence models"""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_dense = Dense(1, use_bias=False)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention_scores = self.attention_dense(inputs)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


def fix_h5_config(model_path: str) -> str:
    """Fix H5 model config for old Keras 2.3 models"""
    print("  → Fixing H5 config (batch_shape, dtype issues)...")

    temp_model_path = tempfile.mktemp(suffix='.h5')
    shutil.copy(model_path, temp_model_path)

    try:
        with h5py.File(temp_model_path, 'r+') as f:
            if 'model_config' not in f.attrs:
                print("  ⚠️  No model_config found, using original file")
                os.remove(temp_model_path)
                return model_path

            config_str = f.attrs['model_config']
            if isinstance(config_str, bytes):
                config_str = config_str.decode('utf-8')

            config = json.loads(config_str)

            # Fix batch_shape, dtype, and dtype_policy in all layers
            if 'config' in config and 'layers' in config['config']:
                for layer in config['config']['layers']:
                    if 'config' in layer:
                        # Fix batch_shape → batch_input_shape
                        if 'batch_shape' in layer['config']:
                            batch_shape = layer['config']['batch_shape']
                            if batch_shape and len(batch_shape) > 1:
                                layer['config']['batch_input_shape'] = batch_shape
                                del layer['config']['batch_shape']

                        # Fix dtype if it's a DTypePolicy object
                        if 'dtype' in layer['config']:
                            dtype_val = layer['config']['dtype']
                            if isinstance(dtype_val, dict) and 'class_name' in dtype_val:
                                if dtype_val.get('class_name') == 'DTypePolicy':
                                    policy_name = dtype_val.get('config', {}).get('name', 'float32')
                                    if 'float16' in policy_name:
                                        layer['config']['dtype'] = 'float32'
                                    else:
                                        layer['config']['dtype'] = 'float32'

                        # Remove dtype_policy if it exists
                        if 'dtype_policy' in layer['config']:
                            del layer['config']['dtype_policy']

                        # Remove synchronized parameter (old BatchNormalization)
                        if 'synchronized' in layer['config']:
                            del layer['config']['synchronized']

                    # Fix dtype in inbound_nodes
                    if 'inbound_nodes' in layer:
                        for node in layer['inbound_nodes']:
                            if isinstance(node, list):
                                for item in node:
                                    if isinstance(item, list):
                                        for subitem in item:
                                            if isinstance(subitem, dict) and 'config' in subitem:
                                                if 'dtype' in subitem['config']:
                                                    dtype_val = subitem['config']['dtype']
                                                    if isinstance(dtype_val, str) and 'float16' in dtype_val:
                                                        subitem['config']['dtype'] = 'float32'

            # Remove top-level dtype_policy if it exists
            if 'dtype_policy' in config:
                del config['dtype_policy']

            # Write back fixed config
            config_str_fixed = json.dumps(config)
            f.attrs.modify('model_config', config_str_fixed.encode('utf-8'))

        print("  ✅ H5 config fixed successfully")
        return temp_model_path

    except Exception as e:
        print(f"  ❌ Failed to fix H5 config: {e}")
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        return model_path


def convert_model(input_path: str, output_path: str):
    """Convert old Keras model to modern format"""
    print(f"\n{'='*80}")
    print(f"Converting: {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*80}")

    fixed_model_path = None

    try:
        # Fix H5 config first
        fixed_model_path = fix_h5_config(input_path)

        # Load old model with custom objects
        print("Loading model...")
        custom_objects = {'AttentionLayer': AttentionLayer}

        # Try loading with safe_mode=False if available
        try:
            model = tf.keras.models.load_model(
                fixed_model_path,
                custom_objects=custom_objects,
                safe_mode=False,
                compile=False
            )
        except TypeError:
            # Older TF version without safe_mode
            model = tf.keras.models.load_model(
                fixed_model_path,
                custom_objects=custom_objects,
                compile=False
            )

        print(f"✅ Loaded successfully!")
        print(f"   - Layers: {len(model.layers)}")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")

        # Compile with original settings
        print("\nCompiling model...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00008),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Save in modern format
        print(f"\nSaving to: {output_path}")
        model.save(output_path, save_format='h5')

        print(f"✅ Conversion successful!")
        return True

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up temporary file
        if fixed_model_path and fixed_model_path != input_path and os.path.exists(fixed_model_path):
            os.remove(fixed_model_path)


def main():
    """Convert all models"""
    print(f"\n{'='*80}")
    print("MODEL CONVERSION SCRIPT")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"{'='*80}")

    # Model directory
    models_dir = Path("/app/models")
    output_dir = Path("/app/models_converted")
    output_dir.mkdir(exist_ok=True)

    # Models to convert
    models = [
        'best_model.h5',
        'ultimate_best_model.h5',
        'ensemble_m1_best.h5',
        'ensemble_m2_best.h5',
        'ensemble_m3_best.h5'
    ]

    success_count = 0
    failed_models = []

    for model_name in models:
        input_path = models_dir / model_name
        output_path = output_dir / model_name

        if not input_path.exists():
            print(f"\n⚠️  Model not found: {model_name}")
            continue

        if convert_model(str(input_path), str(output_path)):
            success_count += 1
        else:
            failed_models.append(model_name)

    # Summary
    print(f"\n{'='*80}")
    print("CONVERSION SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successfully converted: {success_count}/{len(models)}")
    if failed_models:
        print(f"❌ Failed models: {', '.join(failed_models)}")
    print(f"\nConverted models saved to: {output_dir}")
    print(f"{'='*80}\n")

    return success_count == len(models)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
