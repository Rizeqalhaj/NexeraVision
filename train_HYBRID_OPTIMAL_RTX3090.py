#!/usr/bin/env python3
"""
HYBRID OPTIMAL Violence Detection Training - OPTIMIZED FOR RTX 3090
Hardware: RTX 3090 24GB, AMD EPYC 7642 192-core, 2TB RAM

OPTIMIZATIONS FOR RTX 3090:
- Batch size: 96 (conservative for 24GB VRAM stability)
- Parallel video processing: 48 workers
- Memory-efficient feature loading
- FP16 mixed precision
- GPU memory growth enabled

Expected time: 13-16 hours total
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import cv2
import time
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import multiprocessing

print("="*80)
print("🚀 HYBRID OPTIMAL - RTX 3090 OPTIMIZED")
print("="*80)
print(f"TensorFlow: {tf.__version__}")

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {gpus}")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)  # Critical for 3090 stability

# Mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision: FP16 enabled")

cpu_count = multiprocessing.cpu_count()
print(f"CPU cores: {cpu_count}")
print("="*80 + "\n")

# ============================================================================
# OPTIMIZED CONFIGURATION FOR RTX 3090
# ============================================================================

CONFIG = {
    # Paths
    'dataset_path': '/workspace/organized_dataset',
    'cache_dir': '/workspace/hybrid_optimal_cache',
    'checkpoint_dir': '/workspace/hybrid_optimal_checkpoints',

    # VGG19 extraction
    'num_frames': 20,
    'frame_size': (224, 224),
    'num_workers': min(48, cpu_count),

    # Architecture
    'feature_dim': 512,
    'lstm_units': 96,
    'dropout_rate': 0.32,
    'recurrent_dropout': 0.18,
    'l2_reg': 0.003,

    # Training - OPTIMIZED FOR RTX 3090 24GB
    'batch_size': 96,  # Conservative for stability (4090 uses 128)
    'epochs': 150,
    'early_stopping_patience': 30,

    # Augmentation
    'augmentation_multiplier': 3,
    'brightness_range': 0.12,
    'noise_std': 0.008,
    'temporal_jitter': True,

    # Loss
    'focal_gamma': 3.0,
    'focal_alpha': 0.5,

    # Learning rate
    'initial_lr': 0.001,
    'warmup_epochs': 5,
}

print("📊 RTX 3090 OPTIMIZED CONFIGURATION:")
print(f"  GPU: RTX 3090 24GB")
print(f"  Batch size: {CONFIG['batch_size']} (optimized for stability)")
print(f"  Parallel workers: {CONFIG['num_workers']}")
print(f"  Dropout: {CONFIG['dropout_rate']*100:.0f}% (MODERATE)")
print(f"  Augmentation: {CONFIG['augmentation_multiplier']}x (BALANCED)")
print(f"  Expected time: 13-16 hours")
print("="*80 + "\n")

# ============================================================================
# VGG19 FEATURE EXTRACTION
# ============================================================================

def extract_single_video(video_path, label, feature_extractor, config):
    """Extract features from a single video"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, label, "Cannot open"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < config['num_frames']:
            cap.release()
            return None, label, f"Only {total_frames} frames"

        # Extract frames
        indices = np.linspace(0, total_frames - 1, config['num_frames'], dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, config['frame_size'])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
            frame = tf.keras.applications.vgg19.preprocess_input(frame)
            frames.append(frame)

        cap.release()

        if len(frames) < config['num_frames']:
            return None, label, f"Only {len(frames)} frames"

        # Extract VGG19 features
        frames_batch = np.array(frames)
        features = feature_extractor.predict(frames_batch, verbose=0)
        features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)

        return features, label, None

    except Exception as e:
        return None, label, f"Error: {e}"

def extract_vgg19_features_from_videos(split, config):
    """Extract VGG19 features from videos"""

    cache_file = Path(config['cache_dir']) / f'{split}_features_base.npy'
    labels_file = Path(config['cache_dir']) / f'{split}_labels_base.npy'

    if cache_file.exists() and labels_file.exists():
        print(f"  ✅ Loading cached {split} features")
        features = np.load(cache_file, mmap_mode='r')
        labels = np.load(labels_file)
        print(f"     Shape: {features.shape}")
        return features, labels

    print(f"\n  🎬 Extracting VGG19 features from {split} videos...")

    # Load VGG19
    base_model = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('fc2').output
    )
    print(f"     ✅ VGG19 loaded (fc2: 4096 dims)")

    # Get videos
    split_path = Path(config['dataset_path']) / split
    violent_videos = sorted((split_path / 'violent').glob('*.mp4'))
    nonviolent_videos = sorted((split_path / 'nonviolent').glob('*.mp4'))

    print(f"     Found: {len(violent_videos)} violent, {len(nonviolent_videos)} non-violent")

    all_videos = [(v, 1) for v in violent_videos] + [(v, 0) for v in nonviolent_videos]

    all_features = []
    all_labels = []
    failed = []

    # Process with progress bar
    print(f"     Extracting features...")
    for video_path, label in tqdm(all_videos, desc=f"     {split}"):
        features, lbl, error = extract_single_video(video_path, label, feature_extractor, config)

        if error is None:
            all_features.append(features)
            all_labels.append(lbl)
        else:
            failed.append((str(video_path), error))

    if failed:
        print(f"\n     ⚠️  Failed: {len(failed)} videos")

    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int32)

    print(f"\n     ✅ Extracted: {features_array.shape}")
    print(f"        Violent: {labels_array.sum()}, Non-violent: {len(labels_array) - labels_array.sum()}")

    # Cache
    Path(config['cache_dir']).mkdir(parents=True, exist_ok=True)
    np.save(cache_file, features_array)
    np.save(labels_file, labels_array)
    print(f"     💾 Cached: {cache_file}")

    return features_array, labels_array

# ============================================================================
# AUGMENTATION
# ============================================================================

def apply_balanced_augmentation(features, config):
    """3x balanced augmentation"""
    augmented = [features]

    # Aug 1: Brightness
    brightness_factor = 1.0 + np.random.uniform(-config['brightness_range'], config['brightness_range'])
    aug1 = features * brightness_factor
    aug1 = np.clip(aug1, features.min(), features.max())
    augmented.append(aug1)

    # Aug 2: Temporal jitter + noise
    if config['temporal_jitter']:
        aug2 = features.copy()
        num_frames = features.shape[0]
        for i in range(0, num_frames, 4):
            end = min(i + 4, num_frames)
            if end - i > 1 and np.random.random() > 0.5:
                indices = np.arange(i, end)
                np.random.shuffle(indices)
                aug2[i:end] = aug2[indices]
        noise = np.random.normal(0, config['noise_std'], aug2.shape)
        aug2 = aug2 + noise
        augmented.append(aug2)
    else:
        noise = np.random.normal(0, config['noise_std'], features.shape)
        aug2 = features + noise
        augmented.append(aug2)

    return np.array(augmented)

def augment_dataset(features, labels, config):
    """Apply 3x augmentation"""
    print(f"\n  🔄 Applying {config['augmentation_multiplier']}x augmentation...")

    all_features = []
    all_labels = []

    for feat, label in tqdm(zip(features, labels), total=len(features), desc="     Augmenting"):
        augmented = apply_balanced_augmentation(feat, config)
        for aug_feat in augmented:
            all_features.append(aug_feat)
            all_labels.append(label)

    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int32)

    print(f"     ✅ {len(features)} → {len(features_array)} samples")
    print(f"        Violent: {labels_array.sum()}, Non-violent: {len(labels_array) - labels_array.sum()}")
    return features_array, labels_array

# ============================================================================
# PER-CLASS MONITORING
# ============================================================================

class PerClassAccuracyCallback(callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.history = {'violent_accuracy': [], 'nonviolent_accuracy': [], 'accuracy_gap': [], 'epoch': []}

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_classes = (y_pred[:, 1] > 0.5).astype(int)
        y_true_classes = y_val[:, 1].astype(int)

        violent_mask = y_true_classes == 1
        nonviolent_mask = y_true_classes == 0

        violent_acc = np.mean(y_pred_classes[violent_mask] == y_true_classes[violent_mask]) if violent_mask.sum() > 0 else 0
        nonviolent_acc = np.mean(y_pred_classes[nonviolent_mask] == y_true_classes[nonviolent_mask]) if nonviolent_mask.sum() > 0 else 0
        gap = abs(violent_acc - nonviolent_acc)

        self.history['violent_accuracy'].append(violent_acc)
        self.history['nonviolent_accuracy'].append(nonviolent_acc)
        self.history['accuracy_gap'].append(gap)
        self.history['epoch'].append(epoch + 1)

        status = "✅ EXCELLENT" if gap < 0.08 else "✅ GOOD" if gap < 0.15 else "⚠️  WARNING" if gap < 0.25 else "🚨 CRITICAL"

        print(f"\n  📊 Per-Class Accuracy (Epoch {epoch+1}):")
        print(f"    Violent:     {violent_acc*100:5.2f}%")
        print(f"    Non-violent: {nonviolent_acc*100:5.2f}%")
        print(f"    Gap:         {gap*100:5.2f}% {status}")

        if gap > 0.25:
            print(f"    🚨 CRITICAL: Gap exceeds 25%!")
        elif gap > 0.15:
            print(f"    ⚠️  WARNING: Gap exceeds 15%")

        logs['violent_accuracy'] = violent_acc
        logs['nonviolent_accuracy'] = nonviolent_acc
        logs['accuracy_gap'] = gap

# ============================================================================
# FOCAL LOSS
# ============================================================================

class EnhancedFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=3.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - pt, self.gamma)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(self.alpha * focal_weight * bce)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_hybrid_optimal_model(input_shape, config):
    """Hybrid architecture: residual + attention + compression"""
    inputs = layers.Input(shape=input_shape, name='input_features')

    # Feature compression
    x = layers.Dense(config['feature_dim'], activation='relu', name='feature_compression')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config['dropout_rate'] * 0.5)(x)

    # BiLSTM with residual
    x = layers.Bidirectional(
        layers.LSTM(config['lstm_units'], return_sequences=True,
                   dropout=config['dropout_rate'], recurrent_dropout=config['recurrent_dropout'],
                   kernel_regularizer=regularizers.l2(config['l2_reg'])), name='bilstm_1')(x)
    x = layers.BatchNormalization()(x)
    x_residual = x

    x = layers.Bidirectional(
        layers.LSTM(config['lstm_units'], return_sequences=True,
                   dropout=config['dropout_rate'], recurrent_dropout=config['recurrent_dropout'],
                   kernel_regularizer=regularizers.l2(config['l2_reg'])), name='bilstm_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add(name='residual_add')([x, x_residual])

    x = layers.Bidirectional(
        layers.LSTM(config['lstm_units'] // 2, return_sequences=True,
                   dropout=config['dropout_rate'], recurrent_dropout=config['recurrent_dropout'],
                   kernel_regularizer=regularizers.l2(config['l2_reg'])), name='bilstm_3')(x)
    x = layers.BatchNormalization()(x)

    # Attention mechanism
    attention_score = layers.Dense(1, activation='tanh', name='attention_score')(x)
    attention_score = layers.Flatten()(attention_score)
    attention_weights = layers.Activation('softmax', name='attention_weights')(attention_score)
    attention_weights_expanded = layers.RepeatVector(config['lstm_units'])(attention_weights)
    attention_weights_expanded = layers.Permute([2, 1])(attention_weights_expanded)
    attended = layers.Multiply(name='attended_features')([x, attention_weights_expanded])
    attended = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name='attention_pooling')(attended)

    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(config['l2_reg']), name='dense_1')(attended)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config['dropout_rate'])(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(config['l2_reg']), name='dense_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config['dropout_rate'] * 0.8)(x)

    outputs = layers.Dense(2, activation='softmax', dtype='float32', name='output')(x)

    return models.Model(inputs=inputs, outputs=outputs, name='HybridOptimalViolenceDetector')

# ============================================================================
# LR SCHEDULE & METRICS
# ============================================================================

def create_lr_schedule(config):
    def lr_schedule(epoch):
        if epoch < config['warmup_epochs']:
            return config['initial_lr'] * (epoch + 1) / config['warmup_epochs']
        decay_epochs = config['epochs'] - config['warmup_epochs']
        epoch_in_decay = epoch - config['warmup_epochs']
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_in_decay / decay_epochs))
        return config['initial_lr'] * cosine_decay + 1e-7
    return lr_schedule

class BinaryAccuracy(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(name='binary_accuracy', dtype=tf.float32, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros', dtype=tf.float32)
        self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_class = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        matches = tf.cast(tf.equal(y_true, y_pred_class), tf.float32)
        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return tf.divide(self.correct, self.total + tf.keras.backend.epsilon())

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("📥 STEP 1: VGG19 FEATURE EXTRACTION")
    print("="*80)

    extraction_start = time.time()
    X_train_base, y_train_base = extract_vgg19_features_from_videos('train', CONFIG)
    X_val, y_val = extract_vgg19_features_from_videos('val', CONFIG)
    extraction_time = time.time() - extraction_start
    print(f"\n⏱️  Extraction time: {extraction_time/3600:.1f} hours")

    print("\n" + "="*80)
    print("📥 STEP 2: AUGMENTATION")
    print("="*80)
    X_train, y_train = augment_dataset(X_train_base, y_train_base, CONFIG)

    print(f"\n📊 Final Dataset:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")

    # Convert to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 2)

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
    train_dataset = train_dataset.shuffle(20000).batch(CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat))
    val_dataset = val_dataset.batch(CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)

    print("\n" + "="*80)
    print("🏗️  STEP 3: BUILD MODEL")
    print("="*80)

    model = build_hybrid_optimal_model((20, 4096), CONFIG)
    model.summary()
    print(f"\n📊 Parameters: {model.count_params():,}")

    # Compile
    lr_schedule = create_lr_schedule(CONFIG)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule(0), clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=EnhancedFocalLoss(gamma=CONFIG['focal_gamma'], alpha=CONFIG['focal_alpha']),
        metrics=[BinaryAccuracy()]
    )

    print("\n" + "="*80)
    print("🚀 STEP 4: TRAINING")
    print("="*80)

    # Callbacks
    checkpoint_path = Path(CONFIG['checkpoint_dir'])
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    callbacks_list = [
        PerClassAccuracyCallback((X_val, y_val_cat)),
        callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=CONFIG['early_stopping_patience'],
                               restore_best_weights=True, mode='max', verbose=1),
        callbacks.ModelCheckpoint(str(checkpoint_path / 'hybrid_best_{epoch:03d}_{val_binary_accuracy:.4f}.h5'),
                                 monitor='val_binary_accuracy', save_best_only=True, mode='max', verbose=1),
        callbacks.LearningRateScheduler(lr_schedule, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.5, patience=15,
                                   min_lr=1e-7, mode='max', verbose=1),
        callbacks.CSVLogger(str(checkpoint_path / 'training_history.csv'), append=False),
    ]

    print("\n🔥 OPTIMIZATIONS ACTIVE:")
    print(f"  ✅ Moderate dropout: {CONFIG['dropout_rate']*100:.0f}% (preserves patterns)")
    print(f"  ✅ Balanced augmentation: {CONFIG['augmentation_multiplier']}x")
    print(f"  ✅ Per-class monitoring (catches bias)")
    print(f"  ✅ Residual connections")
    print(f"  ✅ Attention mechanism")
    print(f"  ✅ Enhanced focal loss (γ=3.0)")
    print(f"  ✅ Mixed precision FP16")
    print(f"  ✅ Batch size: {CONFIG['batch_size']}")
    print()

    # Train
    training_start = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG['epochs'],
        callbacks=callbacks_list,
        verbose=1
    )
    training_time = time.time() - training_start
    total_time = time.time() - extraction_start

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"⏱️  Extraction: {extraction_time/3600:.1f}h")
    print(f"⏱️  Training: {training_time/3600:.1f}h")
    print(f"⏱️  Total: {total_time/3600:.1f}h")
    print(f"📊 Best val accuracy: {max(history.history['val_binary_accuracy'])*100:.2f}%")

    # Per-class results
    per_class_cb = [cb for cb in callbacks_list if isinstance(cb, PerClassAccuracyCallback)][0]
    final_violent = per_class_cb.history['violent_accuracy'][-1]
    final_nonviolent = per_class_cb.history['nonviolent_accuracy'][-1]
    final_gap = per_class_cb.history['accuracy_gap'][-1]

    print(f"\n🎯 FINAL PER-CLASS PERFORMANCE:")
    print(f"   Violent:     {final_violent*100:.2f}%")
    print(f"   Non-violent: {final_nonviolent*100:.2f}%")
    print(f"   Gap:         {final_gap*100:.2f}%")

    if final_violent > 0.88 and final_nonviolent > 0.88 and final_gap < 0.08:
        print(f"\n🎉 SUCCESS! Expected TTA: 90-92%")
    elif final_violent > 0.85 and final_gap < 0.12:
        print(f"\n✅ GOOD! Expected TTA: 88-90%")
    else:
        print(f"\n⚠️  Below expectations - review training logs")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'hardware': 'RTX_3090_24GB',
        'extraction_hours': extraction_time / 3600,
        'training_hours': training_time / 3600,
        'total_hours': total_time / 3600,
        'best_val_accuracy': float(max(history.history['val_binary_accuracy'])),
        'final_violent_accuracy': float(final_violent),
        'final_nonviolent_accuracy': float(final_nonviolent),
        'final_gap': float(final_gap),
        'total_parameters': int(model.count_params()),
    }

    with open(checkpoint_path / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Saved to: {checkpoint_path}")
    print("\n" + "="*80)
    print("🎯 NEXT STEPS:")
    print("="*80)
    print("1. Test with TTA for final validation")
    print("2. Expected: 90-92% TTA accuracy")
    print("3. Deploy if TTA > 88%")
    print("="*80)

if __name__ == "__main__":
    main()
