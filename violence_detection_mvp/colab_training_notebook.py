"""
Google Colab Training Notebook Template
Copy this entire file content to a new Colab notebook to train with FREE GPU.

Instructions:
1. Go to https://colab.research.google.com
2. Create new notebook
3. Copy cells below
4. Runtime > Change runtime type > GPU (T4)
5. Run all cells
"""

# ============================================================================
# CELL 1: Check GPU and Setup
# ============================================================================
"""
import tensorflow as tf
import subprocess

# Check GPU
gpu_info = !nvidia-smi
print('\\n'.join(gpu_info))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Available: {gpus}")
    gpu_details = !nvidia-smi --query-gpu=name,memory.total --format=csv
    print('\\n'.join(gpu_details))
else:
    print("❌ No GPU found! Please enable GPU:")
    print("   Runtime > Change runtime type > Hardware accelerator > GPU")

# Enable mixed precision for 2x memory efficiency
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')
print("✅ Mixed precision (FP16) enabled")
"""

# ============================================================================
# CELL 2: Clone Project from GitHub (if public) or Upload Files
# ============================================================================
"""
# Option A: Clone from GitHub (if you push your project)
# !git clone https://github.com/yourusername/violence-detection.git
# %cd violence-detection

# Option B: Upload project as zip
from google.colab import files
import zipfile
import os

print("📤 Please upload your violence_detection_mvp.zip file")
uploaded = files.upload()

# Extract
for filename in uploaded.keys():
    print(f'Extracting {filename}...')
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('.')

%cd violence_detection_mvp
!ls -la
"""

# ============================================================================
# CELL 3: Install Dependencies
# ============================================================================
"""
!pip install -q tensorflow opencv-python-headless scikit-learn huggingface_hub
print("✅ Dependencies installed")
"""

# ============================================================================
# CELL 4: Download Dataset (RWF-2000)
# ============================================================================
"""
from huggingface_hub import snapshot_download
from pathlib import Path

# Create data directory
Path("data/raw/rwf2000").mkdir(parents=True, exist_ok=True)

print("⬇️  Downloading RWF-2000 dataset from Hugging Face...")
print("This will take 5-10 minutes depending on network speed...")

snapshot_download(
    repo_id="DanJoshua/RWF-2000",
    repo_type="dataset",
    local_dir="data/raw/rwf2000",
    local_dir_use_symlinks=False
)

print("✅ Dataset downloaded!")

# Check structure
!ls -R data/raw/rwf2000 | head -20
"""

# ============================================================================
# CELL 5: Prepare Dataset Structure
# ============================================================================
"""
import shutil
from pathlib import Path

# RWF-2000 structure: train/Fight, train/NonFight, val/Fight, val/NonFight
rwf_base = Path("data/raw/rwf2000")

# Check what structure we have
print("Dataset structure:")
!find data/raw/rwf2000 -type d | head -10

# You may need to adjust based on actual structure
# Typical: rwf2000/train/Fight/*.avi, rwf2000/val/Fight/*.avi
"""

# ============================================================================
# CELL 6: Train with Feature Caching (Memory Efficient)
# ============================================================================
"""
# GPU memory check
import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')[0]
print(f"GPU: {gpu}")

# Run optimized training
!python src/train_optimized.py \\
    --mode cached \\
    --train-dir data/raw/rwf2000/train \\
    --val-dir data/raw/rwf2000/val \\
    --epochs 50 \\
    --cache-dir data/processed/features

print("\\n✅ Training completed!")
"""

# ============================================================================
# CELL 7: Evaluate Model
# ============================================================================
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = tf.keras.models.load_model('models/violence_detector_cached.h5')

# Load validation features
X_val = np.load('data/processed/features/val/vgg19_features.npy')
y_val = np.load('data/processed/features/val/labels.npy')

# Predictions
y_pred = (model.predict(X_val) > 0.5).astype(int).flatten()

# Classification report
print("\\n📊 Classification Report:")
print(classification_report(y_val, y_pred, target_names=['NonViolence', 'Violence']))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Accuracy
accuracy = (y_pred == y_val).mean()
print(f"\\n✅ Validation Accuracy: {accuracy*100:.2f}%")
"""

# ============================================================================
# CELL 8: Download Trained Model
# ============================================================================
"""
from google.colab import files

# Download trained model
files.download('models/violence_detector_cached.h5')
print("✅ Model downloaded to your computer!")

# Optional: Download training history
import json
# Save history if available
# with open('training_history.json', 'w') as f:
#     json.dump(history.history, f)
# files.download('training_history.json')
"""

# ============================================================================
# CELL 9: Test on Sample Video
# ============================================================================
"""
import cv2
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications import VGG19

# Upload test video
from google.colab import files
print("📤 Upload a test video (5-10 seconds, .mp4 or .avi)")
uploaded = files.upload()

test_video = list(uploaded.keys())[0]

# Load models
vgg19 = VGG19(weights='imagenet', include_top=True)
feature_extractor = tf.keras.Model(
    inputs=vgg19.input,
    outputs=vgg19.get_layer('fc2').output
)
violence_model = tf.keras.models.load_model('models/violence_detector_cached.h5')

# Process video
cap = cv2.VideoCapture(test_video)
frames = []
frame_count = 0

while frame_count < 150:  # SEQUENCE_LENGTH
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
    frame_count += 1

cap.release()

# Pad if needed
while len(frames) < 150:
    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

frames = frames[:150]
frames_array = np.array(frames)
frames_array = preprocess_input(frames_array)

# Extract features
features = feature_extractor.predict(frames_array, verbose=0)
features = np.expand_dims(features, axis=0)

# Predict
prediction = violence_model.predict(features)[0][0]

print(f"\\n{'='*50}")
print(f"🎬 Video: {test_video}")
print(f"🔮 Violence Probability: {prediction*100:.2f}%")
print(f"📝 Classification: {'⚠️ VIOLENCE DETECTED' if prediction > 0.5 else '✅ No Violence'}")
print(f"{'='*50}")
"""

# ============================================================================
# CELL 10: Training Time Estimation
# ============================================================================
"""
import time

gpu_name = !nvidia-smi --query-gpu=name --format=csv,noheader
print(f"GPU: {gpu_name[0]}")

# Estimated times based on GPU
times = {
    'Tesla T4': '2-3 hours (Colab Free)',
    'Tesla P100': '1.5-2 hours (Colab Pro)',
    'Tesla V100': '1-1.5 hours (Colab Pro)',
    'Tesla A100': '30-45 minutes (Colab Pro+)',
}

for gpu, time_est in times.items():
    if gpu in gpu_name[0]:
        print(f"\\n⏱️  Estimated Training Time: {time_est}")
        break
else:
    print(f"\\n⏱️  Estimated Training Time: 2-3 hours")

print(f"\\n💡 Tips:")
print("- Keep browser tab open during training")
print("- Colab free has ~12 hour session limit")
print("- Training will save checkpoints automatically")
"""

# ============================================================================
# ADDITIONAL: Save to Google Drive (Recommended)
# ============================================================================
"""
# Mount Google Drive to save model permanently
from google.colab import drive
drive.mount('/content/drive')

# Copy trained model to Drive
!mkdir -p "/content/drive/MyDrive/violence_detection"
!cp models/violence_detector_cached.h5 "/content/drive/MyDrive/violence_detection/"
!cp -r data/processed/features "/content/drive/MyDrive/violence_detection/"

print("✅ Model saved to Google Drive: MyDrive/violence_detection/")
"""

print("""
╔══════════════════════════════════════════════════════════════╗
║       GOOGLE COLAB TRAINING NOTEBOOK - READY TO USE         ║
╚══════════════════════════════════════════════════════════════╝

📋 INSTRUCTIONS:

1. Go to https://colab.research.google.com
2. File > New Notebook
3. Copy each cell from this file (marked with CELL 1, CELL 2, etc.)
4. Runtime > Change runtime type > GPU (select T4 or better)
5. Run cells in order

🎮 FREE GPU OPTIONS:
- Google Colab Free: Tesla T4 (15 GB) - FREE
- Google Colab Pro: V100/A100 - $10/month
- Kaggle Notebooks: Tesla P100 - FREE (30h/week)

⏱️  ESTIMATED TRAINING TIME:
- Tesla T4 (Colab Free): 2-3 hours for 50 epochs
- Tesla P100: 1.5-2 hours
- Tesla V100: 1-1.5 hours

💾 RESULTS:
- Trained model: violence_detector_cached.h5
- Accuracy: Expected 85-95% on RWF-2000
- Download model to your computer or save to Google Drive

🚀 START TRAINING NOW:
   https://colab.research.google.com

""")
