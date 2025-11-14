# NexaraVision Training Guide

Complete guide for training the violence detection model.

---

## 📋 Overview

**Model**: ResNet50V2 + Bidirectional GRU
**Dataset**: 10,738 videos (50.22 GB)
**Hardware**: 2x RTX 3090 Ti (48GB VRAM)
**Target Accuracy**: 90-95% (with 10K videos)

---

## 📁 Training Scripts

| Script | Purpose |
|--------|---------|
| `data_preprocessing.py` | Video loading, frame extraction, data splits |
| `model_architecture.py` | ResNet50V2 + Bi-LSTM model implementation |
| `train_model.py` | Main training pipeline with callbacks |
| `test_pipeline.py` | Validation script to test all components |

---

## 🚀 Step-by-Step Training

### Step 1: Upload Scripts to Vast.ai Instance

Upload these 4 files to `/workspace/`:
- `data_preprocessing.py`
- `model_architecture.py`
- `train_model.py`
- `test_pipeline.py`

### Step 2: Test Pipeline (IMPORTANT!)

Before starting full training, validate everything works:

```bash
cd /workspace
python3 test_pipeline.py
```

**Expected output**:
```
================================================================================
NexaraVision Pipeline Validation
================================================================================

1️⃣ Testing imports...
   ✅ All imports successful

================================================================================
TEST 1: Data Preprocessing
================================================================================

📂 Scanning RWF2000...
   Violence: 1,000
   Non-Violence: 1,000
   Total: 2,000

📂 Scanning UCF_Crime...
   Violence: 550
   Non-Violence: 550
   Total: 1,100

📂 Scanning SCVD...
   Violence: 1,816
   Non-Violence: 1,816
   Total: 3,632

📂 Scanning RealLife...
   Violence: 2,000
   Non-Violence: 2,000
   Total: 4,000

================================================================================
Total Videos: 10,732
Violence: 5,366
Non-Violence: 5,366
================================================================================

✅ Found 10,732 videos

2️⃣ Testing frame extraction...
   ✅ Extracted frames shape: (20, 224, 224, 3)
   ✅ Value range: [0.000, 1.000]

3️⃣ Testing data splits...
   ✅ Splits created successfully

================================================================================
TEST 2: Model Architecture
================================================================================

4️⃣ Building model...
   ✅ Model built successfully

5️⃣ Compiling model...
   ✅ Model compiled successfully

6️⃣ Counting parameters...
   ✅ Total parameters: 28,XXX,XXX

================================================================================
✅ ALL TESTS PASSED!
🎉 SYSTEM READY FOR TRAINING!
================================================================================
```

**If any test fails, DO NOT proceed with training!** Share the error output.

### Step 3: Start Training

Once all tests pass:

```bash
python3 train_model.py
```

---

## ⏱️ Expected Training Time

| Phase | Epochs | Duration | Purpose |
|-------|--------|----------|---------|
| **Initial Training** | 30 | 3-5 hours | Transfer learning (frozen backbone) |
| **Fine-Tuning** | 20 | 2-3 hours | Unfreeze backbone, fine-tune |
| **Total** | 50 | **5-8 hours** | Complete training |

---

## 📊 Monitoring Training

### Option 1: Watch Terminal Output

```
Epoch 1/30
672/672 [==============================] - 287s 425ms/step
  loss: 0.3456
  accuracy: 0.8512
  precision: 0.8623
  recall: 0.8401
  val_loss: 0.2987
  val_accuracy: 0.8834
  val_precision: 0.8912
  val_recall: 0.8756
```

### Option 2: TensorBoard (Real-time Graphs)

In a separate terminal:

```bash
tensorboard --logdir /workspace/logs/training --host 0.0.0.0 --port 6006
```

Then access via browser: `http://YOUR_INSTANCE_IP:6006`

### Option 3: CSV Logs

Training metrics are saved to CSV files:

```bash
# View training progress
cat /workspace/logs/training/initial_training_log.csv
cat /workspace/logs/training/finetuning_training_log.csv

# Follow in real-time
tail -f /workspace/logs/training/initial_training_log.csv
```

---

## 💾 Saved Outputs

After training completes, you'll have:

### Model Checkpoints
```
/workspace/models/checkpoints/
  ├── initial_best_model.keras      # Best model from initial training
  └── finetuning_best_model.keras   # Best model from fine-tuning
```

### Final Model
```
/workspace/models/saved_models/
  └── final_model.keras              # Final trained model
```

### Training Logs
```
/workspace/logs/training/
  ├── initial_training_log.csv       # Epoch-by-epoch metrics
  ├── finetuning_training_log.csv
  └── initial_YYYYMMDD_HHMMSS/       # TensorBoard logs
```

### Evaluation Results
```
/workspace/logs/evaluation/
  └── test_results.json              # Final test set performance
```

### Data Splits
```
/workspace/processed/
  └── splits.json                    # Train/val/test split information
```

---

## 🎯 Expected Results

Based on 10,738 videos (above 10K threshold):

| Metric | Target Range | Expectation |
|--------|--------------|-------------|
| **Accuracy** | 90-95% | ✅ Achievable |
| **Precision** | 88-93% | ✅ Achievable |
| **Recall** | 88-93% | ✅ Achievable |
| **F1-Score** | 88-93% | ✅ Achievable |
| **False Positives** | <5% | ✅ Achievable |

With 10K+ videos, you should hit **90-93% accuracy range**.

---

## 🛑 Stopping/Resuming Training

### Graceful Stop

Press `Ctrl+C` once - the model will save the last checkpoint.

### Resume Training

The model automatically uses best checkpoint from previous run. To resume:

```bash
# Modify train_model.py to load checkpoint
# Or start fresh training (overwrites previous)
python3 train_model.py
```

### Check Current Progress

```bash
# See latest checkpoints
ls -lh /workspace/models/checkpoints/

# See latest logs
tail -100 /workspace/logs/training/*.csv
```

---

## ⚠️ Common Issues & Solutions

### Issue: Out of Memory (OOM)

```python
# Reduce batch size in /workspace/training_config.json
{
  "training": {
    "batch_size": 16  # Change from 32 to 16 or 8
  }
}
```

### Issue: Training Too Slow

Check GPU utilization:
```bash
watch -n 1 nvidia-smi
```

Should show:
- GPU Memory: 15-20GB used (per GPU)
- GPU Utilization: 90-100%

If low utilization, increase batch size.

### Issue: Accuracy Not Improving

- **After 15 epochs**: Normal, model is still learning
- **After 30 epochs**: Check validation loss - if decreasing, continue
- **Stuck at 60-70%**: Check data labels are correct

### Issue: Validation Loss Increasing

Early stopping will trigger automatically after 15 epochs of no improvement.

---

## 📈 Next Steps After Training

### 1. Evaluate Final Model

```bash
# Results are automatically saved to:
cat /workspace/logs/evaluation/test_results.json
```

### 2. Test on New Videos

Create `predict.py` to test on new videos:

```python
import tensorflow as tf
from data_preprocessing import VideoDataPreprocessor

# Load model
model = tf.keras.models.load_model('/workspace/models/saved_models/final_model.keras')

# Load video
preprocessor = VideoDataPreprocessor()
frames = preprocessor.extract_frames('test_video.mp4')

# Predict
prediction = model.predict(frames[np.newaxis, ...])
print(f"Violence probability: {prediction[0][1]:.2%}")
```

### 3. Optimize for Production

- Convert to TensorFlow Lite for mobile
- Optimize inference speed with TensorRT
- Deploy as REST API with Flask/FastAPI

---

## 💡 Pro Tips

1. **Monitor during initial phase**: First 5 epochs show if data pipeline works
2. **Save often**: Checkpoints save automatically, but good models are rare
3. **Watch validation metrics**: If val_loss diverges from train_loss, you have overfitting
4. **GPU temperature**: Keep below 80°C, check with `nvidia-smi`
5. **Storage space**: Training generates ~5-10GB of logs/checkpoints

---

## 📞 Support

If training fails or results are poor:

1. Run `test_pipeline.py` again to isolate the issue
2. Check training logs for errors
3. Verify GPU is being used (not CPU)
4. Share error messages for debugging

**Remember**: With 10,738 videos, you should achieve **90-93% accuracy**.
If results are significantly lower, there may be a data or configuration issue.

---

**Ready to train? Run:**

```bash
python3 test_pipeline.py  # Validate first (5 minutes)
python3 train_model.py     # Start training (5-8 hours)
```

Good luck! 🚀
