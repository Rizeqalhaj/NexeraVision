# 🚀 HOW TO RUN - Violence Detection MVP

## ✅ **FINAL SOLUTION**

The issue was: Need to use `python3` instead of `python`.

## 📍 **EXACT COMMANDS TO RUN**

### **Copy and paste these commands:**

```bash
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp
source venv/bin/activate
python3 run.py info
```

## 🎯 **WORKING COMMANDS**

### **1. System Status**
```bash
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp
source venv/bin/activate
python3 run.py info
```

### **2. Model Test (No data needed)**
```bash
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp
source venv/bin/activate
python3 test_model_architecture.py
```

### **3. See All Commands**
```bash
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp
source venv/bin/activate
python3 run.py --help
```

### **4. Complete Validation**
```bash
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp
source venv/bin/activate
python3 validate_implementation.py
```

## ✅ **Expected Output**
When you run `python3 run.py info`, you should see:

```
Violence Detection MVP - System Information
==================================================
System Information:
  Platform: Linux-6.14.0-32-generic-x86_64-with-glibc2.39
  Python Version: 3.13.7
  CPU Count: 12
  Memory Total: 10.6 GB
  Memory Available: 6.2 GB

Dependency Status:
  ✓ tensorflow: 2.20.0
  ✓ opencv: 4.12.0
  ✓ sklearn: 1.7.2
  ✓ matplotlib: 3.10.6
  ✓ seaborn: 0.13.2
  ✓ h5py: 3.14.0
  ✓ numpy: 2.2.6

Project Structure:
  ✓ src/
  ✓ data/raw/
  ✓ data/processed/
  ✓ models/
  ✓ notebooks/

Required Files:
  ✓ src/config.py
  ✓ src/data_preprocessing.py
  ✓ src/feature_extraction.py
  ✓ src/model_architecture.py
  ✓ src/training.py
  ✓ src/evaluation.py
  ✓ src/inference.py
  ✓ src/utils.py
  ✓ src/visualization.py

Model Status: No trained model found

Configuration:
  Image Size: 224x224
  Frames per Video: 20
  RNN Size: 128
  Batch Size: 64
  Learning Rate: 0.0001
```

## 🎯 **All Available Commands**

After activating the environment, you can run:

```bash
# System information
python3 run.py info

# Complete help
python3 run.py --help

# Model architecture test
python3 test_model_architecture.py

# Implementation validation
python3 validate_implementation.py

# Training (needs video data)
python3 run.py train --data-dir data/raw

# Demo (downloads VGG19 weights)
python3 run.py demo

# Real-time detection (needs trained model)
python3 run.py realtime --model-path models/model.h5 --camera 0
```

## 🔧 **Troubleshooting**

### **If you get "python3: command not found"**
Try:
```bash
/home/linuxbrew/.linuxbrew/opt/python@3.13/bin/python3 run.py info
```

### **If virtual environment fails**
```bash
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 run.py info
```

## ✅ **SUCCESS CONFIRMATION**

If `python3 run.py info` shows all the ✓ marks above, then **everything is working perfectly**!

The Violence Detection MVP is ready to use.