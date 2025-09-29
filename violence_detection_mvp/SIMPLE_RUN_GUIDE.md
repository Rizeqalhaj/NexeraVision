# 🚀 SIMPLE RUN GUIDE - Fixed!

## ✅ **SOLUTION: 3 Simple Steps**

### **Step 1: Navigate to Directory**
```bash
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp
```

### **Step 2: Activate Virtual Environment**
```bash
source venv/bin/activate
```

### **Step 3: Run Commands**
```bash
python3 run.py info
```

## 🎯 **All-in-One Command**
```bash
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp && source venv/bin/activate && python3 run.py info
```

## 🛠️ **Alternative: Use Start Script**
```bash
cd /home/admin/Desktop/NexaraVision/violence_detection_mvp
./start.sh
# Then run: python run.py info
```

## ✅ **What You Should See**
```
Violence Detection MVP - System Information
==================================================
System Information:
  Platform: Linux-6.14.0-32-generic-x86_64-with-glibc2.39
  Python Version: 3.13.7
  CPU Count: 12
  Memory Total: 10.6 GB

Dependency Status:
  ✓ tensorflow: 2.20.0
  ✓ opencv: 4.12.0
  ✓ sklearn: 1.7.2
  ...
```

## 🎯 **Next Commands to Try**
```bash
# Test model architecture (no data needed)
python3 test_model_architecture.py

# See all available commands
python3 run.py --help

# Test preprocessing
python3 validate_implementation.py
```

## ⚠️ **If You Still Get Errors**
Make sure you're in the right directory and venv is activated:
```bash
pwd  # Should show: /home/admin/Desktop/NexaraVision/violence_detection_mvp
which python  # Should show: .../venv/bin/python
```

The system is working - just need to ensure virtual environment activation!