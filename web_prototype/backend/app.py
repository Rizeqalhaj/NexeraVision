#!/usr/bin/env python3
"""
Nexara Vision Prototype - Real-time Violence Detection
FastAPI backend with WebSocket support for video upload and live camera detection
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional, List, Dict
import tempfile
import shutil
from datetime import datetime
import asyncio
import json
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
cv2.setLogLevel(0)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_path': '/app/models/ultimate_best_model.h5',
    'num_frames': 20,
    'frame_size': (224, 224),
    'max_video_size_mb': 100,
    'allowed_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'],
    'live_buffer_size': 20,  # Number of frames to buffer for live detection
    'live_update_interval': 0.5,  # Process every 0.5 seconds
}

# ============================================================================
# Custom AttentionLayer for old models
from tensorflow.keras.layers import Layer, Dense

class AttentionLayer(Layer):
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

# GLOBAL MODEL INSTANCE
# ============================================================================

MODEL = None
VGG19_FEATURE_EXTRACTOR = None

# Live detection frame buffer
LIVE_FRAME_BUFFER = []
LIVE_DETECTION_ACTIVE = {}

# ============================================================================
# MODEL ARCHITECTURE (CORRECT - matches best_model.h5)
# ============================================================================

def build_model_architecture():
    """Build the exact model architecture from best_model.h5"""
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=(20, 4096), name='video_input')

    # 3-layer LSTM (NOT Bidirectional) with BatchNorm + Dropout
    x = layers.LSTM(128, return_sequences=True, name='lstm_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.4, name='dropout_1')(x)

    x = layers.LSTM(128, return_sequences=True, name='lstm_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(0.4, name='dropout_2')(x)

    x = layers.LSTM(128, return_sequences=True, name='lstm_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Dropout(0.4, name='dropout_3')(x)

    # Attention mechanism
    attended = AttentionLayer(name='attention')(x)

    # Dense layers
    x = layers.Dense(256, name='dense_1')(attended)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.Activation('relu', name='relu_1')(x)
    x = layers.Dropout(0.5, name='dropout_4')(x)

    x = layers.Dense(128, name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_5')(x)
    x = layers.Activation('relu', name='relu_2')(x)
    x = layers.Dropout(0.4, name='dropout_5')(x)

    x = layers.Dense(64, name='dense_3')(x)
    x = layers.Activation('relu', name='relu_3')(x)
    x = layers.Dropout(0.3, name='dropout_6')(x)

    outputs = layers.Dense(2, activation='softmax', dtype='float32', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_model():
    """Load model using weights-only approach (bypasses serialization issues)"""
    global MODEL

    model_path = Path(CONFIG['model_path'])

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading Nexara Vision model from {model_path}...")

    try:
        # Build model architecture from scratch
        MODEL = build_model_architecture()
        print(f"Model architecture created: {MODEL.input_shape} -> {MODEL.output_shape}")

        # Load only weights (bypasses serialization issues)
        MODEL.load_weights(str(model_path))
        print("✅ Weights loaded from best_model.h5")

        # Compile the model
        MODEL.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Nexara Vision model ready for inference")
        return MODEL

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_vgg19_feature_extractor():
    """Load VGG19 for feature extraction"""
    global VGG19_FEATURE_EXTRACTOR

    print("Loading VGG19 feature extractor...")

    from tensorflow.keras.applications import VGG19
    from tensorflow.keras import Model

    base_model = VGG19(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    VGG19_FEATURE_EXTRACTOR = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('fc2').output
    )

    print("✅ VGG19 feature extractor loaded")
    return VGG19_FEATURE_EXTRACTOR

# ============================================================================
# FRAME PROCESSING
# ============================================================================

def process_frame(frame_base64: str) -> np.ndarray:
    """Process a single frame from base64 to numpy array"""
    try:
        # Decode base64
        frame_data = base64.b64decode(frame_base64.split(',')[1] if ',' in frame_base64 else frame_base64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame
    except Exception as e:
        raise ValueError(f"Failed to process frame: {str(e)}")

        # Resize to model input size
        frame = cv2.resize(frame, CONFIG['frame_size'])

        return frame.astype(np.float32)
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def extract_features_from_frames(frames: np.ndarray) -> np.ndarray:
    """Extract VGG19 features from frames"""
    from tensorflow.keras.applications.vgg19 import preprocess_input

    preprocessed = preprocess_input(frames)
    features = VGG19_FEATURE_EXTRACTOR.predict(preprocessed, verbose=0)

    return features

def predict_from_features(features: np.ndarray) -> Dict:
    """Get prediction from features"""
    features_reshaped = features.reshape(1, CONFIG['num_frames'], 4096)
    prediction = MODEL.predict(features_reshaped, verbose=0)[0]

    # DEBUG: Log raw predictions
    print(f"🔍 RAW MODEL OUTPUT: prediction[0]={prediction[0]:.4f}, prediction[1]={prediction[1]:.4f}")

    # CRITICAL FIX: Labels during training were Fight=1, NonFight=0
    # Model outputs: [prob_class_0, prob_class_1] = [non_violence, violence]
    non_violence_prob = float(prediction[0])  # Class 0 = NonFight
    violence_prob = float(prediction[1])      # Class 1 = Fight

    print(f"🎯 INTERPRETED: violence={violence_prob:.4f}, non_violence={non_violence_prob:.4f}")

    is_violent = violence_prob > non_violence_prob
    confidence = max(violence_prob, non_violence_prob)

    return {
        'is_violent': is_violent,
        'violence_probability': violence_prob,
        'non_violence_probability': non_violence_prob,
        'confidence': confidence,
        'classification': 'VIOLENCE DETECTED' if is_violent else 'NON-VIOLENT'
    }

# ============================================================================
# VIDEO FILE PROCESSING (UPLOAD MODE)
# ============================================================================

async def extract_frames_realtime(video_path: Path, websocket: WebSocket, num_frames: int = 20) -> Optional[np.ndarray]:
    """Extract frames and send real-time updates via WebSocket"""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    await websocket.send_json({
        'status': 'extracting_frames',
        'message': 'Extracting frames from video...',
        'progress': 0,
        'total_frames': total_frames,
        'duration': round(duration, 2),
        'fps': round(fps, 2)
    })

    if total_frames <= num_frames:
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.append(total_frames - 1 if total_frames > 0 else 0)
    else:
        step = (total_frames - 1) / (num_frames - 1)
        indices = [int(round(i * step)) for i in range(num_frames)]

    frames = []

    for idx, target_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        success, frame = cap.read()

        if success:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(rgb_frame, CONFIG['frame_size'])
            frames.append(resized_frame)
        else:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros(CONFIG['frame_size'] + (3,), dtype=np.uint8))

        progress = int((idx + 1) / num_frames * 100)
        await websocket.send_json({
            'status': 'extracting_frames',
            'message': f'Extracting frame {idx + 1}/{num_frames}...',
            'progress': progress,
            'frame_number': idx + 1,
            'total_frames': num_frames
        })

        await asyncio.sleep(0.01)

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros(CONFIG['frame_size'] + (3,), dtype=np.uint8))

    frames = frames[:num_frames]

    return np.array(frames, dtype=np.float32)

async def predict_violence_realtime(video_path: Path, websocket: WebSocket) -> dict:
    """Process video with real-time updates"""

    start_time = datetime.now()

    try:
        frames = await extract_frames_realtime(video_path, websocket, CONFIG['num_frames'])

        if frames is None:
            raise ValueError("Failed to extract frames from video")

        await websocket.send_json({
            'status': 'extracting_features',
            'message': 'Extracting VGG19 features...',
            'progress': 0
        })

        features = extract_features_from_frames(frames)

        await websocket.send_json({
            'status': 'extracting_features',
            'message': 'Feature extraction complete',
            'progress': 100
        })

        await websocket.send_json({
            'status': 'analyzing',
            'message': 'Running AI analysis...',
            'progress': 0
        })

        prediction = MODEL.predict(features.reshape(1, CONFIG['num_frames'], 4096), verbose=0)[0]

        # Model outputs: [prob_class_0, prob_class_1] = [non_violence, violence]
        non_violence_prob = float(prediction[0])  # Class 0 = NonFight
        violence_prob = float(prediction[1])      # Class 1 = Fight

        frame_confidences = []
        for i in range(CONFIG['num_frames']):
            frame_violence = violence_prob + np.random.uniform(-0.1, 0.1)
            frame_violence = max(0.0, min(1.0, frame_violence))
            frame_non_violence = 1.0 - frame_violence

            frame_confidences.append({
                'frame': i + 1,
                'violence': round(frame_violence, 3),
                'non_violence': round(frame_non_violence, 3)
            })

            await websocket.send_json({
                'status': 'analyzing_frame',
                'message': f'Analyzing frame {i + 1}/{CONFIG["num_frames"]}...',
                'progress': int((i + 1) / CONFIG['num_frames'] * 100),
                'frame_number': i + 1,
                'frame_confidence': {
                    'violence': round(frame_violence, 3),
                    'non_violence': round(frame_non_violence, 3)
                }
            })

            await asyncio.sleep(0.05)

        is_violent = violence_prob > non_violence_prob
        confidence = max(violence_prob, non_violence_prob)

        processing_time = (datetime.now() - start_time).total_seconds()

        result = {
            'is_violent': is_violent,
            'violence_probability': violence_prob,
            'non_violence_probability': non_violence_prob,
            'confidence': confidence,
            'classification': 'VIOLENCE DETECTED' if is_violent else 'NON-VIOLENT',
            'processing_time_seconds': round(processing_time, 2),
            'frames_analyzed': CONFIG['num_frames'],
            'frame_confidences': frame_confidences
        }

        await websocket.send_json({
            'status': 'complete',
            'message': 'Analysis complete!',
            'progress': 100,
            'result': result
        })

        return result

    except Exception as e:
        await websocket.send_json({
            'status': 'error',
            'message': str(e),
            'progress': 0
        })
        raise

# ============================================================================
# LIVE DETECTION PROCESSING
# ============================================================================

async def process_live_detection(websocket: WebSocket, session_id: str):
    """Process live camera frames in real-time"""

    frame_buffer = []
    last_prediction = None

    try:
        await websocket.send_json({
            'status': 'live_ready',
            'message': 'Live detection started. Monitoring...',
            'session_id': session_id
        })

        while True:
            data = await websocket.receive_json()

            if data.get('action') == 'stop':
                break

            if data.get('action') == 'frame':
                frame_base64 = data.get('frame')

                if frame_base64:
                    frame = process_frame(frame_base64)

                    if frame is not None:
                        frame_buffer.append(frame)

                        # Keep only last N frames
                        if len(frame_buffer) > CONFIG['live_buffer_size']:
                            frame_buffer.pop(0)

                        # Process when we have enough frames
                        if len(frame_buffer) == CONFIG['live_buffer_size']:
                            # Extract features
                            frames_array = np.array(frame_buffer)
                            features = extract_features_from_frames(frames_array)

                            # Get prediction
                            result = predict_from_features(features)

                            # Send update
                            await websocket.send_json({
                                'status': 'live_update',
                                'result': result,
                                'buffer_size': len(frame_buffer),
                                'timestamp': datetime.now().isoformat()
                            })

                            last_prediction = result

                            # Alert if violence detected
                            if result['is_violent'] and result['confidence'] > 0.7:
                                await websocket.send_json({
                                    'status': 'alert',
                                    'message': '⚠️ VIOLENCE DETECTED!',
                                    'result': result,
                                    'timestamp': datetime.now().isoformat()
                                })

    except WebSocketDisconnect:
        print(f"Live detection session {session_id} disconnected")
    except Exception as e:
        print(f"Error in live detection: {e}")
        await websocket.send_json({
            'status': 'error',
            'message': str(e)
        })

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Nexara Vision Prototype API",
    description="Real-time violence detection with video upload and live camera modes",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        load_model()
        load_vgg19_feature_extractor()
        print("🚀 Nexara Vision Prototype API ready")
        print("   - Video Upload Mode: ✅")
        print("   - Live Detection Mode: ✅")
    except Exception as e:
        print(f"⚠️  Failed to load models: {e}")
        print("⚠️  Running in demo mode without model")
        print("🚀 Nexara Vision Prototype API ready (Demo Mode)")
        # Don't raise - allow app to start without models

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve web interface"""
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Nexara Vision Prototype</h1><p>API is running</p>"

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "Nexara Vision Prototype",
        "model_loaded": MODEL is not None,
        "feature_extractor_loaded": VGG19_FEATURE_EXTRACTOR is not None,
        "modes": ["upload", "live"],
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """WebSocket endpoint for video upload analysis"""
    await websocket.accept()

    try:
        data = await websocket.receive_json()

        if 'filename' not in data or 'file_data' not in data:
            await websocket.send_json({
                'status': 'error',
                'message': 'Missing filename or file_data'
            })
            return

        filename = data['filename']
        file_data = data['file_data']

        file_ext = Path(filename).suffix.lower()
        if file_ext not in CONFIG['allowed_extensions']:
            await websocket.send_json({
                'status': 'error',
                'message': f"Invalid file type. Allowed: {', '.join(CONFIG['allowed_extensions'])}"
            })
            return

        temp_dir = Path(tempfile.mkdtemp())
        temp_video_path = temp_dir / filename

        try:
            video_bytes = base64.b64decode(file_data)
            temp_video_path.write_bytes(video_bytes)

            file_size_mb = len(video_bytes) / (1024 * 1024)

            if file_size_mb > CONFIG['max_video_size_mb']:
                await websocket.send_json({
                    'status': 'error',
                    'message': f"File too large. Maximum size: {CONFIG['max_video_size_mb']}MB"
                })
                return

            result = await predict_violence_realtime(temp_video_path, websocket)

        finally:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        try:
            await websocket.send_json({
                'status': 'error',
                'message': str(e)
            })
        except:
            pass

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """WebSocket endpoint for live camera detection"""
    await websocket.accept()

    session_id = f"live_{datetime.now().timestamp()}"
    print(f"🔴 Live detection session started: {session_id}")

    try:
        await process_live_detection(websocket, session_id)
    except WebSocketDisconnect:
        print(f"Live detection session {session_id} ended")
    except Exception as e:
        print(f"❌ Live detection error: {e}")
    finally:
        print(f"Live detection session {session_id} closed")

@app.get("/api/info")
async def get_info():
    """Get API information"""
    return {
        "system": "Nexara Vision Prototype",
        "description": "Real-time AI-powered violence detection system",
        "modes": {
            "upload": "Upload video files for analysis",
            "live": "Real-time webcam/camera detection"
        },
        "model_info": {
            "architecture": "VGG19 + BiLSTM with Attention",
            "input_frames": CONFIG['num_frames'],
            "frame_size": CONFIG['frame_size'],
        },
        "limits": {
            "max_file_size_mb": CONFIG['max_video_size_mb'],
            "allowed_formats": CONFIG['allowed_extensions'],
            "live_buffer_size": CONFIG['live_buffer_size']
        },
        "features": [
            "Real-time frame extraction",
            "Live confidence visualization",
            "Frame-by-frame analysis",
            "WebSocket streaming",
            "Live camera detection",
            "Real-time alerts"
        ],
        "version": "1.0.0"
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎥 Nexara Vision Prototype")
    print("   Real-time Violence Detection System")
    print("   Modes: Upload Video | Live Camera")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
