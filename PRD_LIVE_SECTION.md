# Product Requirements Document (PRD)
## NexaraVision /live Section - Multi-Camera Violence Detection System

**Document Version:** 1.0
**Date:** November 14, 2025
**Author:** NexaraVision Product Team
**Status:** Approved for Implementation

---

## 📋 EXECUTIVE SUMMARY

### Vision Statement
Build the world's first cost-effective AI violence detection system that works with existing CCTV infrastructure through innovative video segmentation preprocessing, enabling small/medium businesses to monitor 20-100+ cameras at 1/10th the cost of enterprise solutions.

### Business Objectives
- **Market Position:** First-to-market screen-based multi-camera violence detection
- **Target Accuracy:** 90-95% violence detection accuracy
- **Target Market:** Small/medium businesses with existing 20-100 camera CCTV systems
- **Pricing Strategy:** $5-15 per camera/month (vs $50-200 for enterprise solutions)
- **Go-to-Market:** MVP launch in 12 weeks with 5 pilot customers

### Success Metrics
| Metric | Target | Timeline |
|--------|--------|----------|
| Detection Accuracy | 90-95% | Week 12 |
| Latency (end-to-end) | <500ms | Week 10 |
| Grid Segmentation Success Rate | >85% | Week 8 |
| Beta Customer Acquisition | 5 customers | Week 12 |
| Customer Retention (3-month) | >80% | Month 6 |

---

## 🎯 PROBLEM STATEMENT

### User Pain Points

**Current State (Without NexaraVision):**
1. **High Cost:** Enterprise solutions (Verkada, Avigilon) charge $50-200 per camera/month
2. **Infrastructure Lock-in:** Requires replacing existing cameras with vendor-specific hardware
3. **Complex Installation:** Weeks of professional installation, network configuration
4. **Limited Accessibility:** Only large enterprises can afford comprehensive AI monitoring

**User Personas:**

**Persona 1: Small Retail Chain Security Manager**
- **Company Size:** 5-10 locations, 20-50 cameras total
- **Current Solution:** Human guards monitoring camera grids manually
- **Pain:** Cannot afford enterprise AI solutions ($40K-100K annual cost)
- **Job-to-be-Done:** Reduce monitoring burden, catch incidents faster

**Persona 2: Restaurant/Bar Owner**
- **Company Size:** 1-3 locations, 10-25 cameras
- **Current Solution:** DVR recording only, review after incidents
- **Pain:** Liability exposure, insurance claims require video evidence
- **Job-to-be-Done:** Real-time alerts, evidence collection, insurance compliance

**Persona 3: Parking Lot / Property Management**
- **Company Size:** Multiple properties, 50-100 cameras
- **Current Solution:** Occasional manual review of footage
- **Pain:** Incidents discovered hours/days later
- **Job-to-be-Done:** Immediate response capability, deterrent effect

### Market Opportunity

**Total Addressable Market (TAM):**
- 62M small businesses in US
- 15% have CCTV systems (~9.3M businesses)
- Average 15 cameras per business
- **TAM:** ~140M cameras

**Serviceable Addressable Market (SAM):**
- Businesses with 10-100 cameras
- Security-conscious verticals (retail, hospitality, education)
- **SAM:** ~30M cameras

**Serviceable Obtainable Market (SOM - Year 1):**
- Target: 0.1% market penetration
- **SOM:** 30,000 cameras (2,000 customers × 15 cameras avg)
- **Revenue:** $2.7M ARR at $9/camera/month

---

## 💡 SOLUTION OVERVIEW

### Core Innovation: Video Segmentation Preprocessing

**Architectural Breakthrough:**
Instead of processing entire screen (100 tiny cameras), we:
1. **Record** multi-camera grid display (4K/1080p screen capture)
2. **Segment** individual camera feeds using computer vision
3. **Process** each camera independently with trained model
4. **Achieve** 90-95% accuracy (vs 70-85% for whole-screen processing)

**Value Proposition:**
> "Enterprise-grade AI violence detection for your existing CCTV system - no camera replacement, no complex installation. 90-95% accuracy at 1/10th the cost."

### Product Strategy

**Phase 1 MVP:** /live Section (12 weeks)
- File upload + detection
- Live single-camera detection
- Multi-camera screen recording + segmentation

**Phase 2:** Dashboard & Alerts (8 weeks)
- Incident review dashboard
- Alert system (email, SMS, webhook)
- Analytics and reporting

**Phase 3:** Enterprise Features (12 weeks)
- Multi-user access control
- API for integrations
- Mobile apps

---

## 🏗️ TECHNICAL ARCHITECTURE

### Technology Stack

#### **Frontend: Next.js 14 + shadcn/ui**
**Rationale:**
- ✅ Server-side rendering for SEO, performance
- ✅ shadcn/ui provides beautiful, accessible components
- ✅ TypeScript for type safety
- ✅ Easy deployment (Vercel, self-hosted)
- ✅ Built-in API routes for simple backend tasks

**Components:**
- React 18 (with Suspense, Server Components)
- Tailwind CSS (styling framework)
- shadcn/ui (UI component library)
- Zustand or Jotai (state management - lightweight)
- React Query (data fetching, caching)
- Framer Motion (animations)

#### **Backend: NestJS + Python ML Service**
**Rationale:**
- ✅ NestJS for REST API, WebSocket, business logic
- ✅ TypeScript end-to-end consistency
- ✅ Dependency injection, modular architecture
- ✅ Separate Python service for ML (optimal for TensorFlow)

**NestJS Components:**
- RESTful API (video upload, config management)
- WebSocket Gateway (live camera streaming)
- Authentication/Authorization (JWT)
- Database ORM (Prisma or TypeORM)
- Queue Management (Bull/BullMQ for async processing)

**Python ML Service:**
- FastAPI (lightweight, async)
- TensorFlow 2.x (model inference)
- OpenCV (video processing, segmentation)
- NumPy (array operations)
- Redis (frame buffer, job queue)

**Why Separate Services:**
- Python optimized for ML (numpy, opencv, tensorflow)
- NestJS optimized for business logic, API
- Independent scaling (ML service needs GPU, API doesn't)
- Technology freedom (upgrade ML without touching API)

#### **Database: PostgreSQL + Redis**
- **PostgreSQL:** User accounts, camera configs, incident logs
- **Redis:** Frame buffering, real-time data, job queues

#### **Infrastructure:**
- **Development:** Docker Compose (all services local)
- **Production:** Kubernetes or Docker Swarm (orchestration)
- **GPU:** NVIDIA Docker runtime for ML service
- **Storage:** S3-compatible (incident video clips)
- **CDN:** CloudFlare (static assets, DDoS protection)

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER BROWSER                              │
│  Next.js Frontend (shadcn/ui components)                     │
│  • File Upload UI                                            │
│  • Live Camera Stream                                        │
│  • Multi-Camera Dashboard                                    │
└──────────────────┬──────────────────────────────────────────┘
                   │ HTTP/WebSocket
                   ▼
┌─────────────────────────────────────────────────────────────┐
│               NESTJS API GATEWAY                             │
│  ┌───────────────────────────────────────────────┐           │
│  │ REST API Controllers                          │           │
│  │ • /api/upload (video file upload)            │           │
│  │ • /api/cameras (camera configuration)        │           │
│  │ • /api/incidents (query detection history)   │           │
│  └───────────────────────────────────────────────┘           │
│  ┌───────────────────────────────────────────────┐           │
│  │ WebSocket Gateway                             │           │
│  │ • /ws/live (real-time video stream)          │           │
│  │ • /ws/alerts (real-time alert notifications) │           │
│  └───────────────────────────────────────────────┘           │
│  ┌───────────────────────────────────────────────┐           │
│  │ Business Logic Services                       │           │
│  │ • User Management                             │           │
│  │ • Camera Configuration                        │           │
│  │ • Alert Routing                               │           │
│  └───────────────────────────────────────────────┘           │
└──────────────────┬──────────────────────────────────────────┘
                   │ gRPC / HTTP
                   ▼
┌─────────────────────────────────────────────────────────────┐
│           PYTHON ML SERVICE (FastAPI)                        │
│  ┌───────────────────────────────────────────────┐           │
│  │ Video Processing Pipeline                     │           │
│  │                                               │           │
│  │  1. VIDEO INPUT                               │           │
│  │     ├─ Single Video (file upload)            │           │
│  │     ├─ Live Camera Stream (WebRTC/WebSocket) │           │
│  │     └─ Screen Recording (multi-camera grid)  │           │
│  │                                               │           │
│  │  2. SEGMENTATION (if multi-camera)            │           │
│  │     ├─ Grid Detection Algorithm               │           │
│  │     ├─ Camera Boundary Extraction             │           │
│  │     ├─ Individual Feed Cropping               │           │
│  │     └─ Super-Resolution Upscaling             │           │
│  │                                               │           │
│  │  3. FRAME EXTRACTION                          │           │
│  │     ├─ Extract 20 frames uniformly            │           │
│  │     ├─ Resize to 224×224                      │           │
│  │     └─ Normalize pixel values                 │           │
│  │                                               │           │
│  │  4. FEATURE EXTRACTION                        │           │
│  │     └─ ResNet50V2 (frozen backbone)           │           │
│  │        Output: (20, 2048) features            │           │
│  │                                               │           │
│  │  5. TEMPORAL MODELING                         │           │
│  │     └─ Bidirectional GRU (128 units)          │           │
│  │        Output: (256) temporal features        │           │
│  │                                               │           │
│  │  6. CLASSIFICATION                            │           │
│  │     └─ Dense Layers (256→128→64→2)            │           │
│  │        Output: [P(non-violence), P(violence)] │           │
│  └───────────────────────────────────────────────┘           │
│  ┌───────────────────────────────────────────────┐           │
│  │ Model Management                              │           │
│  │ • Load trained model (.keras file)           │           │
│  │ • GPU memory optimization                     │           │
│  │ • Batch processing (multiple cameras)        │           │
│  └───────────────────────────────────────────────┘           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA LAYER                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │ PostgreSQL     │  │ Redis          │  │ S3 Storage    │  │
│  │ • Users        │  │ • Frame buffer │  │ • Video clips │  │
│  │ • Cameras      │  │ • Job queue    │  │ • Incidents   │  │
│  │ • Incidents    │  │ • Real-time    │  │               │  │
│  └────────────────┘  └────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

#### **Scenario 1: File Upload Detection**
```
1. User uploads video via Next.js UI
2. Next.js sends to NestJS /api/upload endpoint
3. NestJS stores file temporarily, creates job in Redis queue
4. Python ML service picks up job from queue
5. ML service:
   a. Extracts 20 frames
   b. Runs ResNet50V2 + Bi-LSTM
   c. Returns violence probability
6. NestJS stores result in PostgreSQL
7. NestJS returns result to Next.js frontend
8. Frontend displays result with confidence visualization
```

**Latency Target:** <3 seconds for 30-second video

#### **Scenario 2: Live Single-Camera Detection**
```
1. User clicks "Start Live Detection" button
2. Next.js requests webcam access
3. Next.js captures frames at 30fps
4. Next.js buffers 20 frames (0.66 seconds)
5. Next.js sends 20-frame batch to Python ML service via WebSocket
6. ML service processes batch (<200ms)
7. ML service returns violence probability
8. Next.js displays real-time probability meter
9. If violence detected (>85% confidence):
   a. NestJS triggers alert
   b. Saves clip to S3
   c. Logs incident in PostgreSQL
```

**Latency Target:** <500ms per inference cycle

#### **Scenario 3: Multi-Camera Screen Recording**
```
1. User uploads screen recording (100-camera grid)
2. NestJS creates job, passes to Python ML service
3. ML service:
   a. Loads grid configuration (camera bounding boxes)
   b. For each frame:
      - Crop each camera feed
      - Upscale to 640×360
   c. Buffer 20 frames per camera
   d. Process all cameras in parallel (batch inference)
   e. Collect violence probabilities per camera
4. ML service returns results:
   {
     "camera_1": {"violence_prob": 0.12, "timestamp": "00:01:23"},
     "camera_2": {"violence_prob": 0.94, "timestamp": "00:02:45"}, ← ALERT
     ...
   }
5. NestJS logs incidents where violence_prob > 0.85
6. NestJS sends alerts for flagged cameras
7. Frontend displays per-camera timeline with highlights
```

**Latency Target:** <10 seconds per minute of recorded footage (6x real-time)

---

## 📱 FEATURE REQUIREMENTS

### Priority Framework

**P0 (Must Have - MVP):** Required for launch
**P1 (Should Have):** Needed for production readiness
**P2 (Nice to Have):** Future enhancements

---

### Feature 1: File Upload + Single Video Detection

**Priority:** P0 (Must Have)

**User Story:**
> As a security manager, I want to upload existing CCTV footage and get AI analysis of violence probability, so that I can review incidents efficiently without watching hours of video.

**Acceptance Criteria:**
1. ✅ User can drag-and-drop or click to upload video files
2. ✅ Supported formats: MP4, AVI, MOV, MKV (H.264/H.265 codec)
3. ✅ Max file size: 500MB (configurable)
4. ✅ Upload progress indicator (0-100%)
5. ✅ System extracts 20 frames uniformly from video
6. ✅ System processes frames through trained model
7. ✅ Results displayed within 5 seconds for 30-second video
8. ✅ Results show:
   - Overall violence probability (0-100%)
   - Confidence level (Low/Medium/High)
   - Thumbnail preview of video
   - Timestamp of peak violence probability
9. ✅ User can download detection report (PDF/JSON)

**UI/UX Design (shadcn/ui components):**

```tsx
// File Upload Component
<Card className="border-2 border-dashed border-blue-500 p-8">
  <div className="text-center">
    <Upload className="mx-auto h-12 w-12 text-gray-400" />
    <p className="mt-2 text-sm text-gray-600">
      Drag and drop video file, or click to browse
    </p>
    <p className="text-xs text-gray-500 mt-1">
      MP4, AVI, MOV up to 500MB
    </p>
  </div>
</Card>

// Results Display
<Card className="mt-4">
  <CardHeader>
    <CardTitle>Detection Results</CardTitle>
  </CardHeader>
  <CardContent>
    <div className="space-y-4">
      {/* Violence Probability Gauge */}
      <div className="flex items-center justify-center">
        <Progress value={violenceProbability} className="w-full" />
        <span className="ml-4 text-2xl font-bold">
          {violenceProbability}%
        </span>
      </div>

      {/* Confidence Badge */}
      <Badge variant={confidence === 'High' ? 'destructive' : 'secondary'}>
        {confidence} Confidence
      </Badge>

      {/* Timeline with Peak Violence Moment */}
      <Separator />
      <div className="flex items-center gap-2">
        <Clock className="h-4 w-4" />
        <span className="text-sm">Peak Violence at {timestamp}</span>
      </div>
    </div>
  </CardContent>
</Card>
```

**Technical Implementation:**

**Frontend (Next.js):**
```typescript
// app/live/upload/page.tsx
'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { Upload, AlertCircle } from 'lucide-react';

export default function FileUploadPage() {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setUploading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv'],
    },
    maxSize: 500 * 1024 * 1024, // 500MB
    multiple: false,
  });

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <h1 className="text-3xl font-bold mb-6">Upload Video for Detection</h1>

      <Card
        {...getRootProps()}
        className={`border-2 border-dashed p-12 cursor-pointer transition-colors ${
          isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
        }`}
      >
        <input {...getInputProps()} />
        <div className="text-center">
          <Upload className="mx-auto h-16 w-16 text-gray-400" />
          <p className="mt-4 text-lg font-medium">
            {isDragActive ? 'Drop video here' : 'Drag & drop video, or click to browse'}
          </p>
          <p className="mt-2 text-sm text-gray-500">
            MP4, AVI, MOV, MKV up to 500MB
          </p>
        </div>
      </Card>

      {uploading && (
        <Card className="mt-6">
          <CardContent className="pt-6">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Processing video...</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} />
            </div>
          </CardContent>
        </Card>
      )}

      {result && (
        <DetectionResult result={result} />
      )}
    </div>
  );
}
```

**Backend (NestJS):**
```typescript
// src/upload/upload.controller.ts
import { Controller, Post, UploadedFile, UseInterceptors } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { UploadService } from './upload.service';

@Controller('api/upload')
export class UploadController {
  constructor(private readonly uploadService: UploadService) {}

  @Post()
  @UseInterceptors(FileInterceptor('video'))
  async uploadVideo(@UploadedFile() file: Express.Multer.File) {
    // Validate file
    if (!file) {
      throw new BadRequestException('No video file provided');
    }

    // Send to ML service for processing
    const result = await this.uploadService.processVideo(file);

    return {
      violenceProbability: result.violence_probability,
      confidence: result.confidence,
      timestamp: result.peak_timestamp,
      frameAnalysis: result.frame_analysis,
    };
  }
}

// src/upload/upload.service.ts
import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class UploadService {
  constructor(
    private httpService: HttpService,
    private configService: ConfigService,
  ) {}

  async processVideo(file: Express.Multer.File) {
    const mlServiceUrl = this.configService.get('ML_SERVICE_URL');

    // Send file to Python ML service
    const formData = new FormData();
    formData.append('video', file.buffer, file.originalname);

    const response = await this.httpService.axiosRef.post(
      `${mlServiceUrl}/detect`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      },
    );

    return response.data;
  }
}
```

**ML Service (Python/FastAPI):**
```python
# ml_service/app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tensorflow as tf
import tempfile
from pathlib import Path

app = FastAPI()

# Load trained model
MODEL = tf.keras.models.load_model('/models/nexara_model.keras')

@app.post("/detect")
async def detect_violence(video: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        # Extract frames
        frames = extract_frames(tmp_path, num_frames=20)

        # Preprocess
        frames_processed = preprocess_frames(frames)

        # Inference
        prediction = MODEL.predict(np.expand_dims(frames_processed, axis=0))
        violence_prob = float(prediction[0][1])  # Probability of violence class

        # Analyze frame-by-frame for timeline
        frame_analysis = analyze_per_frame(frames)

        return {
            "violence_probability": violence_prob,
            "confidence": get_confidence_level(violence_prob),
            "peak_timestamp": get_peak_timestamp(frame_analysis),
            "frame_analysis": frame_analysis,
        }
    finally:
        # Clean up temp file
        Path(tmp_path).unlink()

def extract_frames(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return np.array(frames)

def preprocess_frames(frames):
    # Normalize to [0, 1]
    return frames.astype(np.float32) / 255.0

def get_confidence_level(probability):
    if probability > 0.9:
        return "High"
    elif probability > 0.7:
        return "Medium"
    else:
        return "Low"

def get_peak_timestamp(frame_analysis):
    # Find frame with highest violence probability
    peak_frame = max(frame_analysis, key=lambda x: x['violence_prob'])
    return peak_frame['timestamp']
```

**Performance Requirements:**
- Upload speed: Limited by network bandwidth
- Processing time: <5 seconds for 30-second video
- Concurrent uploads: Support 10 simultaneous uploads
- GPU utilization: <50% per video (allow parallel processing)

**Testing Strategy:**
1. **Unit Tests:**
   - File upload validation (size, format)
   - Frame extraction accuracy (20 frames evenly distributed)
   - Model inference correctness

2. **Integration Tests:**
   - End-to-end upload → processing → results
   - Error handling (corrupt video, unsupported format)
   - Concurrent upload handling

3. **Performance Tests:**
   - Load test with 50 concurrent uploads
   - Measure latency at different video lengths (10s, 30s, 60s)

---

### Feature 2: Live Single-Camera Detection

**Priority:** P0 (Must Have)

**User Story:**
> As a security guard, I want to turn my webcam into a live violence detector with one button click, so that I can monitor a location in real-time without specialized hardware.

**Acceptance Criteria:**
1. ✅ "Start Detection" button activates webcam
2. ✅ Real-time video preview shown to user
3. ✅ Violence probability meter updates every 0.5-1 second
4. ✅ Visual alert (red border, sound) when violence detected (>85%)
5. ✅ System buffers 20 frames before inference (0.66s at 30fps)
6. ✅ Latency <500ms from camera frame to displayed result
7. ✅ Recording starts automatically when violence detected
8. ✅ User can stop detection at any time
9. ✅ System saves incident clips to storage

**UI/UX Design:**

```tsx
// Live Detection Component
<Card className="w-full max-w-4xl mx-auto">
  <CardHeader>
    <CardTitle className="flex items-center justify-between">
      <span>Live Violence Detection</span>
      {isDetecting && (
        <Badge variant="destructive" className="animate-pulse">
          LIVE
        </Badge>
      )}
    </CardTitle>
  </CardHeader>
  <CardContent>
    {/* Video Preview */}
    <div className={`relative rounded-lg overflow-hidden ${
      violenceDetected ? 'ring-4 ring-red-500 animate-pulse' : ''
    }`}>
      <video ref={videoRef} autoPlay className="w-full" />

      {/* Real-time Probability Overlay */}
      <div className="absolute bottom-4 left-4 right-4">
        <Card className="bg-black/70 backdrop-blur">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-white text-sm font-medium">
                Violence Probability
              </span>
              <span className={`text-2xl font-bold ${
                violenceProb > 85 ? 'text-red-500' : 'text-green-500'
              }`}>
                {violenceProb}%
              </span>
            </div>
            <Progress
              value={violenceProb}
              className="h-2"
              indicatorClassName={violenceProb > 85 ? 'bg-red-500' : 'bg-green-500'}
            />
          </CardContent>
        </Card>
      </div>
    </div>

    {/* Controls */}
    <div className="flex gap-4 mt-6">
      {!isDetecting ? (
        <Button
          size="lg"
          onClick={startDetection}
          className="flex-1"
        >
          <Video className="mr-2 h-5 w-5" />
          Start Live Detection
        </Button>
      ) : (
        <Button
          size="lg"
          variant="destructive"
          onClick={stopDetection}
          className="flex-1"
        >
          <Square className="mr-2 h-5 w-5" />
          Stop Detection
        </Button>
      )}
    </div>

    {/* Alert History */}
    {alerts.length > 0 && (
      <div className="mt-6">
        <h3 className="font-semibold mb-3">Recent Alerts</h3>
        <div className="space-y-2">
          {alerts.map((alert, idx) => (
            <Alert key={idx} variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Violence Detected</AlertTitle>
              <AlertDescription>
                {alert.timestamp} - Confidence: {alert.confidence}%
              </AlertDescription>
            </Alert>
          ))}
        </div>
      </div>
    )}
  </CardContent>
</Card>
```

**Technical Implementation:**

**Frontend (Next.js with WebSocket):**
```typescript
// app/live/camera/page.tsx
'use client';

import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Video, Square, AlertCircle } from 'lucide-react';

export default function LiveCameraPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [isDetecting, setIsDetecting] = useState(false);
  const [violenceProb, setViolenceProb] = useState(0);
  const [alerts, setAlerts] = useState([]);
  const frameBuffer = useRef<ImageData[]>([]);

  const startDetection = async () => {
    try {
      // Request webcam access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // Connect to WebSocket
      wsRef.current = new WebSocket('ws://localhost:3001/ws/live');

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setIsDetecting(true);
        startFrameCapture();
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setViolenceProb(Math.round(data.violence_probability * 100));

        if (data.violence_probability > 0.85) {
          handleViolenceDetected(data);
        }
      };
    } catch (error) {
      console.error('Failed to start detection:', error);
      alert('Camera access denied or not available');
    }
  };

  const startFrameCapture = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Capture frames at 30fps, send batch of 20 frames every 0.66s
    const interval = setInterval(() => {
      if (!isDetecting) {
        clearInterval(interval);
        return;
      }

      // Capture frame
      ctx.drawImage(video, 0, 0, 224, 224);
      const imageData = ctx.getImageData(0, 0, 224, 224);
      frameBuffer.current.push(imageData);

      // Send batch when we have 20 frames
      if (frameBuffer.current.length === 20) {
        sendFrameBatch(frameBuffer.current);
        frameBuffer.current = frameBuffer.current.slice(10); // Keep 50% overlap
      }
    }, 1000 / 30); // 30fps
  };

  const sendFrameBatch = (frames: ImageData[]) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    // Convert frames to base64 for transmission
    const framesData = frames.map(frame => {
      const canvas = document.createElement('canvas');
      canvas.width = frame.width;
      canvas.height = frame.height;
      const ctx = canvas.getContext('2d');
      ctx?.putImageData(frame, 0, 0);
      return canvas.toDataURL('image/jpeg', 0.8);
    });

    wsRef.current.send(JSON.stringify({
      type: 'analyze_frames',
      frames: framesData,
    }));
  };

  const handleViolenceDetected = (data) => {
    // Play alert sound
    const audio = new Audio('/alert-sound.mp3');
    audio.play();

    // Add to alerts list
    setAlerts(prev => [
      {
        timestamp: new Date().toLocaleTimeString(),
        confidence: Math.round(data.violence_probability * 100),
      },
      ...prev.slice(0, 4), // Keep last 5 alerts
    ]);

    // Optionally save clip (implement clip recording)
  };

  const stopDetection = () => {
    setIsDetecting(false);

    // Stop camera stream
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }

    // Close WebSocket
    wsRef.current?.close();
  };

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      {/* UI components from design above */}
    </div>
  );
}
```

**Backend (NestJS WebSocket Gateway):**
```typescript
// src/live/live.gateway.ts
import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  OnGatewayConnection,
  OnGatewayDisconnect,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { LiveService } from './live.service';

@WebSocketGateway({ namespace: '/ws/live' })
export class LiveGateway implements OnGatewayConnection, OnGatewayDisconnect {
  @WebSocketServer()
  server: Server;

  constructor(private liveService: LiveService) {}

  handleConnection(client: Socket) {
    console.log(`Client connected: ${client.id}`);
  }

  handleDisconnect(client: Socket) {
    console.log(`Client disconnected: ${client.id}`);
  }

  @SubscribeMessage('analyze_frames')
  async handleFrameAnalysis(client: Socket, payload: any) {
    const { frames } = payload;

    // Send frames to ML service
    const result = await this.liveService.analyzeFrames(frames);

    // Send result back to client
    client.emit('detection_result', {
      violence_probability: result.violence_probability,
      confidence: result.confidence,
      timestamp: new Date().toISOString(),
    });

    // If violence detected, log incident
    if (result.violence_probability > 0.85) {
      await this.liveService.logIncident(client.id, result);
    }
  }
}

// src/live/live.service.ts
import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class LiveService {
  constructor(
    private httpService: HttpService,
    private configService: ConfigService,
  ) {}

  async analyzeFrames(frames: string[]) {
    const mlServiceUrl = this.configService.get('ML_SERVICE_URL');

    const response = await this.httpService.axiosRef.post(
      `${mlServiceUrl}/detect_live`,
      { frames },
    );

    return response.data;
  }

  async logIncident(clientId: string, result: any) {
    // Save to database
    // Trigger alerts (email, SMS, webhook)
    // Save video clip to S3
  }
}
```

**ML Service (Python):**
```python
# ml_service/app.py (add endpoint)
@app.post("/detect_live")
async def detect_live(payload: dict):
    frames_base64 = payload['frames']

    # Decode base64 images
    frames = []
    for frame_b64 in frames_base64:
        # Remove data URI prefix
        img_data = frame_b64.split(',')[1]
        img_bytes = base64.b64decode(img_data)

        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    frames_array = np.array(frames)

    # Preprocess and predict
    frames_processed = preprocess_frames(frames_array)
    prediction = MODEL.predict(np.expand_dims(frames_processed, axis=0))
    violence_prob = float(prediction[0][1])

    return {
        "violence_probability": violence_prob,
        "confidence": get_confidence_level(violence_prob),
    }
```

**Performance Requirements:**
- Frame capture: 30fps
- Frame buffer: 20 frames (0.66 seconds)
- Inference latency: <200ms
- Total latency: <500ms (capture + transmission + inference + display)
- WebSocket throughput: Support 20 concurrent connections

---

### Feature 3: Multi-Camera Screen Recording + Segmentation

**Priority:** P1 (Should Have - Core Innovation)

**User Story:**
> As a security manager with 100 cameras displayed on a monitor grid, I want to record my screen and have the system automatically detect violence in each individual camera feed, so that I can monitor all cameras without per-camera installation.

**Acceptance Criteria:**
1. ✅ User can upload screen recording of multi-camera grid
2. ✅ System detects grid layout automatically OR allows manual calibration
3. ✅ System segments individual camera feeds from grid
4. ✅ System processes each camera independently
5. ✅ Results show per-camera violence timeline
6. ✅ Grid segmentation success rate >85%
7. ✅ Processing speed: 6x real-time (1 minute video in 10 seconds)
8. ✅ User can configure grid layout (10×10, 5×5, 4×4, custom)
9. ✅ System provides confidence score for segmentation quality
10. ✅ Fallback to manual grid calibration if auto-detection fails

**UI/UX Design:**

```tsx
// Grid Calibration Tool
<Card>
  <CardHeader>
    <CardTitle>Configure Camera Grid</CardTitle>
    <CardDescription>
      Define the layout of cameras in your screen recording
    </CardDescription>
  </CardHeader>
  <CardContent>
    {/* Grid Preset Selection */}
    <div className="mb-6">
      <Label>Grid Preset</Label>
      <div className="grid grid-cols-4 gap-2 mt-2">
        {[
          { rows: 1, cols: 1, label: '1×1' },
          { rows: 2, cols: 2, label: '2×2' },
          { rows: 3, cols: 3, label: '3×3' },
          { rows: 4, cols: 4, label: '4×4' },
          { rows: 5, cols: 5, label: '5×5' },
          { rows: 10, cols: 10, label: '10×10' },
        ].map(preset => (
          <Button
            key={preset.label}
            variant={selectedPreset === preset.label ? 'default' : 'outline'}
            onClick={() => setSelectedPreset(preset)}
          >
            {preset.label}
          </Button>
        ))}
        <Button variant="outline">Custom</Button>
      </div>
    </div>

    {/* Visual Grid Overlay Editor */}
    <div className="relative border rounded-lg overflow-hidden">
      {/* Video Frame */}
      <img src={firstFrame} alt="First frame" className="w-full" />

      {/* Grid Overlay (draggable boundaries) */}
      <svg className="absolute inset-0 w-full h-full">
        {gridLines.map((line, idx) => (
          <line
            key={idx}
            x1={line.x1}
            y1={line.y1}
            x2={line.x2}
            y2={line.y2}
            stroke="blue"
            strokeWidth="2"
            strokeDasharray="5,5"
          />
        ))}
        {/* Individual camera boundaries */}
        {cameraBounds.map((bounds, idx) => (
          <rect
            key={idx}
            x={bounds.x}
            y={bounds.y}
            width={bounds.w}
            height={bounds.h}
            fill="none"
            stroke="green"
            strokeWidth="2"
            className="cursor-move"
          />
        ))}
      </svg>
    </div>

    {/* Auto-Detect Button */}
    <div className="mt-4 flex gap-2">
      <Button onClick={autoDetectGrid} disabled={detecting}>
        {detecting ? 'Detecting...' : 'Auto-Detect Grid'}
      </Button>
      <Button variant="outline" onClick={resetGrid}>
        Reset
      </Button>
    </div>

    {/* Confidence Score */}
    {segmentationConfidence && (
      <Alert className="mt-4">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Segmentation Confidence</AlertTitle>
        <AlertDescription>
          {segmentationConfidence >= 0.9 ? (
            <span className="text-green-600">
              High confidence ({(segmentationConfidence * 100).toFixed(1)}%) - Grid detected accurately
            </span>
          ) : segmentationConfidence >= 0.7 ? (
            <span className="text-yellow-600">
              Medium confidence ({(segmentationConfidence * 100).toFixed(1)}%) - Review grid boundaries
            </span>
          ) : (
            <span className="text-red-600">
              Low confidence ({(segmentationConfidence * 100).toFixed(1)}%) - Manual calibration recommended
            </span>
          )}
        </AlertDescription>
      </Alert>
    )}

    {/* Save Configuration */}
    <div className="mt-6">
      <Button size="lg" className="w-full" onClick={saveGridConfig}>
        Save Grid Configuration
      </Button>
    </div>
  </CardContent>
</Card>

// Multi-Camera Results Dashboard
<Card>
  <CardHeader>
    <CardTitle>Detection Results - {totalCameras} Cameras</CardTitle>
  </CardHeader>
  <CardContent>
    {/* Summary Stats */}
    <div className="grid grid-cols-3 gap-4 mb-6">
      <Card>
        <CardContent className="pt-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-red-500">
              {violenceDetected}
            </div>
            <div className="text-sm text-gray-500">Violence Detected</div>
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="pt-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-green-500">
              {safeCamera}
            </div>
            <div className="text-sm text-gray-500">Safe Cameras</div>
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="pt-6">
          <div className="text-center">
            <div className="text-3xl font-bold">
              {processingTime}s
            </div>
            <div className="text-sm text-gray-500">Processing Time</div>
          </div>
        </CardContent>
      </Card>
    </div>

    {/* Per-Camera Timeline */}
    <div className="space-y-4">
      {cameraResults
        .filter(camera => camera.max_violence_prob > 0.5)
        .map(camera => (
          <Card key={camera.id} className="border-l-4 border-red-500">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">
                  Camera {camera.id}
                </CardTitle>
                <Badge variant="destructive">
                  {(camera.max_violence_prob * 100).toFixed(1)}% Violence
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm">
                  <Clock className="h-4 w-4" />
                  <span>Peak at {camera.peak_timestamp}</span>
                </div>
                {/* Timeline visualization */}
                <div className="h-12 bg-gray-100 rounded relative">
                  {camera.timeline.map((point, idx) => (
                    <div
                      key={idx}
                      className="absolute top-0 bottom-0"
                      style={{
                        left: `${(point.timestamp / videoDuration) * 100}%`,
                        width: `${(1 / camera.timeline.length) * 100}%`,
                        backgroundColor: `rgba(239, 68, 68, ${point.violence_prob})`,
                      }}
                    />
                  ))}
                </div>
                <Button variant="outline" size="sm">
                  View Clip
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
    </div>
  </CardContent>
</Card>
```

**Technical Implementation:**

This feature requires advanced computer vision for grid segmentation. I'll provide the architecture in the next section due to length constraints.

**Performance Requirements:**
- Grid detection accuracy: >85% success rate
- Segmentation confidence scoring: Minimum 0.7 for auto-proceed
- Processing speed: 6x real-time (1 minute video in 10 seconds)
- Support up to 100 cameras per grid
- GPU memory optimization: Batch processing of 32 cameras simultaneously

---

## 📊 NON-FUNCTIONAL REQUIREMENTS

### Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| **File Upload Latency** | <5s for 30s video | 95th percentile |
| **Live Detection Latency** | <500ms | End-to-end (camera → display) |
| **Multi-Camera Processing** | 6x real-time | 1 minute video in <10 seconds |
| **Concurrent Users** | 100 simultaneous | Without degradation |
| **GPU Utilization** | <80% average | During peak load |
| **API Response Time** | <200ms | 99th percentile |

### Scalability

- **Horizontal Scaling:** API and ML services independently scalable
- **Load Balancing:** Distribute ML inference across multiple GPU nodes
- **Database:** PostgreSQL read replicas for query scaling
- **Caching:** Redis for real-time data, session management
- **CDN:** Static assets served via CloudFlare

### Security

| Requirement | Implementation |
|-------------|----------------|
| **Authentication** | JWT tokens, refresh tokens |
| **Authorization** | Role-based access control (RBAC) |
| **Data Encryption** | TLS 1.3 for transit, AES-256 for rest |
| **Video Privacy** | Auto-delete uploaded videos after 30 days |
| **API Rate Limiting** | 100 requests/minute per user |
| **Input Validation** | Sanitize all user inputs, file type validation |

### Reliability

- **Uptime Target:** 99.5% (43.8 hours downtime/year)
- **Error Handling:** Graceful degradation, user-friendly error messages
- **Data Backup:** Daily PostgreSQL backups, 30-day retention
- **Incident Recovery:** Automated rollback for failed deployments
- **Monitoring:** Prometheus + Grafana for metrics, alerts

### Usability

- **Mobile Responsive:** Fully functional on tablets/phones
- **Accessibility:** WCAG 2.1 Level AA compliance
- **Browser Support:** Chrome, Firefox, Safari, Edge (latest 2 versions)
- **Loading States:** Skeleton screens, progress indicators
- **Error Messages:** Clear, actionable guidance for users

---

## 🚀 IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Weeks 1-2)

**Week 1: Project Setup**
- ✅ Initialize Next.js 14 project with TypeScript
- ✅ Configure shadcn/ui components
- ✅ Setup NestJS backend with modules structure
- ✅ Create Python ML service (FastAPI) skeleton
- ✅ Configure Docker Compose for local development
- ✅ Setup PostgreSQL + Redis
- ✅ Configure ESLint, Prettier, Husky (code quality)

**Week 2: Core Infrastructure**
- ✅ Implement authentication system (JWT)
- ✅ Create user management API
- ✅ Setup file upload infrastructure (multer, S3)
- ✅ Integrate trained ResNet50V2 + Bi-LSTM model
- ✅ Build basic ML inference pipeline
- ✅ Create database schema (users, cameras, incidents)

**Deliverable:** Working dev environment, authenticated API, model loaded

---

### Phase 2: MVP Features (Weeks 3-5)

**Week 3: File Upload Detection**
- ✅ Build file upload UI (drag-and-drop, shadcn/ui)
- ✅ Implement upload API endpoint
- ✅ Connect to ML service for inference
- ✅ Display results with visualization
- ✅ Add progress indicators, error handling

**Week 4: Live Single-Camera**
- ✅ Implement WebSocket gateway (NestJS)
- ✅ Build webcam capture UI (Next.js)
- ✅ Real-time frame buffering and transmission
- ✅ Live inference endpoint (ML service)
- ✅ Real-time probability meter display
- ✅ Alert system (visual, audio)

**Week 5: MVP Polish & Testing**
- ✅ End-to-end testing (Playwright)
- ✅ Performance optimization
- ✅ Error handling, edge cases
- ✅ User acceptance testing with 3 beta users
- ✅ Documentation (user guide, API docs)

**Deliverable:** Functional /live page with file upload + live camera

---

### Phase 3: Multi-Camera Segmentation (Weeks 6-10)

**Week 6-7: Grid Calibration Tool**
- ✅ Visual grid editor (drag boundaries)
- ✅ Grid preset templates (1×1 to 10×10)
- ✅ Auto-detection algorithm (computer vision)
- ✅ Confidence scoring for segmentation
- ✅ Save/load grid configurations

**Week 8-9: Video Segmentation Pipeline**
- ✅ Frame-by-frame camera extraction
- ✅ Super-resolution upscaling (optional)
- ✅ Parallel per-camera processing
- ✅ Batch inference optimization (32 cameras/batch)
- ✅ Results aggregation and timeline generation

**Week 10: Multi-Camera Dashboard**
- ✅ Per-camera results display
- ✅ Violence timeline visualization
- ✅ Incident clip extraction
- ✅ Export reports (PDF, CSV)

**Deliverable:** Full multi-camera processing capability

---

### Phase 4: Production Readiness (Weeks 11-14)

**Week 11: Performance Optimization**
- ✅ GPU memory optimization
- ✅ Database query optimization
- ✅ Caching strategy (Redis)
- ✅ Load testing (100 concurrent users)

**Week 12: Deployment Infrastructure**
- ✅ Production Docker images
- ✅ Kubernetes manifests (or Docker Swarm)
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ Monitoring (Prometheus, Grafana)
- ✅ Logging (ELK stack or CloudWatch)

**Week 13: Security & Compliance**
- ✅ Security audit
- ✅ Penetration testing
- ✅ GDPR compliance review
- ✅ Data retention policies
- ✅ Terms of Service, Privacy Policy

**Week 14: Beta Launch**
- ✅ Deploy to production environment
- ✅ Onboard 5 pilot customers
- ✅ Customer training, documentation
- ✅ Collect feedback, iterate

**Deliverable:** Production-ready system with paying customers

---

## 🎯 SUCCESS METRICS & KPIs

### Product Metrics

| Metric | Target | Tracking Method |
|--------|--------|----------------|
| **Detection Accuracy** | 90-95% | Ground truth validation on test set |
| **False Positive Rate** | <5% | User feedback, manual review |
| **False Negative Rate** | <10% | Missed incident reports |
| **Grid Segmentation Success** | >85% | Auto-detection confidence scores |
| **System Latency** | <500ms | Application performance monitoring |
| **Uptime** | 99.5% | Uptime monitoring (Pingdom, UptimeRobot) |

### Business Metrics

| Metric | Month 3 Target | Month 6 Target | Month 12 Target |
|--------|----------------|----------------|-----------------|
| **Active Customers** | 5 (beta) | 50 | 500 |
| **Total Cameras Monitored** | 75 | 750 | 7,500 |
| **Monthly Recurring Revenue** | $675 | $6,750 | $67,500 |
| **Customer Retention** | 80% | 85% | 90% |
| **Net Promoter Score (NPS)** | 30+ | 40+ | 50+ |

### User Engagement Metrics

- **Daily Active Users (DAU):** Track user logins
- **Feature Adoption:** % using file upload vs live camera vs multi-camera
- **Session Duration:** Average time spent on platform
- **Incidents Detected:** Total violence incidents flagged
- **Incident Review Time:** How quickly users review alerts

---

## ⚠️ RISKS & MITIGATION STRATEGIES

### Technical Risks

**Risk 1: Grid Segmentation Failure Rate >15%** (HIGH PRIORITY)
- **Probability:** 30%
- **Impact:** System unusable for affected customers
- **Mitigation:**
  - Build robust manual calibration tool
  - Template library for common CCTV systems (Hikvision, Dahua, Avigilon)
  - Confidence scoring - warn users if auto-detection uncertain
  - Fallback to human-assisted calibration workflow

**Risk 2: Model Accuracy <90% on Segmented Feeds**
- **Probability:** 25%
- **Impact:** Value proposition undermined
- **Mitigation:**
  - Test model on cropped/upscaled footage BEFORE launch
  - Fine-tune model on segmented feed samples
  - Add super-resolution preprocessing if needed
  - Set conservative accuracy expectations (87-92% range)

**Risk 3: Real-Time Performance Bottlenecks**
- **Probability:** 20%
- **Impact:** >500ms latency, poor UX
- **Mitigation:**
  - Profile early and often (Weeks 3-4)
  - Optimize hot paths (frame preprocessing, model inference)
  - Use lightweight WebSocket server (not full NestJS overhead)
  - GPU batch processing optimization

**Risk 4: Next.js/NestJS Complexity Overhead**
- **Probability:** 15%
- **Impact:** Slower development, harder debugging
- **Mitigation:**
  - Keep architecture simple initially
  - Use monorepo (Turborepo/Nx) for code sharing
  - Separate ML service from API (different concerns)
  - Extensive logging, monitoring from Day 1

### Business Risks

**Risk 5: 90% Accuracy Insufficient for Market**
- **Probability:** 20%
- **Impact:** Poor customer retention, bad reviews
- **Mitigation:**
  - Position as "AI-assisted" not "automated"
  - Emphasize cost savings over perfect accuracy
  - Human review dashboard for low-confidence detections
  - Over-communicate limitations in sales process

**Risk 6: Competitive Response from Incumbents**
- **Probability:** 40% (within 12 months)
- **Impact:** Price pressure, feature parity
- **Mitigation:**
  - Build data moat (collect incident feedback for model improvement)
  - Integrate with ecosystem (security systems, alarm companies)
  - Focus on underserved SMB segment (harder for enterprise vendors to serve)
  - Build switching costs (saved grid configurations, historical data)

**Risk 7: Slow Customer Acquisition**
- **Probability:** 35%
- **Impact:** Delayed revenue, runway concerns
- **Mitigation:**
  - Free tier (10 cameras free, upsell to paid)
  - Referral program (existing customers refer new ones)
  - Partnership with security system installers
  - Content marketing (thought leadership on AI security)

### Legal/Compliance Risks

**Risk 8: Privacy Regulations (GDPR, CCPA)**
- **Probability:** Medium
- **Impact:** Fines, business restrictions
- **Mitigation:**
  - Auto-delete videos after 30 days (configurable)
  - Clear privacy policy, user consent
  - Data processing agreements with customers
  - Legal review before EU/CA expansion

**Risk 9: Liability for Missed Incidents**
- **Probability:** Low (but high impact)
- **Impact:** Lawsuits, reputational damage
- **Mitigation:**
  - Disclaimer: "AI-assisted, not replacement for human guards"
  - Terms of Service: No guarantee of incident detection
  - Liability insurance (E&O coverage)
  - Transparent accuracy metrics (don't overpromise)

---

## 🛠️ DEVELOPMENT STANDARDS

### Code Quality

**TypeScript Standards:**
- Strict mode enabled
- No `any` types (use `unknown` with type guards)
- Interface over type for object shapes
- Enums for fixed constants

**Testing Requirements:**
- Unit test coverage: >80%
- Integration test coverage: >60%
- E2E tests for critical paths (file upload, live detection)
- Performance tests for latency-sensitive operations

**Documentation:**
- JSDoc comments for public APIs
- README per module/service
- Architecture Decision Records (ADRs) for major choices

### Git Workflow

**Branching Strategy:**
- `main` - production-ready code
- `develop` - integration branch
- `feature/*` - feature branches
- `hotfix/*` - urgent production fixes

**Commit Messages:**
- Format: `type(scope): description`
- Types: feat, fix, docs, style, refactor, test, chore
- Example: `feat(live): add WebSocket connection for real-time detection`

**Pull Request Process:**
1. Create feature branch from `develop`
2. Implement feature with tests
3. Create PR with description, screenshots
4. Code review (at least 1 approval)
5. CI/CD checks pass
6. Merge to `develop`
7. Deploy to staging for QA
8. Merge to `main` for production

### CI/CD Pipeline

**GitHub Actions Workflow:**
```yaml
# .github/workflows/ci.yml
name: CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: npm run lint
      - run: npm run test:unit
      - run: npm run test:e2e

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v3
      - uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: nexaravision/live:latest

  deploy-staging:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop'
    steps:
      - run: kubectl set image deployment/nexara-live nexara-live=nexaravision/live:latest
```

---

## 📚 APPENDIX

### Technology Alternatives Considered

**Frontend Alternatives:**
| Technology | Pros | Cons | Decision |
|------------|------|------|----------|
| **Next.js** (chosen) | SSR, great DX, Vercel deployment | Heavy for simple apps | ✅ Chosen - mature ecosystem |
| React (CRA) | Simple, lightweight | No SSR, SEO challenges | ❌ Rejected - need SEO |
| Svelte | Fast, small bundle | Smaller ecosystem | ❌ Rejected - less mature |

**Backend Alternatives:**
| Technology | Pros | Cons | Decision |
|------------|------|------|----------|
| **NestJS** (chosen) | TypeScript, modular, enterprise-ready | Learning curve | ✅ Chosen - scalability |
| Express.js | Simple, fast | No structure, boilerplate | ❌ Rejected - lacks organization |
| FastAPI (Python) | Fast, async, great for ML | Can't share types with frontend | ✅ Chosen for ML service only |

**Database Alternatives:**
| Technology | Pros | Cons | Decision |
|------------|------|------|----------|
| **PostgreSQL** (chosen) | ACID, JSON support, mature | Vertical scaling limits | ✅ Chosen - reliability |
| MongoDB | Flexible schema, horizontal scaling | No ACID (multi-doc) | ❌ Rejected - need transactions |
| MySQL | Mature, familiar | Less feature-rich than Postgres | ❌ Rejected - Postgres better |

### Glossary

- **GRU:** Gated Recurrent Unit - type of RNN for sequence modeling
- **ResNet50V2:** Deep convolutional neural network architecture
- **Bi-LSTM:** Bidirectional Long Short-Term Memory - sequence model
- **WebSocket:** Full-duplex communication protocol for real-time data
- **shadcn/ui:** Beautifully designed UI component library for React
- **JWT:** JSON Web Token - authentication standard
- **CORS:** Cross-Origin Resource Sharing - browser security mechanism
- **CDN:** Content Delivery Network - distributed edge servers

### References

**Technical Documentation:**
- Next.js: https://nextjs.org/docs
- NestJS: https://docs.nestjs.com
- shadcn/ui: https://ui.shadcn.com
- TensorFlow: https://www.tensorflow.org/api_docs
- FastAPI: https://fastapi.tiangolo.com

**Research Papers:**
- Violence Detection: "Deep Learning for Violence Detection in Videos" (2021)
- ResNet: "Deep Residual Learning for Image Recognition" (He et al., 2016)
- Bi-LSTM: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)

---

## ✅ APPROVAL & SIGN-OFF

**Product Owner:** [Name]
**Tech Lead:** [Name]
**Date:** November 14, 2025

**Approved for Implementation:** YES ✅

**Next Steps:**
1. Kickoff meeting with development team (Week 1, Day 1)
2. Sprint planning (2-week sprints)
3. Daily standups at 9:00 AM
4. Weekly demos on Fridays
5. Retrospectives at end of each sprint

**Success Criteria for MVP Launch (Week 12):**
- ✅ File upload detection working
- ✅ Live single-camera detection working
- ✅ Multi-camera grid segmentation >85% success
- ✅ 5 pilot customers onboarded
- ✅ <500ms latency for live detection
- ✅ 90%+ uptime during beta period

---

**Document Revision History:**
- v1.0 (Nov 14, 2025): Initial PRD approved
