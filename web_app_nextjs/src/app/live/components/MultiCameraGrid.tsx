'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { GridControls } from './GridControls';
import { CameraCell } from './CameraCell';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertCircle } from 'lucide-react';

interface CameraData {
  id: string;
  label: string;
  violenceProb: number;
  imageData?: string;
  isActive: boolean;
}

export function MultiCameraGrid() {
  const [rows, setRows] = useState(3);
  const [cols, setCols] = useState(3);
  const [isRecording, setIsRecording] = useState(false);
  const [cameras, setCameras] = useState<CameraData[]>([]);
  const [error, setError] = useState<string | null>(null);

  const screenStreamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const totalCameras = rows * cols;

  // Initialize camera grid
  useEffect(() => {
    const newCameras: CameraData[] = [];
    for (let i = 0; i < totalCameras; i++) {
      newCameras.push({
        id: `camera-${i}`,
        label: `Camera ${i + 1}`,
        violenceProb: 0,
        isActive: false,
      });
    }
    setCameras(newCameras);
  }, [totalCameras]);

  const handleGridChange = (newRows: number, newCols: number) => {
    setRows(newRows);
    setCols(newCols);
  };

  const startScreenRecording = async () => {
    try {
      setError(null);

      // Request screen capture
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 },
        },
        audio: false,
      });

      screenStreamRef.current = stream;

      // Create video element to capture screen
      const video = document.createElement('video');
      video.srcObject = stream;
      video.autoplay = true;
      video.playsInline = true;
      videoRef.current = video;

      // Wait for video to load
      await new Promise((resolve) => {
        video.onloadedmetadata = () => {
          video.play();
          resolve(null);
        };
      });

      // Create canvas for segmentation
      const canvas = document.createElement('canvas');
      canvasRef.current = canvas;

      setIsRecording(true);

      // Mark all cameras as active
      setCameras((prev) =>
        prev.map((cam) => ({
          ...cam,
          isActive: true,
        }))
      );

      // Start segmentation and detection
      startSegmentationLoop();
      startDetectionLoop();

      // Handle stream end
      stream.getVideoTracks()[0].addEventListener('ended', () => {
        stopScreenRecording();
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Screen recording failed';
      setError(errorMessage);
      console.error('Failed to start screen recording:', err);
    }
  };

  const startSegmentationLoop = useCallback(() => {
    const segmentFrame = () => {
      if (!isRecording || !videoRef.current || !canvasRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Set canvas size to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw full screen to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Calculate cell dimensions
      const cellWidth = canvas.width / cols;
      const cellHeight = canvas.height / rows;

      // Segment into grid cells
      const newCameraData: string[] = [];

      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const x = col * cellWidth;
          const y = row * cellHeight;

          // Create temporary canvas for this cell
          const cellCanvas = document.createElement('canvas');
          cellCanvas.width = 224; // Model input size
          cellCanvas.height = 224;
          const cellCtx = cellCanvas.getContext('2d');

          if (cellCtx) {
            // Draw this cell region scaled to 224x224
            cellCtx.drawImage(
              canvas,
              x,
              y,
              cellWidth,
              cellHeight,
              0,
              0,
              224,
              224
            );

            // Convert to data URL
            const imageData = cellCanvas.toDataURL('image/jpeg', 0.7);
            newCameraData.push(imageData);
          }
        }
      }

      // Update camera images
      setCameras((prev) =>
        prev.map((cam, index) => ({
          ...cam,
          imageData: newCameraData[index],
        }))
      );

      // Continue loop
      animationFrameRef.current = requestAnimationFrame(segmentFrame);
    };

    segmentFrame();
  }, [isRecording, rows, cols]);

  const startDetectionLoop = () => {
    // Run detection every 2 seconds (throttled for performance)
    detectionIntervalRef.current = setInterval(() => {
      performDetection();
    }, 2000);
  };

  const performDetection = async () => {
    try {
      // Get all active camera image data
      const activeImages = cameras
        .filter((cam) => cam.isActive && cam.imageData)
        .map((cam) => cam.imageData!);

      if (activeImages.length === 0) return;

      // NOTE: Backend API integration
      // The backend needs to implement these endpoints:
      // POST /api/detect/batch - accepts array of base64 images
      // Returns array of { violenceProbability: number } results

      // For now, use mock data until backend is ready
      // Uncomment below when backend endpoints are available:

      // import { detectViolenceBatch } from '@/lib/api';
      // const results = await detectViolenceBatch(activeImages);
      // setCameras((prev) =>
      //   prev.map((cam, index) => ({
      //     ...cam,
      //     violenceProb: results[index]?.violenceProbability * 100 || 0,
      //   }))
      // );

      // Mock detection for demo (remove when backend is ready)
      setCameras((prev) =>
        prev.map((cam) => ({
          ...cam,
          violenceProb: cam.isActive ? Math.random() * 100 : 0,
        }))
      );
    } catch (err) {
      console.error('Detection failed:', err);
      // Don't show error to user, just log it
      // Detection will retry on next interval
    }
  };

  const stopScreenRecording = () => {
    setIsRecording(false);

    // Stop screen stream
    if (screenStreamRef.current) {
      screenStreamRef.current.getTracks().forEach((track) => track.stop());
      screenStreamRef.current = null;
    }

    // Clear animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Clear detection interval
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }

    // Reset cameras to inactive
    setCameras((prev) =>
      prev.map((cam) => ({
        ...cam,
        isActive: false,
        violenceProb: 0,
        imageData: undefined,
      }))
    );

    // Clean up video element
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      stopScreenRecording();
    };
  }, []);

  return (
    <div className="space-y-6">
      {/* Grid Controls */}
      <GridControls
        rows={rows}
        cols={cols}
        onGridChange={handleGridChange}
        isRecording={isRecording}
        onStartRecording={startScreenRecording}
        onStopRecording={stopScreenRecording}
        totalCameras={totalCameras}
      />

      {/* Error Display */}
      {error && (
        <Alert variant="destructive" className="border-[var(--danger-red)]">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Camera Grid */}
      <div
        className="grid gap-4"
        style={{
          gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
        }}
      >
        {cameras.map((camera) => (
          <CameraCell
            key={camera.id}
            id={camera.id}
            label={camera.label}
            violenceProb={camera.violenceProb}
            imageData={camera.imageData}
            isActive={camera.isActive}
          />
        ))}
      </div>

      {/* Performance Note */}
      {isRecording && totalCameras > 16 && (
        <Alert className="border-yellow-500/50 bg-yellow-950/20">
          <AlertCircle className="h-4 w-4 text-yellow-500" />
          <AlertDescription className="text-yellow-500">
            Monitoring {totalCameras} cameras. Detection is throttled to 1 request per 2 seconds per
            camera for optimal performance.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
