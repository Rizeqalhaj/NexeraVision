'use client';

import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  Video,
  Square,
  AlertCircle,
  Activity,
  Clock,
  Brain,
  Shield,
  AlertTriangle,
  Zap,
  BarChart3,
  Cpu,
} from 'lucide-react';
import type { Alert as AlertType } from '@/types/detection';

interface AnalysisResult {
  violence_probability: number;
  confidence: string;
  per_class_scores: {
    non_violence: number;
    violence: number;
  };
  prediction: string;
  inference_time_ms: number;
  backend: string;
}

interface SessionStats {
  totalAnalyses: number;
  avgInferenceTime: number;
  maxViolenceProb: number;
  avgViolenceProb: number;
  detectionRate: number;
  sessionDuration: number;
}

export function LiveCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const sessionStartRef = useRef<number>(0);
  const isDetectingRef = useRef<boolean>(false);

  const [isDetecting, setIsDetecting] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [violenceProb, setViolenceProb] = useState(0);
  const [nonViolenceProb, setNonViolenceProb] = useState(100);
  const [prediction, setPrediction] = useState<string>('none');
  const [confidence, setConfidence] = useState<string>('');
  const [inferenceTime, setInferenceTime] = useState(0);
  const [backend, setBackend] = useState<string>('KERAS');
  const [alerts, setAlerts] = useState<AlertType[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [hasFirstAnalysis, setHasFirstAnalysis] = useState(false);
  const [analysisCount, setAnalysisCount] = useState(0);
  const [sessionStats, setSessionStats] = useState<SessionStats>({
    totalAnalyses: 0,
    avgInferenceTime: 0,
    maxViolenceProb: 0,
    avgViolenceProb: 0,
    detectionRate: 0,
    sessionDuration: 0,
  });
  const frameBuffer = useRef<string[]>([]);
  const analysisHistory = useRef<number[]>([]);
  const inferenceHistory = useRef<number[]>([]);

  const ML_SERVICE_URL = process.env.NEXT_PUBLIC_ML_SERVICE_URL || 'http://localhost:8003/api';

  const updateSessionStats = (newViolenceProb: number, newInferenceTime: number) => {
    analysisHistory.current.push(newViolenceProb);
    inferenceHistory.current.push(newInferenceTime);

    const totalAnalyses = analysisHistory.current.length;
    const avgViolenceProb =
      analysisHistory.current.reduce((a, b) => a + b, 0) / totalAnalyses;
    const avgInferenceTime =
      inferenceHistory.current.reduce((a, b) => a + b, 0) / totalAnalyses;
    const maxViolenceProb = Math.max(...analysisHistory.current);
    const detectionRate =
      (analysisHistory.current.filter((p) => p > 50).length / totalAnalyses) * 100;
    const sessionDuration = Math.floor((Date.now() - sessionStartRef.current) / 1000);

    setSessionStats({
      totalAnalyses,
      avgInferenceTime: Math.round(avgInferenceTime),
      maxViolenceProb: Math.round(maxViolenceProb),
      avgViolenceProb: Math.round(avgViolenceProb),
      detectionRate: Math.round(detectionRate),
      sessionDuration,
    });
  };

  const analyzeFrames = async (frames: string[]) => {
    if (isAnalyzing) return;

    setIsAnalyzing(true);

    try {
      // Convert base64 frames to actual image data for the ML service
      const response = await fetch(`${ML_SERVICE_URL}/detect_live`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          frames: frames.map((f) => f.split(',')[1]), // Remove data:image/jpeg;base64, prefix
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Analysis failed: ${response.statusText} - ${errorText}`);
      }

      const result: AnalysisResult = await response.json();

      // Update state with results
      const violencePct = Math.round(result.violence_probability * 100);
      const nonViolencePct = Math.round(result.per_class_scores.non_violence * 100);

      setViolenceProb(violencePct);
      setNonViolenceProb(nonViolencePct);
      setPrediction(result.prediction);
      setConfidence(result.confidence);
      setInferenceTime(result.inference_time_ms);
      setBackend(result.backend || 'KERAS');
      setHasFirstAnalysis(true);
      setAnalysisCount((prev) => prev + 1);

      // Update session statistics
      updateSessionStats(violencePct, result.inference_time_ms);

      // Check for violence detection
      if (violencePct > 85) {
        handleViolenceDetected(result.violence_probability);
      }
    } catch (err) {
      console.error('Frame analysis failed:', err);
      // Don't stop detection on single analysis failure
    } finally {
      setIsAnalyzing(false);
    }
  };

  const startDetection = async () => {
    try {
      setError(null);

      // Check if mediaDevices API is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
          'Camera API not supported in this browser. Please use HTTPS or localhost.'
        );
      }

      // Check if any camera devices are available
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((device) => device.kind === 'videoinput');

      if (cameras.length === 0) {
        throw new Error(
          'No camera found. Please connect a webcam or use the File Upload tab.'
        );
      }

      // Request webcam access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // Reset session stats
      sessionStartRef.current = Date.now();
      analysisHistory.current = [];
      inferenceHistory.current = [];
      setHasFirstAnalysis(false);
      setAnalysisCount(0);
      setViolenceProb(0);
      setNonViolenceProb(100);
      setPrediction('none');
      setInferenceTime(0);
      setSessionStats({
        totalAnalyses: 0,
        avgInferenceTime: 0,
        maxViolenceProb: 0,
        avgViolenceProb: 0,
        detectionRate: 0,
        sessionDuration: 0,
      });

      setIsDetecting(true);
      isDetectingRef.current = true;
      startFrameCapture();
    } catch (err) {
      let errorMessage = 'Camera access failed';

      if (err instanceof Error) {
        errorMessage = err.message;
      } else if (err instanceof DOMException) {
        switch (err.name) {
          case 'NotFoundError':
            errorMessage =
              'No camera found. Please connect a webcam or use the File Upload tab.';
            break;
          case 'NotAllowedError':
            errorMessage =
              'Camera permission denied. Please allow camera access in browser settings.';
            break;
          case 'NotReadableError':
            errorMessage =
              'Camera is in use by another application. Please close other apps using the camera.';
            break;
          case 'OverconstrainedError':
            errorMessage = 'Camera does not support requested settings. Try a different camera.';
            break;
          default:
            errorMessage = `Camera error: ${err.message}`;
        }
      }

      setError(errorMessage);
      console.error('Failed to start detection:', err);
    }
  };

  const startFrameCapture = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Capture frames at 30fps, send batch of 20 frames every 0.66s
    intervalRef.current = setInterval(() => {
      if (!isDetectingRef.current) return;

      // Capture frame
      ctx.drawImage(video, 0, 0, 224, 224);
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      frameBuffer.current.push(imageData);

      // Send batch when we have 20 frames
      if (frameBuffer.current.length === 20) {
        analyzeFrames(frameBuffer.current);
        frameBuffer.current = frameBuffer.current.slice(10); // Keep 50% overlap
      }
    }, 1000 / 30); // 30fps
  };

  const handleViolenceDetected = (probability: number) => {
    // Play alert sound
    const audio = new Audio('/alert-sound.mp3');
    audio.play().catch(() => {
      // Audio playback failed (user interaction may be required)
    });

    // Add to alerts list
    const newAlert: AlertType = {
      id: Date.now().toString(),
      timestamp: new Date().toLocaleTimeString(),
      confidence: Math.round(probability * 100),
      violenceProbability: probability,
    };

    setAlerts((prev) => [newAlert, ...prev.slice(0, 4)]); // Keep last 5 alerts
  };

  const stopDetection = () => {
    setIsDetecting(false);
    isDetectingRef.current = false;

    // Stop camera stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    // Clear interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Reset state
    frameBuffer.current = [];
  };

  // Update session duration timer
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isDetecting) {
      timer = setInterval(() => {
        const duration = Math.floor((Date.now() - sessionStartRef.current) / 1000);
        setSessionStats((prev) => ({ ...prev, sessionDuration: duration }));
      }, 1000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isDetecting]);

  useEffect(() => {
    return () => {
      stopDetection();
    };
  }, []);

  const violenceDetected = violenceProb > 85;
  const isMediumRisk = violenceProb > 50 && violenceProb <= 85;

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-6">
      {/* Main Camera Card */}
      <Card className="border-[var(--border)] bg-[var(--card-bg)]">
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-[var(--text-primary)]">
            <span className="flex items-center gap-2">
              <Video className="h-5 w-5" />
              Live Camera Feed
            </span>
            {isDetecting && (
              <div className="flex items-center gap-2">
                <Badge variant="destructive" className="bg-[var(--danger-red)]">
                  <span className="animate-pulse mr-1">●</span> LIVE
                </Badge>
                <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/30">
                  <Clock className="h-3 w-3 mr-1" />
                  {formatDuration(sessionStats.sessionDuration)}
                </Badge>
              </div>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Video Preview with Overlay */}
          <div
            className={`relative rounded-lg overflow-hidden bg-black ${
              violenceDetected ? 'ring-4 ring-[var(--danger-red)]' : ''
            }`}
          >
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full aspect-video"
            />
            <canvas ref={canvasRef} width={224} height={224} className="hidden" />

            {/* Analysis Status Indicator */}
            {isDetecting && (
              <div className="absolute top-4 right-4 flex flex-col gap-2">
                <Badge
                  className={`${
                    isAnalyzing
                      ? 'bg-yellow-500/90 text-black'
                      : 'bg-blue-500/90 text-white'
                  }`}
                >
                  <Brain className="h-3 w-3 mr-1" />
                  {isAnalyzing ? 'Processing...' : `#${analysisCount} Active`}
                </Badge>
                {!hasFirstAnalysis && (
                  <Badge className="bg-purple-500/90 text-white animate-pulse">
                    <Activity className="h-3 w-3 mr-1" />
                    Buffering frames...
                  </Badge>
                )}
              </div>
            )}

            {/* Real-time Analysis Overlay - Always visible once first analysis done */}
            {isDetecting && hasFirstAnalysis && (
              <div className="absolute bottom-4 left-4 right-4">
                <Card className="bg-black/80 backdrop-blur border-gray-700">
                  <CardContent className="p-4 space-y-3">
                    {/* Violence vs Non-Violence */}
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-1">
                        <div className="flex items-center gap-1 text-red-400 text-xs font-medium">
                          <AlertTriangle className="h-3 w-3" />
                          Violence
                        </div>
                        <div className="text-xl font-bold text-red-400">{violenceProb}%</div>
                        <Progress
                          value={violenceProb}
                          className="h-1.5 bg-gray-700"
                          indicatorClassName="bg-red-500"
                        />
                      </div>
                      <div className="space-y-1">
                        <div className="flex items-center gap-1 text-green-400 text-xs font-medium">
                          <Shield className="h-3 w-3" />
                          Non-Violence
                        </div>
                        <div className="text-xl font-bold text-green-400">{nonViolenceProb}%</div>
                        <Progress
                          value={nonViolenceProb}
                          className="h-1.5 bg-gray-700"
                          indicatorClassName="bg-green-500"
                        />
                      </div>
                    </div>

                    {/* Prediction and Metrics */}
                    <div className="flex items-center justify-between text-xs">
                      <Badge
                        variant="outline"
                        className={`${
                          prediction === 'violence'
                            ? 'border-red-500 text-red-400 bg-red-500/10'
                            : 'border-green-500 text-green-400 bg-green-500/10'
                        }`}
                      >
                        {prediction === 'violence' ? 'VIOLENCE' : 'SAFE'}
                      </Badge>
                      <div className="flex items-center gap-2 text-gray-400">
                        <span className="flex items-center gap-1">
                          <Zap className="h-3 w-3 text-yellow-400" />
                          {inferenceTime.toFixed(0)}ms
                        </span>
                        <span className="flex items-center gap-1">
                          <Cpu className="h-3 w-3 text-purple-400" />
                          {backend.toUpperCase()}
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <Alert variant="destructive" className="border-[var(--danger-red)]">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Controls */}
          <div className="flex gap-4">
            {!isDetecting ? (
              <Button
                size="lg"
                onClick={startDetection}
                className="flex-1 bg-[var(--accent-blue)] hover:bg-blue-600"
              >
                <Video className="mr-2 h-5 w-5" />
                Start Live Detection
              </Button>
            ) : (
              <Button
                size="lg"
                variant="destructive"
                onClick={stopDetection}
                className="flex-1 bg-[var(--danger-red)] hover:bg-red-600"
              >
                <Square className="mr-2 h-5 w-5" />
                Stop Detection
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Real-time Statistics Dashboard */}
      {isDetecting && sessionStats.totalAnalyses > 0 && (
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-[var(--text-primary)]">
              <BarChart3 className="h-5 w-5" />
              Session Statistics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-xs text-[var(--text-secondary)] mb-1">Total Analyses</div>
                <div className="text-2xl font-bold text-[var(--text-primary)]">
                  {sessionStats.totalAnalyses}
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-xs text-[var(--text-secondary)] mb-1">Avg Inference</div>
                <div className="text-2xl font-bold text-blue-400">
                  {sessionStats.avgInferenceTime}ms
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-xs text-[var(--text-secondary)] mb-1">Max Violence</div>
                <div className="text-2xl font-bold text-red-400">
                  {sessionStats.maxViolenceProb}%
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-xs text-[var(--text-secondary)] mb-1">Avg Violence</div>
                <div className="text-2xl font-bold text-orange-400">
                  {sessionStats.avgViolenceProb}%
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-xs text-[var(--text-secondary)] mb-1">Detection Rate</div>
                <div className="text-2xl font-bold text-yellow-400">
                  {sessionStats.detectionRate}%
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-xs text-[var(--text-secondary)] mb-1">Session Time</div>
                <div className="text-2xl font-bold text-purple-400">
                  {formatDuration(sessionStats.sessionDuration)}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Alert History */}
      {alerts.length > 0 && (
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-[var(--text-primary)]">
              <AlertCircle className="h-5 w-5 text-red-400" />
              Recent Alerts ({alerts.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {alerts.map((alert) => (
                <Alert
                  key={alert.id}
                  variant="destructive"
                  className="border-[var(--danger-red)] bg-red-950/30"
                >
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle className="text-[var(--danger-red)]">Violence Detected</AlertTitle>
                  <AlertDescription className="text-[var(--text-secondary)]">
                    {alert.timestamp} - Confidence: {alert.confidence}%
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
