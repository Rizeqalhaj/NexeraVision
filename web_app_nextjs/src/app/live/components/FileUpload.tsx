'use client';

import { useCallback, useState, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Upload,
  AlertCircle,
  CheckCircle2,
  Cpu,
  Clock,
  Zap,
  Brain,
  Activity,
  BarChart3,
  Shield,
  Target,
  Video,
  Frame,
} from 'lucide-react';
import { DetectionResult } from '@/components/live/DetectionResult';
import { uploadWithStreamingProgress } from '@/lib/api';
import type { DetectionResult as DetectionResultType } from '@/types/detection';

interface ProcessingStage {
  stage: string;
  message: string;
  progress: number;
  timestamp: number;
}

interface VideoMetrics {
  filename: string;
  size_mb: number;
  duration?: number;
  fps?: number;
  resolution?: string;
  total_frames?: number;
}

export function FileUpload() {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState<ProcessingStage | null>(null);
  const [stages, setStages] = useState<ProcessingStage[]>([]);
  const [result, setResult] = useState<DetectionResultType | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [videoMetrics, setVideoMetrics] = useState<VideoMetrics | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [timingInfo, setTimingInfo] = useState<{
    extraction_ms?: number;
    inference_ms?: number;
    total_ms?: number;
  } | null>(null);
  const [analysisPhase, setAnalysisPhase] = useState<
    'idle' | 'upload' | 'validation' | 'extraction' | 'inference' | 'complete'
  >('idle');

  const videoRef = useRef<HTMLVideoElement>(null);

  // Generate video thumbnail
  useEffect(() => {
    if (videoPreview && videoRef.current) {
      const video = videoRef.current;
      video.currentTime = 0.5; // Skip to 0.5s for better thumbnail
    }
  }, [videoPreview]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Create video preview
    const previewUrl = URL.createObjectURL(file);
    setVideoPreview(previewUrl);
    setVideoMetrics({
      filename: file.name,
      size_mb: file.size / (1024 * 1024),
    });

    setUploading(true);
    setProgress(0);
    setCurrentStage(null);
    setStages([]);
    setError(null);
    setResult(null);
    setTimingInfo(null);
    setCurrentFrame(0);
    setAnalysisPhase('upload');

    const startTime = Date.now();

    try {
      const data = await uploadWithStreamingProgress(file, (update) => {
        const now = Date.now();

        if (update.type === 'progress') {
          const newStage: ProcessingStage = {
            stage: update.stage || 'processing',
            message: update.message || 'Processing...',
            progress: update.progress || 0,
            timestamp: now - startTime,
          };

          setCurrentStage(newStage);
          setProgress(update.progress || 0);

          // Update analysis phase
          if (update.stage?.includes('validation')) {
            setAnalysisPhase('validation');
          } else if (update.stage?.includes('extraction')) {
            setAnalysisPhase('extraction');
            if (update.frame) {
              setCurrentFrame(update.frame);
            }
          } else if (update.stage?.includes('inference')) {
            setAnalysisPhase('inference');
          }

          // Track video info
          if (update.video_info) {
            setVideoMetrics((prev) => ({
              ...prev!,
              duration: update.video_info?.duration_seconds,
              fps: update.video_info?.fps,
              resolution: `${update.video_info?.width}x${update.video_info?.height}`,
              total_frames: update.video_info?.total_frames,
            }));
          }

          // Track timing info
          if (update.extraction_time_ms) {
            setTimingInfo((prev) => ({
              ...prev,
              extraction_ms: update.extraction_time_ms,
            }));
          }
          if (update.inference_time_ms) {
            setTimingInfo((prev) => ({
              ...prev,
              inference_ms: update.inference_time_ms,
            }));
          }

          setStages((prev) => {
            if (prev.length > 0 && prev[prev.length - 1].stage === update.stage) {
              return [...prev.slice(0, -1), newStage];
            }
            return [...prev, newStage];
          });
        } else if (update.type === 'start') {
          setCurrentStage({
            stage: 'upload',
            message: `Starting analysis of ${update.filename}`,
            progress: 0,
            timestamp: 0,
          });
        }
      });

      const totalTime = Date.now() - startTime;
      setTimingInfo((prev) => ({
        ...prev,
        total_ms: totalTime,
      }));

      setAnalysisPhase('complete');
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setAnalysisPhase('idle');
    } finally {
      setUploading(false);
      setCurrentStage(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv'],
    },
    maxSize: 500 * 1024 * 1024,
    multiple: false,
    disabled: uploading,
  });

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case 'complete':
        return 'text-green-400';
      case 'inference':
        return 'text-purple-400';
      case 'extraction':
        return 'text-blue-400';
      case 'validation':
        return 'text-yellow-400';
      default:
        return 'text-gray-400';
    }
  };

  const renderFrameGrid = () => {
    const frames = Array.from({ length: 20 }, (_, i) => i + 1);
    return (
      <div className="grid grid-cols-10 gap-1">
        {frames.map((frame) => (
          <div
            key={frame}
            className={`h-3 rounded transition-all duration-300 ${
              frame <= currentFrame
                ? 'bg-blue-500 shadow-lg shadow-blue-500/50'
                : 'bg-gray-700'
            } ${frame === currentFrame ? 'animate-pulse' : ''}`}
          />
        ))}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      {!uploading && !result && (
        <Card
          {...getRootProps()}
          className={`border-2 border-dashed cursor-pointer transition-all duration-300 hover:scale-[1.01] ${
            isDragActive
              ? 'border-[var(--accent-blue)] bg-blue-950/30 shadow-2xl shadow-blue-500/20'
              : 'border-[var(--border)] bg-[var(--card-bg)] hover:border-[var(--accent-blue)] hover:shadow-xl'
          }`}
        >
          <input {...getInputProps()} />
          <CardContent className="p-12">
            <div className="text-center space-y-4">
              <div className="relative">
                <Upload className="mx-auto h-16 w-16 text-[var(--accent-blue)]" />
                <div className="absolute -top-1 -right-1 bg-purple-600 text-white text-[10px] px-2 py-0.5 rounded-full font-medium">
                  AI
                </div>
              </div>
              <div>
                <p className="text-xl font-bold text-[var(--text-primary)]">
                  {isDragActive ? 'Release to Analyze' : 'Drop Your Video Here'}
                </p>
                <p className="text-sm text-[var(--text-secondary)] mt-2">
                  Advanced Violence Detection with Real-Time Analysis
                </p>
                <div className="flex justify-center gap-4 mt-4">
                  <span className="inline-flex items-center gap-1 text-xs bg-gray-800 px-3 py-1 rounded-full">
                    <Brain className="h-3 w-3 text-purple-400" /> Deep Learning
                  </span>
                  <span className="inline-flex items-center gap-1 text-xs bg-gray-800 px-3 py-1 rounded-full">
                    <Zap className="h-3 w-3 text-yellow-400" /> Fast Inference
                  </span>
                  <span className="inline-flex items-center gap-1 text-xs bg-gray-800 px-3 py-1 rounded-full">
                    <Shield className="h-3 w-3 text-green-400" /> Accurate
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Professional Analysis Dashboard */}
      {uploading && (
        <div className="space-y-4">
          {/* Video Preview + Status */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Video Preview Card */}
            <Card className="border-[var(--border)] bg-[var(--card-bg)] overflow-hidden">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Video className="h-4 w-4 text-[var(--accent-blue)]" />
                  Video Preview
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                {videoPreview && (
                  <div className="relative">
                    <video
                      ref={videoRef}
                      src={videoPreview}
                      className="w-full h-48 object-cover"
                      muted
                      playsInline
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent" />
                    <div className="absolute bottom-2 left-2 right-2">
                      <p className="text-xs text-white font-mono truncate">
                        {videoMetrics?.filename}
                      </p>
                      <div className="flex gap-3 mt-1">
                        <span className="text-xs text-gray-300">
                          {videoMetrics?.size_mb.toFixed(1)} MB
                        </span>
                        {videoMetrics?.duration && (
                          <span className="text-xs text-gray-300">
                            {videoMetrics.duration.toFixed(1)}s
                          </span>
                        )}
                        {videoMetrics?.fps && (
                          <span className="text-xs text-gray-300">{videoMetrics.fps} FPS</span>
                        )}
                      </div>
                    </div>
                    {/* Processing indicator */}
                    <div className="absolute top-2 right-2">
                      <div className="bg-blue-500/80 text-white text-[10px] px-2 py-1 rounded font-mono">
                        ANALYZING
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Analysis Status Card */}
            <Card className="border-[var(--border)] bg-[var(--card-bg)]">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Activity className={`h-4 w-4 ${getPhaseColor(analysisPhase)} animate-pulse`} />
                  Analysis Pipeline
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {/* Pipeline Steps */}
                <div className="space-y-2">
                  {[
                    {
                      name: 'Video Validation',
                      icon: Shield,
                      phase: 'validation',
                      color: 'text-yellow-400',
                    },
                    {
                      name: 'Frame Extraction',
                      icon: Frame,
                      phase: 'extraction',
                      color: 'text-blue-400',
                    },
                    {
                      name: 'AI Inference',
                      icon: Brain,
                      phase: 'inference',
                      color: 'text-purple-400',
                    },
                  ].map((step, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <step.icon
                        className={`h-4 w-4 ${
                          analysisPhase === step.phase
                            ? `${step.color} animate-pulse`
                            : analysisPhase === 'complete' ||
                                ['validation', 'extraction', 'inference'].indexOf(analysisPhase) >
                                  ['validation', 'extraction', 'inference'].indexOf(step.phase)
                              ? 'text-green-400'
                              : 'text-gray-600'
                        }`}
                      />
                      <span
                        className={`text-xs ${
                          analysisPhase === step.phase
                            ? 'text-white font-semibold'
                            : 'text-gray-400'
                        }`}
                      >
                        {step.name}
                      </span>
                      {analysisPhase === step.phase && (
                        <div className="ml-auto">
                          <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {/* Current Operation */}
                {currentStage && (
                  <div className="mt-3 p-2 bg-gray-800/50 rounded border border-gray-700">
                    <p className="text-xs text-gray-300 truncate">{currentStage.message}</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Frame Extraction Progress */}
          {analysisPhase === 'extraction' && (
            <Card className="border-[var(--border)] bg-[var(--card-bg)]">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Frame className="h-4 w-4 text-blue-400" />
                  Frame Extraction
                  <span className="ml-auto text-xs font-mono text-blue-400">
                    {currentFrame}/20 frames
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {renderFrameGrid()}
                <p className="text-xs text-gray-400 mt-2 text-center">
                  Extracting key frames for temporal analysis
                </p>
              </CardContent>
            </Card>
          )}

          {/* AI Inference Visualization */}
          {analysisPhase === 'inference' && (
            <Card className="border-purple-500/20 bg-purple-950/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Brain className="h-4 w-4 text-purple-400" />
                  Neural Network Processing
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-3">
                  <div className="text-center p-2 bg-gray-800/30 rounded">
                    <p className="text-lg font-mono text-purple-400">20</p>
                    <p className="text-[10px] text-gray-400">Frames</p>
                  </div>
                  <div className="text-center p-2 bg-gray-800/30 rounded">
                    <p className="text-lg font-mono text-purple-400">2.3M</p>
                    <p className="text-[10px] text-gray-400">Parameters</p>
                  </div>
                  <div className="text-center p-2 bg-gray-800/30 rounded">
                    <p className="text-lg font-mono text-purple-400">2</p>
                    <p className="text-[10px] text-gray-400">Classes</p>
                  </div>
                </div>
                <p className="text-[10px] text-center text-gray-500 mt-2">
                  Deep learning inference with attention mechanism
                </p>
              </CardContent>
            </Card>
          )}

          {/* Overall Progress */}
          <Card className="border-[var(--border)] bg-[var(--card-bg)]">
            <CardContent className="pt-4">
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>Overall Progress</span>
                <span className="font-mono">{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="h-3 bg-gray-700" />
              {timingInfo && (
                <div className="grid grid-cols-3 gap-4 mt-4">
                  {timingInfo.extraction_ms && (
                    <div className="text-center p-2 bg-gray-800/50 rounded">
                      <BarChart3 className="h-4 w-4 mx-auto text-blue-400 mb-1" />
                      <p className="text-lg font-mono text-blue-400">
                        {timingInfo.extraction_ms.toFixed(0)}
                      </p>
                      <p className="text-[10px] text-gray-400">Extraction (ms)</p>
                    </div>
                  )}
                  {timingInfo.inference_ms && (
                    <div className="text-center p-2 bg-gray-800/50 rounded">
                      <Zap className="h-4 w-4 mx-auto text-green-400 mb-1" />
                      <p className="text-lg font-mono text-green-400">
                        {timingInfo.inference_ms.toFixed(0)}
                      </p>
                      <p className="text-[10px] text-gray-400">Inference (ms)</p>
                    </div>
                  )}
                  <div className="text-center p-2 bg-gray-800/50 rounded">
                    <Clock className="h-4 w-4 mx-auto text-yellow-400 mb-1" />
                    <p className="text-lg font-mono text-yellow-400">
                      {currentStage ? (currentStage.timestamp / 1000).toFixed(1) : '0.0'}
                    </p>
                    <p className="text-[10px] text-gray-400">Elapsed (s)</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <Alert variant="destructive" className="border-[var(--danger-red)]">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Success Message */}
      {result && !uploading && (
        <Alert className="border-green-500 bg-green-950/30">
          <CheckCircle2 className="h-4 w-4 text-green-400" />
          <AlertDescription className="text-green-400">
            <span className="font-semibold">Analysis Complete!</span>
            {timingInfo?.inference_ms && (
              <span className="ml-2 font-mono bg-green-500/20 px-2 py-0.5 rounded">
                {timingInfo.inference_ms.toFixed(0)}ms inference
              </span>
            )}
            {timingInfo?.total_ms && (
              <span className="ml-2 font-mono text-gray-300">
                Total: {(timingInfo.total_ms / 1000).toFixed(2)}s
              </span>
            )}
          </AlertDescription>
        </Alert>
      )}

      {/* Results Display */}
      {result && <DetectionResult result={result} />}
    </div>
  );
}
