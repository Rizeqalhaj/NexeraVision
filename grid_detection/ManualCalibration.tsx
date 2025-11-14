"""
Manual Calibration UI Component for Grid Detection Fallback

Used when automatic detection fails (9-20% of cases per research)
Provides drag-and-drop interface for users to define camera boundaries

Research context: https://consensus.app/search/edge-detection-cctv-uis/ZULDwLVzSuWf7E0iNGRkBA/
"""

'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Grid3x3,
  Plus,
  Trash2,
  Save,
  RotateCcw,
  CheckCircle,
  AlertCircle,
  Info
} from 'lucide-react';

interface CameraRegion {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  label?: string;
}

interface ManualCalibrationProps {
  /** Screenshot image URL or base64 */
  imageUrl: string;
  /** Auto-detected regions (if any) - use as starting point */
  autoDetectedRegions?: CameraRegion[];
  /** Callback when calibration is complete */
  onCalibrationComplete: (regions: CameraRegion[]) => void;
  /** Optional: Expected grid layout for validation */
  expectedLayout?: { rows: number; cols: number };
}

export default function ManualCalibration({
  imageUrl,
  autoDetectedRegions = [],
  onCalibrationComplete,
  expectedLayout
}: ManualCalibrationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [regions, setRegions] = useState<CameraRegion[]>(autoDetectedRegions);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [gridRows, setGridRows] = useState(expectedLayout?.rows || 3);
  const [gridCols, setGridCols] = useState(expectedLayout?.cols || 3);
  const [mode, setMode] = useState<'manual' | 'auto-grid'>('manual');

  // Load image
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      setImage(img);
      if (canvasRef.current) {
        const canvas = canvasRef.current;
        canvas.width = img.width;
        canvas.height = img.height;
        redrawCanvas(img);
      }
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // Redraw canvas with image and regions
  const redrawCanvas = (img: HTMLImageElement | null = image) => {
    if (!canvasRef.current || !img) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw image
    ctx.drawImage(img, 0, 0);

    // Draw regions
    regions.forEach((region) => {
      const isSelected = region.id === selectedRegion;

      // Draw rectangle
      ctx.strokeStyle = isSelected ? '#3b82f6' : '#22c55e';
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(region.x, region.y, region.width, region.height);

      // Draw semi-transparent fill
      ctx.fillStyle = isSelected ? 'rgba(59, 130, 246, 0.1)' : 'rgba(34, 197, 94, 0.1)';
      ctx.fillRect(region.x, region.y, region.width, region.height);

      // Draw label
      if (region.label) {
        ctx.fillStyle = isSelected ? '#3b82f6' : '#22c55e';
        ctx.font = 'bold 16px Inter';
        ctx.fillText(region.label, region.x + 10, region.y + 25);
      }

      // Draw resize handles if selected
      if (isSelected) {
        const handleSize = 8;
        ctx.fillStyle = '#3b82f6';

        // Corner handles
        ctx.fillRect(region.x - handleSize / 2, region.y - handleSize / 2, handleSize, handleSize);
        ctx.fillRect(region.x + region.width - handleSize / 2, region.y - handleSize / 2, handleSize, handleSize);
        ctx.fillRect(region.x - handleSize / 2, region.y + region.height - handleSize / 2, handleSize, handleSize);
        ctx.fillRect(region.x + region.width - handleSize / 2, region.y + region.height - handleSize / 2, handleSize, handleSize);
      }
    });

    // Draw current drawing rectangle
    if (isDrawing && drawStart) {
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      const width = (drawStart.x - region.x);
      const height = (drawStart.y - region.y);
      // Will be drawn on mouse move
      ctx.setLineDash([]);
    }
  };

  useEffect(() => {
    redrawCanvas();
  }, [regions, selectedRegion, image]);

  // Mouse event handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if clicking on existing region
    const clickedRegion = regions.find(
      (r) => x >= r.x && x <= r.x + r.width && y >= r.y && y <= r.y + r.height
    );

    if (clickedRegion) {
      setSelectedRegion(clickedRegion.id);
    } else {
      // Start drawing new region
      setIsDrawing(true);
      setDrawStart({ x, y });
      setSelectedRegion(null);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !drawStart || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Redraw with preview rectangle
    redrawCanvas();

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(
      Math.min(drawStart.x, x),
      Math.min(drawStart.y, y),
      Math.abs(x - drawStart.x),
      Math.abs(y - drawStart.y)
    );
    ctx.setLineDash([]);
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !drawStart || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Create new region
    const width = Math.abs(x - drawStart.x);
    const height = Math.abs(y - drawStart.y);

    // Only create if region is large enough
    if (width > 20 && height > 20) {
      const newRegion: CameraRegion = {
        id: `region-${Date.now()}`,
        x: Math.min(drawStart.x, x),
        y: Math.min(drawStart.y, y),
        width,
        height,
        label: `Camera ${regions.length + 1}`
      };

      setRegions([...regions, newRegion]);
    }

    setIsDrawing(false);
    setDrawStart(null);
  };

  // Auto-grid generation
  const generateAutoGrid = () => {
    if (!image) return;

    const newRegions: CameraRegion[] = [];
    const cellWidth = image.width / gridCols;
    const cellHeight = image.height / gridRows;

    for (let row = 0; row < gridRows; row++) {
      for (let col = 0; col < gridCols; col++) {
        newRegions.push({
          id: `auto-${row}-${col}`,
          x: Math.round(col * cellWidth),
          y: Math.round(row * cellHeight),
          width: Math.round(cellWidth),
          height: Math.round(cellHeight),
          label: `Camera ${row * gridCols + col + 1}`
        });
      }
    }

    setRegions(newRegions);
  };

  // Delete selected region
  const deleteSelectedRegion = () => {
    if (!selectedRegion) return;
    setRegions(regions.filter((r) => r.id !== selectedRegion));
    setSelectedRegion(null);
  };

  // Clear all regions
  const clearAllRegions = () => {
    setRegions([]);
    setSelectedRegion(null);
  };

  // Save calibration
  const saveCalibration = () => {
    if (regions.length === 0) {
      alert('Please define at least one camera region');
      return;
    }

    // Validate regions
    const validated = regions.every(
      (r) => r.width > 0 && r.height > 0 && r.x >= 0 && r.y >= 0
    );

    if (!validated) {
      alert('Some regions have invalid dimensions');
      return;
    }

    onCalibrationComplete(regions);
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Grid3x3 className="h-5 w-5" />
              Manual Camera Grid Calibration
            </CardTitle>
            <CardDescription>
              Define camera boundaries manually (used when auto-detection fails)
            </CardDescription>
          </div>
          <Badge variant={regions.length > 0 ? 'default' : 'secondary'}>
            {regions.length} cameras defined
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Research context alert */}
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            <strong>Research-validated approach:</strong> Automatic detection succeeds in 80-91% of cases.
            Use manual calibration for challenging layouts (overlays, low contrast).
          </AlertDescription>
        </Alert>

        {/* Mode selector */}
        <div className="flex gap-2">
          <Button
            variant={mode === 'manual' ? 'default' : 'outline'}
            onClick={() => setMode('manual')}
            className="flex-1"
          >
            Manual Draw
          </Button>
          <Button
            variant={mode === 'auto-grid' ? 'default' : 'outline'}
            onClick={() => setMode('auto-grid')}
            className="flex-1"
          >
            Auto Grid
          </Button>
        </div>

        {/* Auto-grid controls */}
        {mode === 'auto-grid' && (
          <div className="space-y-4 p-4 border rounded-lg bg-muted/50">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">Rows: {gridRows}</label>
                <Slider
                  value={[gridRows]}
                  onValueChange={(v) => setGridRows(v[0])}
                  min={1}
                  max={10}
                  step={1}
                  className="mt-2"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Columns: {gridCols}</label>
                <Slider
                  value={[gridCols]}
                  onValueChange={(v) => setGridCols(v[0])}
                  min={1}
                  max={10}
                  step={1}
                  className="mt-2"
                />
              </div>
            </div>
            <Button onClick={generateAutoGrid} className="w-full">
              Generate {gridRows}×{gridCols} Grid
            </Button>
          </div>
        )}

        {/* Instructions */}
        {mode === 'manual' && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              <strong>Instructions:</strong> Click and drag to draw camera boundaries.
              Click a region to select it. Use buttons below to manage regions.
            </AlertDescription>
          </Alert>
        )}

        {/* Canvas */}
        <div className="relative border rounded-lg overflow-hidden bg-black">
          <canvas
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            className="max-w-full h-auto cursor-crosshair"
          />
        </div>

        {/* Actions */}
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={deleteSelectedRegion}
            disabled={!selectedRegion}
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Delete Selected
          </Button>

          <Button variant="outline" size="sm" onClick={clearAllRegions}>
            <RotateCcw className="h-4 w-4 mr-2" />
            Clear All
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setRegions(autoDetectedRegions)}
            disabled={autoDetectedRegions.length === 0}
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset to Auto-Detected
          </Button>

          <div className="flex-1" />

          <Button onClick={saveCalibration} disabled={regions.length === 0}>
            <CheckCircle className="h-4 w-4 mr-2" />
            Save Calibration ({regions.length} cameras)
          </Button>
        </div>

        {/* Expected layout validation */}
        {expectedLayout && (
          <Alert variant={regions.length === expectedLayout.rows * expectedLayout.cols ? 'default' : 'destructive'}>
            <Info className="h-4 w-4" />
            <AlertDescription>
              Expected {expectedLayout.rows}×{expectedLayout.cols} = {expectedLayout.rows * expectedLayout.cols} cameras.
              Currently defined: {regions.length} cameras.
            </AlertDescription>
          </Alert>
        )}

        {/* Region list */}
        {regions.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Defined Regions:</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {regions.map((region) => (
                <Button
                  key={region.id}
                  variant={selectedRegion === region.id ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedRegion(region.id)}
                  className="justify-start"
                >
                  {region.label || region.id}
                  <span className="ml-auto text-xs opacity-70">
                    {region.width}×{region.height}
                  </span>
                </Button>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Example usage in /live page:
/*
'use client';

import { useState } from 'react';
import ManualCalibration from '@/components/ManualCalibration';

export default function LivePage() {
  const [screenshot, setScreenshot] = useState<string | null>(null);
  const [autoDetectionResult, setAutoDetectionResult] = useState<any>(null);

  const handleScreenshot = async () => {
    // Capture screen using getDisplayMedia
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { width: 3840, height: 2160 }
    });

    const track = stream.getVideoTracks()[0];
    const imageCapture = new ImageCapture(track);
    const bitmap = await imageCapture.grabFrame();

    // Convert to blob and URL
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);

    canvas.toBlob(async (blob) => {
      const url = URL.createObjectURL(blob);
      setScreenshot(url);

      // Try auto-detection
      const formData = new FormData();
      formData.append('screenshot', blob);

      const response = await fetch('/api/detect-grid', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      setAutoDetectionResult(result);

      track.stop();
    });
  };

  const handleCalibrationComplete = async (regions) => {
    console.log('Calibration complete:', regions);

    // Save grid configuration
    await fetch('/api/save-grid-config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ regions })
    });

    // Start monitoring with saved configuration
    startMonitoring(regions);
  };

  return (
    <div>
      {!screenshot ? (
        <Button onClick={handleScreenshot}>
          Capture Screen for Calibration
        </Button>
      ) : autoDetectionResult?.requires_manual ? (
        <ManualCalibration
          imageUrl={screenshot}
          autoDetectedRegions={autoDetectionResult.regions}
          expectedLayout={autoDetectionResult.grid_layout}
          onCalibrationComplete={handleCalibrationComplete}
        />
      ) : (
        <div>Auto-detection succeeded! {autoDetectionResult.regions.length} cameras detected.</div>
      )}
    </div>
  );
}
*/
