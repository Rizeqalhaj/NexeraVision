'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Clock, AlertCircle } from 'lucide-react';
import type { DetectionResult as DetectionResultType } from '@/types/detection';
import { getConfidenceColor } from '@/lib/utils';

interface DetectionResultProps {
  result: DetectionResultType;
}

export function DetectionResult({ result }: DetectionResultProps) {
  const violencePercentage = Math.round(result.violenceProbability * 100);
  const isHighRisk = violencePercentage > 85;

  return (
    <Card className="mt-6 border-[var(--border)] bg-[var(--card-bg)]">
      <CardHeader>
        <CardTitle className="text-[var(--text-primary)]">Detection Results</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Violence Probability Gauge */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-[var(--text-secondary)]">
              Violence Probability
            </span>
            <span
              className={`text-3xl font-bold ${
                isHighRisk ? 'text-[var(--danger-red)]' : 'text-[var(--success-green)]'
              }`}
            >
              {violencePercentage}%
            </span>
          </div>
          <Progress
            value={violencePercentage}
            className="h-3 bg-gray-700"
            indicatorClassName={isHighRisk ? 'bg-[var(--danger-red)]' : 'bg-[var(--success-green)]'}
          />
        </div>

        {/* Confidence Badge */}
        <div className="flex items-center gap-2">
          <Badge
            variant={result.confidence === 'High' ? 'destructive' : 'secondary'}
            className={getConfidenceColor(result.confidence)}
          >
            {result.confidence} Confidence
          </Badge>
          {isHighRisk && (
            <Badge variant="destructive" className="bg-[var(--danger-red)]">
              <AlertCircle className="mr-1 h-3 w-3" />
              High Risk
            </Badge>
          )}
        </div>

        {/* Timeline with Peak Violence Moment */}
        {result.peakTimestamp && (
          <>
            <Separator className="bg-[var(--border)]" />
            <div className="flex items-center gap-2 text-[var(--text-secondary)]">
              <Clock className="h-4 w-4" />
              <span className="text-sm">Peak Violence at {result.peakTimestamp}</span>
            </div>
          </>
        )}

        {/* Frame Analysis Timeline */}
        {result.frameAnalysis && result.frameAnalysis.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-[var(--text-primary)]">
              Frame-by-Frame Analysis
            </h4>
            <div className="h-16 bg-gray-800 rounded-lg relative overflow-hidden">
              {result.frameAnalysis.map((frame, idx) => (
                <div
                  key={idx}
                  className="absolute top-0 bottom-0"
                  style={{
                    left: `${(idx / result.frameAnalysis!.length) * 100}%`,
                    width: `${(1 / result.frameAnalysis!.length) * 100}%`,
                    backgroundColor: `rgba(239, 68, 68, ${frame.violenceProb})`,
                  }}
                  title={`Frame ${frame.frameIndex}: ${Math.round(frame.violenceProb * 100)}%`}
                />
              ))}
            </div>
            <div className="flex justify-between text-xs text-[var(--text-secondary)]">
              <span>Start</span>
              <span>End</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
