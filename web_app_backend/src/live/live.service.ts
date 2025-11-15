import { Injectable, Logger } from '@nestjs/common';
import { MlService, DetectionResult } from '../ml/ml.service';
import { PrismaService } from '../database/prisma.service';

@Injectable()
export class LiveService {
  private readonly logger = new Logger(LiveService.name);

  constructor(
    private readonly mlService: MlService,
    private readonly prisma: PrismaService,
  ) {}

  async analyzeFrames(frames: string[]): Promise<DetectionResult> {
    return await this.mlService.detectLive(frames);
  }

  async logIncident(clientId: string, result: DetectionResult): Promise<void> {
    try {
      // For now, we'll just log to console
      // In production, this would save to database with proper user/camera associations
      this.logger.warn(
        `Violence detected for client ${clientId}: ${result.violence_probability * 100}%`,
      );

      // TODO: Save to database when auth is implemented
      // await this.prisma.incident.create({
      //   data: {
      //     cameraId: cameraId,
      //     userId: userId,
      //     violenceProb: result.violence_probability,
      //     confidence: result.confidence,
      //     timestamp: new Date(),
      //   },
      // });
    } catch (error) {
      this.logger.error('Failed to log incident:', error);
    }
  }
}
