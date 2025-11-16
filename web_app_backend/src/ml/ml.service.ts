import { Injectable, HttpException, HttpStatus } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';
import FormData from 'form-data';

export interface DetectionResult {
  violence_probability: number;
  confidence: string;
  prediction: string;
  per_class_scores: {
    non_violence: number;
    violence: number;
  };
  video_metadata?: {
    filename: string;
    duration_seconds: number;
    fps: number;
    resolution: string;
    total_frames: number;
  };
  peak_timestamp?: string;
  frame_analysis?: any[];
}

@Injectable()
export class MlService {
  private mlServiceUrl: string;

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService,
  ) {
    this.mlServiceUrl = this.configService.get<string>('mlService.url') || 'http://localhost:8000';
  }

  async detectViolence(videoBuffer: Buffer, filename: string): Promise<DetectionResult> {
    try {
      const formData = new FormData();
      formData.append('video', videoBuffer, {
        filename,
        contentType: 'video/mp4',
      });

      const response = await firstValueFrom(
        this.httpService.post(`${this.mlServiceUrl}/detect`, formData, {
          headers: formData.getHeaders(),
          maxBodyLength: Infinity,
          maxContentLength: Infinity,
        }),
      );

      return response.data;
    } catch (error) {
      console.error('ML Service Error:', error.message);
      throw new HttpException(
        'Failed to process video with ML service',
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }

  async detectLive(frames: string[]): Promise<DetectionResult> {
    try {
      const response = await firstValueFrom(
        this.httpService.post(`${this.mlServiceUrl}/detect_live`, { frames }),
      );

      return response.data;
    } catch (error) {
      console.error('ML Service Error:', error.message);
      throw new HttpException(
        'Failed to process live frames with ML service',
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await firstValueFrom(
        this.httpService.get(`${this.mlServiceUrl}/health`),
      );
      return response.status === 200;
    } catch (error) {
      return false;
    }
  }
}
