import { Injectable, BadRequestException } from '@nestjs/common';
import { MlService, DetectionResult } from '../ml/ml.service';

@Injectable()
export class UploadService {
  constructor(private readonly mlService: MlService) {}

  async processVideo(file: Express.Multer.File): Promise<DetectionResult> {
    // Validate file
    if (!file) {
      throw new BadRequestException('No video file provided');
    }

    // Validate file type
    const allowedMimeTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv'];
    if (!allowedMimeTypes.includes(file.mimetype)) {
      throw new BadRequestException(
        `Invalid file type. Allowed types: ${allowedMimeTypes.join(', ')}`,
      );
    }

    // Send to ML service for processing
    const result = await this.mlService.detectViolence(file.buffer, file.originalname);

    return result;
  }
}
