import {
  Controller,
  Post,
  UploadedFile,
  UseInterceptors,
  BadRequestException,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { UploadService } from './upload.service';

@Controller('upload')
export class UploadController {
  constructor(private readonly uploadService: UploadService) {}

  @Post()
  @UseInterceptors(
    FileInterceptor('video', {
      limits: {
        fileSize: 524288000, // 500MB
      },
      fileFilter: (req, file, callback) => {
        const allowedMimeTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv'];
        if (!allowedMimeTypes.includes(file.mimetype)) {
          return callback(
            new BadRequestException('Only video files are allowed'),
            false,
          );
        }
        callback(null, true);
      },
    }),
  )
  async uploadVideo(@UploadedFile() file: Express.Multer.File) {
    const result = await this.uploadService.processVideo(file);

    // Wrap response for frontend compatibility
    return {
      success: true,
      data: {
        violenceProbability: result.violence_probability,
        confidence: result.confidence,
        prediction: result.prediction,
        perClassScores: result.per_class_scores,
        videoMetadata: result.video_metadata,
        timestamp: result.peak_timestamp || new Date().toISOString(),
        frameAnalysis: result.frame_analysis || [],
      },
    };
  }
}
