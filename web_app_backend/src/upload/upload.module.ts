import { Module } from '@nestjs/common';
import { MlModule } from '../ml/ml.module';
import { UploadService } from './upload.service';
import { UploadController } from './upload.controller';

@Module({
  imports: [MlModule],
  providers: [UploadService],
  controllers: [UploadController],
})
export class UploadModule {}
