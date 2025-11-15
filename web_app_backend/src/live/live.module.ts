import { Module } from '@nestjs/common';
import { MlModule } from '../ml/ml.module';
import { LiveService } from './live.service';
import { LiveGateway } from './live.gateway';

@Module({
  imports: [MlModule],
  providers: [LiveService, LiveGateway],
})
export class LiveModule {}
