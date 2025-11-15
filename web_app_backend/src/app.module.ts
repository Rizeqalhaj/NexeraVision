import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { AuthModule } from './auth/auth.module';
import { UploadModule } from './upload/upload.module';
import { LiveModule } from './live/live.module';
import { CamerasModule } from './cameras/cameras.module';
import { IncidentsModule } from './incidents/incidents.module';
import { MlModule } from './ml/ml.module';
import { DatabaseModule } from './database/database.module';
import configuration from './config/configuration';
import { validationSchema } from './config/validation';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      load: [configuration],
      validationSchema,
    }),
    DatabaseModule,
    AuthModule,
    UploadModule,
    LiveModule,
    CamerasModule,
    IncidentsModule,
    MlModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
