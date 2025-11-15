import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  OnGatewayConnection,
  OnGatewayDisconnect,
  MessageBody,
  ConnectedSocket,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { LiveService } from './live.service';
import { Logger } from '@nestjs/common';

@WebSocketGateway({
  namespace: '/live',
  cors: {
    origin: '*',
    credentials: true,
  },
})
export class LiveGateway implements OnGatewayConnection, OnGatewayDisconnect {
  @WebSocketServer()
  server: Server;

  private readonly logger = new Logger(LiveGateway.name);

  constructor(private readonly liveService: LiveService) {}

  handleConnection(client: Socket) {
    this.logger.log(`Client connected: ${client.id}`);
  }

  handleDisconnect(client: Socket) {
    this.logger.log(`Client disconnected: ${client.id}`);
  }

  @SubscribeMessage('analyze_frames')
  async handleFrameAnalysis(
    @ConnectedSocket() client: Socket,
    @MessageBody() payload: { frames: string[] },
  ) {
    try {
      const { frames } = payload;

      if (!frames || !Array.isArray(frames) || frames.length === 0) {
        client.emit('error', { message: 'Invalid frames data' });
        return;
      }

      // Send frames to ML service
      const result = await this.liveService.analyzeFrames(frames);

      // Send result back to client
      client.emit('detection_result', {
        violence_probability: result.violence_probability,
        confidence: result.confidence,
        timestamp: new Date().toISOString(),
      });

      // If violence detected, log incident
      if (result.violence_probability > 0.85) {
        await this.liveService.logIncident(client.id, result);

        // Send alert
        client.emit('alert', {
          message: 'Violence detected!',
          probability: result.violence_probability,
          confidence: result.confidence,
        });
      }
    } catch (error) {
      this.logger.error('Error analyzing frames:', error);
      client.emit('error', { message: 'Failed to analyze frames' });
    }
  }

  @SubscribeMessage('ping')
  handlePing(@ConnectedSocket() client: Socket) {
    client.emit('pong', { timestamp: Date.now() });
  }
}
