# NexaraVision Backend API

NestJS backend API for the NexaraVision multi-camera violence detection platform.

## Features

- **REST API**: File upload, camera configuration, incident management
- **WebSocket Gateway**: Real-time live camera frame analysis
- **ML Service Integration**: HTTP client to communicate with Python FastAPI ML service
- **Database**: PostgreSQL with Prisma ORM
- **Cache**: Redis for session management and real-time data
- **Type Safety**: Full TypeScript implementation with strict mode

## Prerequisites

- Node.js >= 18.x
- npm >= 9.x
- Docker & Docker Compose (for PostgreSQL and Redis)
- Python ML Service running on port 8000

## Project Structure

```
web_app_backend/
├── src/
│   ├── auth/               # JWT authentication (TODO)
│   ├── upload/             # File upload handling
│   │   ├── upload.controller.ts
│   │   ├── upload.service.ts
│   │   └── upload.module.ts
│   ├── live/               # WebSocket gateway
│   │   ├── live.gateway.ts
│   │   ├── live.service.ts
│   │   └── live.module.ts
│   ├── cameras/            # Camera configuration (TODO)
│   ├── incidents/          # Incident logging (TODO)
│   ├── ml/                 # ML service client
│   │   ├── ml.service.ts
│   │   └── ml.module.ts
│   ├── database/           # Prisma ORM
│   │   ├── prisma.service.ts
│   │   └── database.module.ts
│   ├── config/             # Configuration
│   │   ├── configuration.ts
│   │   └── validation.ts
│   └── app.module.ts
├── prisma/
│   └── schema.prisma       # Database schema
├── docker-compose.yml      # PostgreSQL + Redis
└── .env                    # Environment variables
```

## Installation

1. **Install dependencies:**
```bash
npm install
```

2. **Copy environment variables:**
```bash
cp .env.example .env
```

3. **Start PostgreSQL and Redis:**
```bash
docker-compose up -d
```

4. **Generate Prisma client:**
```bash
npx prisma generate
```

5. **Run database migrations:**
```bash
npx prisma migrate dev --name init
```

## Running the Application

### Development Mode
```bash
npm run start:dev
```

The API will be available at: `http://localhost:3001/api`

### Production Mode
```bash
npm run build
npm run start:prod
```

## API Endpoints

### Upload Video Detection
```http
POST /api/upload
Content-Type: multipart/form-data

Body:
- video: File (MP4, AVI, MOV, MKV)
- Max size: 500MB

Response:
{
  "violenceProbability": 0.85,
  "confidence": "High",
  "timestamp": "00:01:23",
  "frameAnalysis": [...]
}
```

### WebSocket Live Detection
```javascript
// Connect to WebSocket
const socket = io('http://localhost:3001/live');

// Send frames for analysis
socket.emit('analyze_frames', {
  frames: ['base64_frame1', 'base64_frame2', ...]
});

// Listen for detection results
socket.on('detection_result', (data) => {
  console.log('Violence probability:', data.violence_probability);
  console.log('Confidence:', data.confidence);
});

// Listen for alerts
socket.on('alert', (data) => {
  console.log('Violence detected!', data);
});
```

### Health Check
```http
GET /api
```

## Database Schema

See `prisma/schema.prisma` for the complete database schema.

Key models:
- **User**: User accounts
- **Camera**: Camera configurations
- **Incident**: Violence detection incidents
- **Session**: Authentication sessions

## Environment Variables

```env
# Database
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/nexara_vision?schema=public"

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# ML Service
ML_SERVICE_URL=http://localhost:8000

# JWT
JWT_SECRET=your-secret-key-change-in-production
JWT_EXPIRES_IN=7d

# Server
PORT=3001
NODE_ENV=development

# CORS
CORS_ORIGIN=http://localhost:3000

# File Upload
MAX_FILE_SIZE=524288000
UPLOAD_DESTINATION=./uploads
```

## Development

### Generate Module
```bash
npx nest g module <module-name>
```

### Generate Controller
```bash
npx nest g controller <controller-name>
```

### Generate Service
```bash
npx nest g service <service-name>
```

### Run Tests
```bash
npm run test
npm run test:e2e
npm run test:cov
```

### Lint and Format
```bash
npm run lint
npm run format
```

## Prisma Commands

### Generate Prisma Client
```bash
npx prisma generate
```

### Create Migration
```bash
npx prisma migrate dev --name <migration-name>
```

### Reset Database
```bash
npx prisma migrate reset
```

### Open Prisma Studio
```bash
npx prisma studio
```

## Integration with ML Service

The backend communicates with the Python ML service via HTTP:

1. **Video Upload Detection**: POST `/detect`
   - Sends video file buffer
   - Receives violence probability and frame analysis

2. **Live Frame Analysis**: POST `/detect_live`
   - Sends array of base64-encoded frames
   - Receives real-time violence probability

3. **Health Check**: GET `/health`
   - Checks ML service availability

## Next Steps

1. **Implement Authentication**: JWT-based authentication system
2. **Camera Management**: CRUD operations for cameras
3. **Incident Management**: Query and review incidents
4. **Alert System**: Email, SMS, webhook notifications
5. **Testing**: Unit tests and E2E tests
6. **API Documentation**: Swagger/OpenAPI integration

## Troubleshooting

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker-compose ps

# View PostgreSQL logs
docker-compose logs postgres

# Restart containers
docker-compose restart
```

### Module Resolution Issues
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Prisma Client Issues
```bash
# Regenerate Prisma client
npx prisma generate

# Reset and migrate
npx prisma migrate reset
```

## License

Proprietary - NexaraVision
