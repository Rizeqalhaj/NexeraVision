# NexaraVision Backend - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Node.js 18+ and npm 9+
- Docker and Docker Compose
- ML Service (Python FastAPI) running on port 8000

### Step-by-Step Setup

#### 1. Start Infrastructure (PostgreSQL + Redis)
```bash
cd /home/admin/Desktop/NexaraVision/web_app_backend
docker-compose up -d
```

Verify containers are running:
```bash
docker-compose ps
```

Expected output:
```
NAME               STATUS         PORTS
nexara_postgres    Up             0.0.0.0:5432->5432/tcp
nexara_redis       Up             0.0.0.0:6379->6379/tcp
```

#### 2. Install Dependencies
```bash
npm install
```

#### 3. Setup Database
```bash
# Generate Prisma client
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/nexara_vision?schema=public" npx prisma generate

# Run migrations
npx prisma migrate dev --name init
```

#### 4. Start Development Server
```bash
npm run start:dev
```

Expected output:
```
[Nest] LOG [NestApplication] Nest application successfully started
🚀 NexaraVision Backend running on http://localhost:3001/api
```

### ✅ Verify Installation

#### Test Health Endpoint
```bash
curl http://localhost:3001/api
```

Expected: `Hello World!`

#### Test Upload Endpoint
```bash
curl -X POST http://localhost:3001/api/upload \
  -F "video=@/path/to/test-video.mp4"
```

Expected (if ML service is running):
```json
{
  "violenceProbability": 0.12,
  "confidence": "Low",
  "timestamp": "00:00:05",
  "frameAnalysis": [...]
}
```

#### Test WebSocket Connection
```javascript
// Install wscat: npm install -g wscat
wscat -c ws://localhost:3001/live

// Send test message:
{"type": "ping"}

// Expected response:
{"type": "pong", "timestamp": 1234567890}
```

## 📦 What's Included

### Implemented Features (Phase 1)
✅ **Upload API** (`POST /api/upload`)
- File upload with validation (MP4, AVI, MOV, MKV)
- Max file size: 500MB
- Integration with ML service

✅ **Live Detection WebSocket** (`ws://localhost:3001/live`)
- Real-time frame analysis
- Violence detection alerts (>85% threshold)
- Connection management

✅ **ML Service Client**
- HTTP communication with Python FastAPI
- Video detection: `POST /detect`
- Live detection: `POST /detect_live`
- Health check: `GET /health`

✅ **Database (Prisma ORM)**
- PostgreSQL integration
- Schema: Users, Cameras, Incidents, Sessions
- Type-safe queries

✅ **Infrastructure**
- Docker Compose (PostgreSQL + Redis)
- Environment configuration
- Global validation pipes
- CORS enabled

### TODO (Phase 2+)
⏳ **Authentication** (Week 3-4)
- JWT implementation
- User registration/login
- Auth guards

⏳ **Camera Management** (Week 4)
- CRUD endpoints
- Grid configuration
- Status tracking

⏳ **Incident Management** (Week 5)
- Query incidents
- Review false positives
- Export reports

## 🔧 Common Commands

### Development
```bash
npm run start:dev      # Start with hot reload
npm run start:debug    # Start with debugging
npm run build          # Compile TypeScript
npm run start:prod     # Start production build
```

### Testing
```bash
npm run test           # Unit tests
npm run test:e2e       # End-to-end tests
npm run test:cov       # Coverage report
```

### Database
```bash
npx prisma generate              # Generate Prisma client
npx prisma migrate dev           # Create and run migration
npx prisma migrate reset         # Reset database
npx prisma studio                # Open database GUI
```

### Docker
```bash
docker-compose up -d             # Start containers
docker-compose down              # Stop containers
docker-compose logs postgres     # View PostgreSQL logs
docker-compose logs redis        # View Redis logs
docker-compose restart           # Restart all containers
```

## 📂 Project Structure

```
web_app_backend/
├── src/
│   ├── auth/           # Authentication (TODO)
│   ├── cameras/        # Camera management (TODO)
│   ├── incidents/      # Incident logging (TODO)
│   ├── upload/         # ✅ File upload endpoint
│   ├── live/           # ✅ WebSocket gateway
│   ├── ml/             # ✅ ML service client
│   ├── database/       # ✅ Prisma service
│   ├── config/         # ✅ Configuration
│   └── main.ts         # ✅ Application entry
├── prisma/
│   └── schema.prisma   # ✅ Database schema
├── docker-compose.yml  # ✅ Infrastructure
└── .env                # ✅ Environment variables
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Check what's using port 3001
lsof -i :3001

# Kill the process
kill -9 <PID>
```

### Database Connection Failed
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Restart PostgreSQL
docker-compose restart postgres

# Check logs
docker-compose logs postgres
```

### Prisma Client Not Found
```bash
# Regenerate Prisma client
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/nexara_vision?schema=public" npx prisma generate

# Clear and reinstall
rm -rf node_modules
npm install
```

### ML Service Unavailable
```bash
# Check ML service is running
curl http://localhost:8000/health

# Start ML service (from Python directory)
cd /path/to/ml_service
python app.py
```

## 🔐 Environment Variables

**Required**:
```env
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/nexara_vision?schema=public"
ML_SERVICE_URL=http://localhost:8000
JWT_SECRET=your-secret-key
CORS_ORIGIN=http://localhost:3000
```

**Optional (with defaults)**:
```env
PORT=3001
REDIS_HOST=localhost
REDIS_PORT=6379
JWT_EXPIRES_IN=7d
MAX_FILE_SIZE=524288000
UPLOAD_DESTINATION=./uploads
```

## 📚 API Documentation

### Upload Video
```http
POST /api/upload
Content-Type: multipart/form-data

Body:
video: <file> (MP4, AVI, MOV, MKV, max 500MB)

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
// Connect
const socket = io('http://localhost:3001/live');

// Send frames
socket.emit('analyze_frames', {
  frames: ['base64_frame1', 'base64_frame2', ...]
});

// Receive results
socket.on('detection_result', (data) => {
  console.log(data.violence_probability);
  console.log(data.confidence);
});

// Receive alerts (when violence > 85%)
socket.on('alert', (data) => {
  console.log('Violence detected!', data);
});

// Error handling
socket.on('error', (error) => {
  console.error('WebSocket error:', error);
});
```

## 🎯 Next Steps

1. **Test the upload endpoint** with sample videos
2. **Test WebSocket** with live camera frames
3. **Implement authentication** (JWT) in Week 3-4
4. **Build camera management** endpoints
5. **Add incident logging** with database storage
6. **Write unit tests** for services
7. **Add E2E tests** for critical paths

## 💡 Tips

- Use `npx prisma studio` to visually inspect the database
- Check WebSocket connections in browser DevTools > Network > WS
- Monitor logs with `npm run start:dev` for real-time debugging
- Use Postman or curl to test API endpoints
- Keep ML service running for upload/live detection to work

## 📞 Integration Points

**With ML Service (Python FastAPI)**:
- Upload detection: Backend → ML service `/detect`
- Live detection: Backend → ML service `/detect_live`
- Health check: Backend → ML service `/health`

**With Frontend (Next.js)**:
- REST API: `/api/upload` for file upload
- WebSocket: `/live` for real-time detection
- CORS enabled for `http://localhost:3000`

**With Database (PostgreSQL)**:
- Prisma ORM for type-safe queries
- Auto-generated migrations
- Database GUI with Prisma Studio

## ✨ Features Status

| Feature | Status | Location |
|---------|--------|----------|
| Upload API | ✅ DONE | `src/upload/` |
| WebSocket Gateway | ✅ DONE | `src/live/` |
| ML Service Client | ✅ DONE | `src/ml/` |
| Database ORM | ✅ DONE | `src/database/` |
| Configuration | ✅ DONE | `src/config/` |
| Authentication | ⏳ TODO | `src/auth/` |
| Camera Management | ⏳ TODO | `src/cameras/` |
| Incident Logging | ⏳ TODO | `src/incidents/` |

---

**Ready to Start**: All Phase 1 infrastructure is complete!

**Next Developer**: Begin with authentication module following PRD timeline.

**Questions?**: See `README.md` for detailed documentation.
