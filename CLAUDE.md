# CLAUDE.md - Project Instructions for Claude

## Project Overview
This is a face diagnosis web application that analyzes user photos and suggests which country they might be from based on facial features. The project consists of:
- Frontend: React + TypeScript + Vite application
- Backend: Python FastAPI server with face recognition capabilities

## Project Structure
```
/
├── backend/                 # Python backend server
│   ├── main.py             # FastAPI server
│   ├── crop_face.py        # Face detection and cropping
│   ├── rank_face.py        # Face similarity ranking
│   └── data/               # Country data and metadata
├── project/                # Frontend React application
│   ├── src/                # Source code
│   ├── public/             # Static assets
│   └── package.json        # Node dependencies
└── metadata.csv            # Country representatives data
```

## Development Commands

### Frontend (in project/ directory)
```bash
npm install              # Install dependencies
npm run dev             # Start development server
npm run build           # Build for production
npm run lint            # Run ESLint
npm run typecheck       # Run TypeScript type checking
```

### Backend (in backend/ directory)
```bash
pip install -r requirements.txt    # Install Python dependencies
python main.py                     # Start FastAPI server
```

## Key Features
1. Face diagnosis using AI/ML
2. Country matching based on facial features
3. Gender-specific analysis
4. Multi-language support (English/Japanese)
5. Google AdSense integration
6. Privacy-focused (no data storage)

## Important Notes
- The application uses face_recognition library for face detection
- Country representatives are pre-analyzed and stored in metadata
- Frontend communicates with backend via REST API
- AdSense is integrated for monetization
- Google Analytics tracks user interactions

## Testing
- Frontend: Use `npm run lint` and `npm run typecheck` before committing
- Backend: Test API endpoints with FastAPI's automatic docs at `/docs`

## Deployment

### Current Deployment Options

#### Frontend
- **Recommended**: Vercel (free tier, optimized for React)
- **Alternative**: Fly.io (see migration guide)
- Built with Vite and can be deployed to any static hosting

#### Backend
- **Current**: Render ($7/month)
- **Recommended**: Fly.io (free tier, $84/year savings)
- Requires Python environment with face_recognition dependencies

### Fly.io Migration
For cost optimization, backend can be migrated from Render to Fly.io:
- **Cost savings**: $84/year (from $7/month to free tier)
- **Performance**: Better latency for Japanese users (Tokyo region)
- **Auto-scaling**: Machines sleep when not in use
- **Migration guide**: See `docs/FLY_IO_MIGRATION.md`

### Deployment Commands

#### Fly.io Backend Deployment
```bash
cd backend
fly auth login
fly launch --copy-config
fly deploy
```

#### Vercel Frontend Deployment
```bash
cd project
vercel --prod
```