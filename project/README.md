# AI Face Diagnosis Frontend

This is the frontend application for the AI Face Diagnosis service, built with React, TypeScript, and Vite.

## Features

- AI-powered face analysis and country matching
- Interactive user interface with drag-and-drop image upload
- Real-time diagnosis results with country rankings
- Responsive design for mobile and desktop
- AdSense integration for monetization
- Privacy-focused (no data storage)

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **Lucide React** - Icons
- **React Router** - Navigation

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start development server:
   ```bash
   npm run dev
   ```

3. Build for production:
   ```bash
   npm run build
   ```

## Development Commands

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run typecheck` - Run TypeScript type checking

## Project Structure

```
src/
├── components/          # React components
├── hooks/              # Custom hooks
├── data/               # Static data files
├── services/           # API services
├── types/              # TypeScript types
├── utils/              # Utility functions
└── config/             # Configuration files
```

## Architecture

This frontend communicates with a FastAPI backend service for face analysis. The application follows modern React patterns with custom hooks for state management and reusable components for UI consistency.
