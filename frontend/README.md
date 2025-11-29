# Frontend - Fire Detection Web Application

## Overview

A modern React-based web application for fire detection using machine learning models. The frontend provides an intuitive interface for users to upload images and receive real-time fire detection and classification results through a progressive two-stage analysis process.

## Features

- **Image Upload**: Drag-and-drop or click to upload images
- **Fire Classification**: Binary classification (fire/no_fire) with confidence scores
- **Object Detection**: Localize and visualize detected fire/smoke regions with bounding boxes
- **Progressive Analysis**: Classifies first, then performs localization for better user experience
- **Real-time Feedback**: Shows processing status (classifying → localizing → complete)
- **Result Visualization**: Displays annotated images with detection boxes and statistics
- **Responsive Design**: Mobile-friendly interface using Tailwind CSS

## Technology Stack

### Frontend Framework
- **React 19**: UI library with hooks
- **Vite**: Next-generation build tool for fast development
- **TypeScript**: Type-safe JavaScript

### Styling & UI
- **Tailwind CSS 4.1**: Utility-first CSS framework
- **Radix UI**: Headless UI components
- **Lucide React**: Icon library
- **tw-animate-css**: Animation utilities

### State Management & HTTP
- **Redux Toolkit**: State management
- **React-Redux**: Redux integration for React
- **Built-in Fetch API**: HTTP requests to backend

### Development Tools
- **ESLint**: Code quality and style linting
- **TypeScript ESLint**: TypeScript-aware linting

## Prerequisites

- Node.js 18+ and npm/yarn
- Backend API running (see Backend README)
- Modern web browser

## Installation & Setup

### 1. Install Dependencies

```bash
cd frontend/CS489-Frontend-Website
npm install
```

### 2. Configure API Endpoint

Edit `src/service/modelsAPI.ts` to point to your backend:

```typescript
// Development
const API_BASE_URL = "http://localhost:8000";

// Production
const API_BASE_URL = "http://your-production-api.com";
```

## Running the Application

### Development Server

```bash
cd frontend/CS489-Frontend-Website
npm run dev
```

The application will be available at `http://localhost:5173`

**Features:**
- Hot module replacement (HMR) for instant updates
- Fast Refresh for React components
- Source maps for debugging

### Build for Production

```bash
npm run build
```

Generates optimized production files in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

Serves the production build locally for testing.

### Lint Code

```bash
npm run lint
```

Runs ESLint to check for code quality issues.

## Project Structure

```
frontend/CS489-Frontend-Website/
├── public/                      (Static assets)
├── src/
│   ├── App.tsx                 (Main app component)
│   ├── main.tsx                (Entry point)
│   ├── index.css               (Global styles)
│   ├── components/
│   │   └── ui/                 (Reusable UI components)
│   ├── features/
│   │   └── landing/
│   │       └── components/
│   │           ├── WaveBackground.tsx     (Animated background)
│   │           ├── landingPage.tsx        (Landing page layout)
│   │           ├── uploadImage.tsx        (Image upload component)
│   │           ├── results.tsx            (Results display)
│   │           └── infoCard.tsx           (Information cards)
│   ├── service/
│   │   └── modelsAPI.ts        (API integration)
│   ├── store/
│   │   └── store.ts            (Redux store setup)
│   ├── styles/
│   │   └── globals.css         (Global styles)
│   ├── types/
│   │   └── service/            (TypeScript type definitions)
│   └── utils/
│       └── utils.ts            (Utility functions)
├── index.html                  (HTML entry point)
├── package.json               (Dependencies)
├── tsconfig.json              (TypeScript config)
├── vite.config.ts             (Vite configuration)
└── tailwind.config.ts         (Tailwind CSS config)
```

## Key Components

### App.tsx
Main component managing:
- Image selection state
- Detection results state
- Processing stage tracking
- API mutations for classification and detection

### Upload Container
- Accepts image files via drag-drop or file input
- Validates file type and size
- Triggers classification mutation

### Results Container
- Displays classification results with confidence scores
- Shows detected objects with bounding boxes
- Presents annotated images from backend

### Wave Background
- Animated gradient background component
- Enhances visual appeal

## API Integration

### Service: `src/service/modelsAPI.ts`

Provides two mutation hooks:

```typescript
// Classification API
const [predictMutation] = usePredictMutation();
await predictMutation({ image });

// Object Detection API
const [detectV2Mutation] = useDetectV2Mutation();
await detectV2Mutation({ image });
```

### Detection Result Type

```typescript
type DetectionResult = {
  classification: "fire" | "no_fire";
  confidence: number;
  localization?: {
    image: string;                    // Base64 encoded image
    detectedObjects: Array<{
      name: string;
      confidence: number;
      box: {
        x1: number;
        y1: number;
        x2: number;
        y2: number;
      };
    }>;
  };
};
```

## Processing Workflow

1. **User uploads image** → Image stored in state
2. **Classification stage** → Sends to `/predict` endpoint
3. **Processing feedback** → Shows "classifying..." status
4. **Localization stage** → Sends to `/detect_v2` endpoint
5. **Processing feedback** → Shows "localizing..." status
6. **Results display** → Shows both classification and detection results
7. **Complete** → User can upload another image or review results

## Environment Variables

Create a `.env` file in the frontend directory (if using environment variables):

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT=30000
```

Note: Currently, the API URL is hardcoded in `modelsAPI.ts`.

## Styling with Tailwind CSS

The application uses Tailwind CSS for all styling:

- **Configuration**: `tailwind.config.ts`
- **Global styles**: `src/styles/globals.css`
- **Component styles**: Inline Tailwind classes in TSX files
- **Custom animations**: From `tw-animate-css` package

### Common Tailwind Classes Used
- Flexbox layouts: `flex`, `flex-col`, `justify-center`, `items-center`
- Spacing: `p-4`, `m-2`, `gap-4`
- Colors: `bg-blue-500`, `text-white`
- Responsive: `md:`, `lg:`, `sm:`
- Effects: `rounded-lg`, `shadow-lg`, `transition`

## Performance Optimization

- **Code Splitting**: Vite automatically splits code for optimal loading
- **Tree Shaking**: Unused code is eliminated in production builds
- **Image Optimization**: Use base64 encoding for displaying results
- **Lazy Loading**: Components can be lazy loaded if needed

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Troubleshooting

### Port Already in Use
```bash
# Change port in vite.config.ts or use:
npm run dev -- --port 3000
```

### CORS Errors
- Ensure backend is running and API URL is correct
- Verify backend CORS configuration includes frontend URL

### Build Errors
```bash
# Clear node_modules and reinstall
rm -r node_modules
npm install
npm run build
```

### TypeScript Errors
```bash
# Regenerate type declarations
npm run build -- --force
```
