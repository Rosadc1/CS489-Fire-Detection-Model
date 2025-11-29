# CS489 Fire Detection Model - Local Development Setup Guide

## Project Overview

**CS489 Fire Detection Model** is a full-stack machine learning application for real-time fire and smoke detection in images. It combines a FastAPI backend with PyTorch/YOLO models and a modern React frontend.

### Key Features
- **Binary Classification**: Fire vs. No Fire detection
- **Object Localization**: Detects and localizes fire/smoke regions
- **Two-Stage Analysis**: Classification followed by object detection
- **Real-time Processing**: Instant results with visual feedback
- **SAHI Enhancement**: Improved detection of small objects

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (Port 5173)               │
│  - Image Upload Interface                                   │
│  - Real-time Processing Feedback                            │
│  - Results Visualization                                    │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP/CORS
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                   FastAPI Backend (Port 8000)               │
│  - /predict (CNN Classification)                            │
│  - /detect (YOLO Object Detection)                          │
│  - /detect_v2 (YOLO + SAHI Detection)                       │
└─────────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
    ┌───▼────┐                    ┌────▼────┐
    │ CNN    │                    │ YOLO    │
    │Model   │                    │ Model   │
    └────────┘                    └─────────┘
```

## System Requirements

### Minimum Specifications
- **OS**: Windows 10/11, macOS 10.14+, Linux (Ubuntu 18.04+)
- **CPU**: Intel i5/AMD Ryzen 5 or equivalent
- **RAM**: 8GB (16GB recommended for smooth operation)
- **Storage**: 10GB free space (for models and dependencies)

### Recommended Specifications
- **CPU**: Intel i7/AMD Ryzen 7 or better
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster inference)

### Software
- **Python**: 3.9 or higher
- **Node.js**: 18.0.0 or higher
- **npm**: 9.0.0 or higher
- **Git**: Latest version

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/Rosadc1/CS489-Fire-Detection-Model.git
cd CS489-Fire-Detection-Model
```

### Step 2: Backend Setup

#### 2.1 Create Python Virtual Environment

```bash
cd backend
python -m venv venv
```

**Activate Virtual Environment:**

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

#### 2.2 Install Python Dependencies

```bash
pip install -r app/requirements.txt
```

**Key packages installed:**
- FastAPI & Uvicorn (web framework)
- PyTorch & TorchVision (deep learning)
- Ultralytics (YOLO)
- SAHI (Sliced inference)
- Pillow (image processing)

#### 2.3 Verify Model Files

Ensure the following files exist:

```
backend/app/Model/
├── classification/
│   └── classification.pth          (CNN model - ~50MB)
└── objectDetectionNoFolds/
    └── YoloNoFolds.pt             (YOLO model - ~100MB)
```

If missing, download from your training source or cloud storage.

### Step 3: Frontend Setup

#### 3.1 Install Node Dependencies

```bash
cd frontend/CS489-Frontend-Website
npm install
```

**Key packages installed:**
- React & React DOM
- Vite (build tool)
- TypeScript
- Tailwind CSS
- Redux Toolkit

#### 3.2 Configure Backend URL

Edit `src/service/modelsAPI.ts`:

```typescript
// Line ~5: Update API endpoint
const API_BASE_URL = "http://localhost:8000";
```

## Running Locally

### Quick Start (Terminal 1 - Backend)

```bash
cd backend
# Activate venv if not already activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Quick Start (Terminal 2 - Frontend)

```bash
cd frontend/CS489-Frontend-Website
npm run dev
```

Expected output:
```
  VITE v7.1.14  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  press h + enter to show help
```

### Access the Application

Open your browser and navigate to:
- **Frontend**: `http://localhost:5173`
- **Backend API Docs**: `http://localhost:8000/docs`
- **Backend ReDoc**: `http://localhost:8000/redoc`

## Testing the Application

### Test 1: Health Check

```bash
# Terminal - Run curl command
curl http://localhost:8000/

# Expected response:
# {"message": "I am alive"}
```

### Test 2: Fire Classification

```bash
# Replace test_image.jpg with your image
curl -X POST "http://localhost:8000/predict" \
  -F "image=@test_image.jpg"

# Expected response:
# {
#   "predicted_class": "fire",
#   "probability_fire": 0.985,
#   "probability_no_fire": 0.015
# }
```

### Test 3: Object Detection

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "image=@test_image.jpg"

# Expected response:
# {
#   "image": "base64-encoded JPEG...",
#   "predicted_boxes": [...]
# }
```

### Test 4: Full UI Testing

1. Open `http://localhost:5173` in browser
2. Click "Upload Image" or drag-drop an image
3. Observe:
   - "Classifying..." status
   - "Localizing..." status
   - Final results with confidence scores and bounding boxes

## Project Structure

```
CS489-Fire-Detection-Model/
├── README.md                          (Project overview)
├── README-LOCAL-SETUP.md             (This file)
├── LICENSE
├── backend/
│   ├── README.md                     (Backend documentation)
│   ├── dockerfile                    (Docker configuration)
│   ├── app/
│   │   ├── main.py                  (FastAPI application)
│   │   ├── requirements.txt          (Python dependencies)
│   │   ├── readMe.md                (API documentation)
│   │   └── Model/
│   │       ├── classification/       (CNN model files)
│   │       ├── objectDetectionNoFolds/  (YOLO model)
│   │       ├── objectDetectionResNet/   (Alternative model)
│   │       └── objectDetectionVanilla/  (Alternative model)
│   └── data/                         (Datasets and additional models)
└── frontend/
    ├── README.md                     (Frontend documentation)
    └── CS489-Frontend-Website/
        ├── package.json              (Node dependencies)
        ├── vite.config.ts            (Vite configuration)
        ├── tsconfig.json             (TypeScript configuration)
        ├── index.html                (HTML entry)
        ├── src/
        │   ├── App.tsx               (Main React component)
        │   ├── main.tsx              (Entry point)
        │   ├── features/             (Feature components)
        │   ├── service/              (API integration)
        │   ├── store/                (Redux store)
        │   ├── types/                (TypeScript types)
        │   └── utils/                (Utilities)
        └── public/                   (Static assets)
```

## Detailed Component Information

### Backend Components

**Main Files:**
- `app/main.py`: FastAPI application with 3 endpoints
- `app/Model/classification/model.py`: CNN architecture
- `app/Model/classification/classification.pth`: Trained CNN weights
- `app/Model/objectDetectionNoFolds/YoloNoFolds.pt`: Trained YOLO weights

**Key Models:**
- **CNN**: ResNet-based binary classifier (fire/no_fire)
- **YOLO**: Object detector for fire and smoke localization
- **SAHI**: Inference engine for improved small object detection

### Frontend Components

**Key Files:**
- `App.tsx`: Main application component
- `service/modelsAPI.ts`: Redux queries for API endpoints
- `features/landing/components/uploadImage.tsx`: Image upload interface
- `features/landing/components/results.tsx`: Results display
- `store/store.ts`: Redux store configuration

**UI Features:**
- Image upload with drag-drop
- Progressive processing indicator
- Real-time result visualization
- Responsive design with Tailwind CSS

