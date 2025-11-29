import logging
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import io
import torch
from torchvision import transforms
from app.Model.classification.model import CNNModel
from contextlib import asynccontextmanager
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

CNN_MODEL_PATH = "./app/Model/classification/classification.pth"
YOLO_MODEL_PATH = "./app/Model/objectDetectionNoFolds/YoloNoFolds.pt"


@asynccontextmanager
async def lifespan(app:FastAPI):
    logger = logging.getLogger("uvicorn")
    logger.setLevel(logging.INFO)
    logger.info("Starting up... loading CNN Model")
    cnn_model = CNNModel(num_classes=2)
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=torch.device('cpu')))
    cnn_model.eval()
    app.state.cnn_model = cnn_model
    logger.info("Loading yolo object detection model...")
    app.state.yolo_model = YOLO(YOLO_MODEL_PATH)
    app.state.yolo_sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=YOLO_MODEL_PATH,
        confidence_threshold=0.3,
        device="cpu"

    )
    logger.info("Models loaded successfully!")

    yield
    logger.info("Shutting down... cleaning up models")
    del app.state.cnn_model
    del app.state.yolo_model



app = FastAPI(lifespan=lifespan, title="Fire Object detection backend")

origins = [
    "http://localhost:5173",
    "http://d3ml9honae97f3.cloudfront.net",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "I am alive"}

@app.get("/favicon.ico")
def favicon():
    return {}

@app.post("/predict", status_code=status.HTTP_202_ACCEPTED)
async def predict_fire(image: UploadFile = File(...)):
    try: 
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError: 
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = app.state.cnn_model(input_tensor)
        
        probs_fire = torch.softmax(outputs, dim=1)[0, 1].item() 
        pred_class = torch.argmax(outputs, dim=1).item()  
        return {
            "predicted_class": "fire" if pred_class == 1 else "no_fire",
            "probability_fire": probs_fire,
            "probability_no_fire": 1 - probs_fire,
        }

@app.post("/detect", status_code=status.HTTP_200_OK)
async def detect_fire_in_image(image: UploadFile = File(...)):
    try: 
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes))
    except UnidentifiedImageError: 
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")
    
    # run Yolo model and get results
    results = app.state.yolo_model.predict(img, imgsz=640)
    img_result = results[0]
    annotated_img = img_result.plot()
    result_img = Image.fromarray(annotated_img)
    # convert results to send to frontend
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    boxes = img_result.summary()
    # gather Yolo object metrics
    return {
        "image": img_base64, 
        "predicted_boxes": boxes
    }
    

@app.post("/detect_v2", status_code=status.HTTP_200_OK)
async def detect_fire_sahi(image: UploadFile = File(...)):
    try: 
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError: 
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")
    result = get_sliced_prediction(
            img, 
            app.state.yolo_sahi_model,
            slice_height=128,
            slice_width=128,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2    
        )
    
    allowed_classes = [0]

    result.object_prediction_list = [
        pred for pred in result.object_prediction_list
        if pred.category.id in allowed_classes
    ]

    boxes = []
    for pred in result.object_prediction_list:
        boxes.append({
            "name": pred.category.name,
            "class": pred.category.id,
            "confidence": pred.score.value,
            "box": {
                "x1": pred.bbox.minx,
                "x2": pred.bbox.maxx,
                "y1": pred.bbox.miny,
                "y2": pred.bbox.maxy
            }
        })
    
    buffered = io.BytesIO()
    result.image.save(buffered, format="PNG")
    buffered.seek(0)
    # Convert to Base64 string
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
   
    return {
        "image": img_base64, 
        "predicted_boxes": boxes
    }
    

