import logging
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from PIL import Image, UnidentifiedImageError
import io
import torch
from torchvision import transforms
from app.Model.classification.model import CNNModel
from contextlib import asynccontextmanager
from ultralytics import YOLO

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
    logger.info("Models loaded successfully!")

    yield
    logger.info("Shutting down... cleaning up models")
    del app.state.cnn_model
    del app.state.yolo_model



app = FastAPI(lifespan=lifespan, title="Fire Object detection backend")

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
    results = app.state.yolo_model.predict(img, imgsz=256)
    img_result = results[0]
    annotated_img = img_result.plot()
    result_img = Image.fromarray(annotated_img)
    img_result.show()
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
    


