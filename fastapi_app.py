from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import base64
import cv2

app = FastAPI(title="AttendAI FastAPI")

# ===== Request Model =====
class ImageRequest(BaseModel):
    image: str
    class_id: int

@app.get("/")
def root():
    return {"message": "FastAPI for AttendAI is running"}

# ===== Recognition API =====
@app.post("/api/recognize")
def recognize(data: ImageRequest):

    # Decode base64 image
    img_data = base64.b64decode(data.image.split(",")[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # TODO: Call your recognition logic here

    return {
        "status": "ok",
        "message": "Recognition API working",
        "class_id": data.class_id
    }