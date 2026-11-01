
from pathlib import Path
from uuid import uuid4
import shutil

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from database import init_db, save_result, list_results
from model import RoadUNet

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
MODEL_PATH = BASE / "models" / "road_unet.keras"

UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)
(MODEL_PATH.parent).mkdir(exist_ok=True)

app = FastAPI(title="Satellite Road Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
model = RoadUNet.load(MODEL_PATH) if MODEL_PATH.exists() else None

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/api/extract")
async def extract_road(file: UploadFile = File(...)):
    global model

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload a valid image file.")

    job_id = str(uuid4())
    original_name = Path(file.filename or "satellite.png").name
    input_path = UPLOADS / f"{job_id}_{original_name}"
    output_path = OUTPUTS / f"{job_id}_road_mask.png"

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if model is None:
        raise HTTPException(
            503,
            "AI model is not loaded. Train the model first and place road_unet.keras in backend/models/."
        )

    try:
        image = Image.open(input_path).convert("RGB")
        rgb = np.array(image)
        mask = model.predict_mask(rgb)

        # Preserve tiny connected roads and clean isolated noise.
        # Do not use aggressive erosion/opening because it can delete narrow roads.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        Image.fromarray(mask).save(output_path)

        save_result(
            job_id=job_id,
            original_name=original_name,
            input_path=str(input_path),
            output_path=str(output_path),
        )

        return {
            "id": job_id,
            "filename": original_name,
            "mask_url": f"/api/results/{job_id}/mask",
            "original_url": f"/api/results/{job_id}/original",
        }
    except Exception as exc:
        if input_path.exists():
            input_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Image processing failed: {exc}")

@app.get("/api/results")
def history():
    return {"results": list_results()}

@app.get("/api/results/{job_id}/mask")
def get_mask(job_id: str):
    row = next((r for r in list_results() if r["id"] == job_id), None)
    if not row:
        raise HTTPException(404, "Result not found.")
    path = Path(row["output_path"])
    if not path.exists():
        raise HTTPException(404, "Mask file not found.")
    return FileResponse(path, media_type="image/png", filename=f"{job_id}_road_mask.png")

@app.get("/api/results/{job_id}/original")
def get_original(job_id: str):
    row = next((r for r in list_results() if r["id"] == job_id), None)
    if not row:
        raise HTTPException(404, "Result not found.")
    path = Path(row["input_path"])
    if not path.exists():
        raise HTTPException(404, "Original file not found.")
    return FileResponse(path, filename=row["original_name"])

@app.get("/api/results/{job_id}/download")
def download_mask(job_id: str):
    row = next((r for r in list_results() if r["id"] == job_id), None)
    if not row:
        raise HTTPException(404, "Result not found.")
    path = Path(row["output_path"])
    if not path.exists():
        raise HTTPException(404, "Mask file not found.")
    return FileResponse(
        path,
        media_type="image/png",
        filename=f"road_extraction_{job_id}.png",
        headers={"Content-Disposition": f'attachment; filename="road_extraction_{job_id}.png"'}
    )
