
# RoadVision AI — Satellite Road Network Extraction

## 1. Project structure
backend/
  main.py
  model.py
  train.py
  database.py
  requirements.txt
  dataset/images/
  dataset/masks/
  models/road_unet.keras

frontend/
  package.json
  index.html
  src/main.jsx
  src/style.css

## 2. Backend setup
Windows:
    cd backend
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    uvicorn main:app --reload

API:
    http://127.0.0.1:8000
Swagger:
    http://127.0.0.1:8000/docs

## 3. Train the CNN/U-Net
Put paired images and masks in:
    backend/dataset/images
    backend/dataset/masks

Example:
    images/001.jpg
    masks/001.png

Run:
    cd backend
    python train.py

Then confirm:
    backend/models/road_unet.keras

Restart FastAPI after training.

## 4. Frontend
Open a second terminal:
    cd frontend
    npm install
    npm run dev

Open:
    http://localhost:5173

## 5. Important small-road design choices
This implementation uses:
- U-Net encoder/decoder CNN architecture
- Dice loss + road-weighted binary cross entropy
- lower inference threshold (0.35) to preserve faint/narrow roads
- 512x512 sliding-window inference with overlap
- no aggressive morphological opening
- nearest-neighbor mask resizing for training labels

These improve small-road retention, but no architecture can guarantee every road is detected. Final quality depends strongly on the dataset, image resolution, annotation quality, and training strategy.

## 6. Recommended dataset
For a major project, use a road-segmentation dataset such as SpaceNet or DeepGlobe, or your own labeled satellite images. Keep train/validation/test areas geographically separated to avoid leakage.

## 7. Production notes
For large images, tile inference is already implemented. For deployment, add authentication, file-size limits, rate limiting, object storage, and HTTPS.
