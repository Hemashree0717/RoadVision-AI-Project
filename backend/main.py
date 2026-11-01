from pathlib import Path
from uuid import uuid4
import shutil

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from database import init_db, save_result, list_results
from model import RoadUNet
from satellite_validator import validate_satellite_image


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"

MODEL_PATH = BASE / "models" / "road_unet.keras"

UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)
MODEL_PATH.parent.mkdir(exist_ok=True)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Satellite Road Extraction API"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE
# ============================================================

init_db()


# ============================================================
# LOAD U-NET MODEL
# ============================================================

model = (
    RoadUNet.load(MODEL_PATH)
    if MODEL_PATH.exists()
    else None
)


# ============================================================
# SATELLITE IMAGE VALIDATION
# ============================================================




# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# ============================================================
# ROAD EXTRACTION
# ============================================================

@app.post("/api/extract")
async def extract_road(
    file: UploadFile = File(...)
):

    global model

    # --------------------------------------------------------
    # CHECK FILE TYPE
    # --------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a valid satellite image "
                "(JPG, PNG or TIFF)."
            )
        )

    # --------------------------------------------------------
    # CREATE JOB ID
    # --------------------------------------------------------

    job_id = str(
        uuid4()
    )

    original_name = Path(
        file.filename or "satellite.png"
    ).name

    input_path = (
        UPLOADS /
        f"{job_id}_{original_name}"
    )

    output_path = (
        OUTPUTS /
        f"{job_id}_road_mask.png"
    )

    # --------------------------------------------------------
    # SAVE UPLOADED IMAGE
    # --------------------------------------------------------

    with input_path.open("wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        input_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "AI model is not loaded. "
                "Place road_unet.keras inside "
                "backend/models/."
            )
        )

    # ========================================================
    # PROCESS IMAGE
    # ========================================================

    try:

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(
            input_path
        ).convert("RGB")

        rgb = np.array(
            image
        )

        # ----------------------------------------------------
        # SATELLITE IMAGE VALIDATION
        # ----------------------------------------------------

        is_satellite, message = (
            validate_satellite_image(rgb)
        )

        if not is_satellite:

            # Delete rejected upload.
            input_path.unlink(
                missing_ok=True
            )

            raise HTTPException(
                status_code=400,
                detail=message
            )

        # ----------------------------------------------------
        # U-NET ROAD EXTRACTION
        # ----------------------------------------------------

        mask = model.predict(
            rgb
        )

        # ----------------------------------------------------
        # SMALL ROAD GAP CONNECTION
        # ----------------------------------------------------

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=1
        )

        # ----------------------------------------------------
        # FINAL BINARY MASK
        # ----------------------------------------------------

        mask = cv2.threshold(
            mask,
            127,
            255,
            cv2.THRESH_BINARY
        )[1]

        # ----------------------------------------------------
        # SAVE MASK
        # ----------------------------------------------------

        Image.fromarray(
            mask
        ).save(
            output_path
        )

        # ----------------------------------------------------
        # SAVE RESULT IN DATABASE
        # ----------------------------------------------------

        save_result(
            job_id=job_id,
            original_name=original_name,
            input_path=str(input_path),
            output_path=str(output_path)
        )

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return {
            "id": job_id,
            "filename": original_name,
            "mask_url": (
                f"/api/results/{job_id}/mask"
            ),
            "original_url": (
                f"/api/results/{job_id}/original"
            )
        }

    # ========================================================
    # EXPECTED VALIDATION ERROR
    # ========================================================

    except HTTPException:
        raise

    # ========================================================
    # OTHER PROCESSING ERROR
    # ========================================================

    except Exception as exc:

        if input_path.exists():

            input_path.unlink(
                missing_ok=True
            )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Image processing failed: {exc}"
            )
        )


# ============================================================
# HISTORY
# ============================================================

@app.get("/api/results")
def history():

    return {
        "results": list_results()
    }


# ============================================================
# GET ROAD MASK
# ============================================================

@app.get(
    "/api/results/{job_id}/mask"
)
def get_mask(
    job_id: str
):

    row = next(
        (
            r
            for r in list_results()
            if r["id"] == job_id
        ),
        None
    )

    if not row:

        raise HTTPException(
            status_code=404,
            detail="Result not found."
        )

    path = Path(
        row["output_path"]
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Mask file not found."
        )

    return FileResponse(
        path,
        media_type="image/png",
        filename=(
            f"{job_id}_road_mask.png"
        )
    )


# ============================================================
# GET ORIGINAL IMAGE
# ============================================================

@app.get(
    "/api/results/{job_id}/original"
)
def get_original(
    job_id: str
):

    row = next(
        (
            r
            for r in list_results()
            if r["id"] == job_id
        ),
        None
    )

    if not row:

        raise HTTPException(
            status_code=404,
            detail="Result not found."
        )

    path = Path(
        row["input_path"]
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Original file not found."
        )

    return FileResponse(
        path,
        filename=row["original_name"]
    )


# ============================================================
# DOWNLOAD MASK
# ============================================================

@app.get(
    "/api/results/{job_id}/download"
)
def download_mask(
    job_id: str
):

    row = next(
        (
            r
            for r in list_results()
            if r["id"] == job_id
        ),
        None
    )

    if not row:

        raise HTTPException(
            status_code=404,
            detail="Result not found."
        )

    path = Path(
        row["output_path"]
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Mask file not found."
        )

    return FileResponse(
        path,
        media_type="image/png",
        filename=(
            f"road_extraction_{job_id}.png"
        ),
        headers={
            "Content-Disposition":
            (
                'attachment; '
                f'filename="road_extraction_{job_id}.png"'
            )
        }
    )