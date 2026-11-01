import torch
import numpy as np

from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# LOAD CLIP MODEL
# ============================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading satellite image checker...")

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

classifier = CLIPModel.from_pretrained(
    MODEL_NAME
)

classifier.eval()

print("Satellite image checker loaded.")


# ============================================================
# TEXT DESCRIPTIONS
# ============================================================

SATELLITE_TEXTS = [
    "a satellite image",
    "a satellite view of the earth",
    "a satellite image of land",
    "a satellite map image",
    "an aerial satellite image",
    "a top-down satellite image",
    "a remote sensing image"
]


NON_SATELLITE_TEXTS = [
    "a normal photograph",
    "a camera photograph",
    "a person photograph",
    "a selfie",
    "a document",
    "a scanned document",
    "a diagram",
    "a drawing",
    "a chart",
    "a screenshot of a computer application"
]


# ============================================================
# SATELLITE IMAGE VALIDATION
# ============================================================

def validate_satellite_image(image):

    try:

        # ----------------------------------------------------
        # CONVERT IMAGE TO PIL
        # ----------------------------------------------------

        if isinstance(image, np.ndarray):

            if image.ndim != 3:

                return False, (
                    "Invalid image. "
                    "Please upload a satellite image."
                )

            if image.shape[2] != 3:

                return False, (
                    "Invalid image. "
                    "Please upload a satellite image."
                )

            pil_image = Image.fromarray(
                image.astype(np.uint8)
            )

        else:

            pil_image = image.convert("RGB")


        # ----------------------------------------------------
        # TEXT PROMPTS
        # ----------------------------------------------------

        all_texts = (
            SATELLITE_TEXTS
            + NON_SATELLITE_TEXTS
        )


        # ----------------------------------------------------
        # PREPARE IMAGE AND TEXT
        # ----------------------------------------------------

        inputs = processor(
            text=all_texts,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = classifier(
                **inputs
            )


        # ----------------------------------------------------
        # GET SCORES
        # ----------------------------------------------------

        logits = outputs.logits_per_image[0]

        probabilities = torch.softmax(
            logits,
            dim=0
        )


        # ----------------------------------------------------
        # SATELLITE SCORES
        # ----------------------------------------------------

        satellite_scores = probabilities[
            :len(SATELLITE_TEXTS)
        ]


        non_satellite_scores = probabilities[
            len(SATELLITE_TEXTS):
        ]


        satellite_score = float(
            torch.mean(
                satellite_scores
            ).item()
        )


        non_satellite_score = float(
            torch.mean(
                non_satellite_scores
            ).item()
        )


        # ----------------------------------------------------
        # PRINT SCORES
        # ----------------------------------------------------

        print(
            "Satellite score:",
            round(
                satellite_score,
                4
            )
        )

        print(
            "Non-satellite score:",
            round(
                non_satellite_score,
                4
            )
        )


        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------

        SATELLITE_THRESHOLD = 0.08


        if (
            satellite_score > non_satellite_score
            and
            satellite_score >= SATELLITE_THRESHOLD
        ):

            return True, ""


        # ----------------------------------------------------
        # REJECT
        # ----------------------------------------------------

        return False, (
            "Invalid image. "
            "Only satellite images are accepted. "
            "Please upload a satellite image."
        )


    except Exception as exc:

        print(
            "Satellite validation error:",
            exc
        )

        return False, (
            "Unable to verify the uploaded image. "
            "Please upload a valid satellite image."
        )