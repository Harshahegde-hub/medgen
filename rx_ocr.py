import cv2
import pytesseract
import re
import json
import os

# ==========================
# LOAD JSON FILES
# ==========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

DRUG_DATABASE = load_json("drug_database.json")
BRAND_GENERIC_MAP = load_json("brand_generic.json")

# ==========================
# PREPROCESSING (FIXED)
# ==========================

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not found")

    # Light denoising ONLY
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img

# ==========================
# OCR (CORRECT MODE)
# ==========================

def ocr_image(image):
    config = r"--oem 1 --psm 6 -l eng"
    return pytesseract.image_to_string(image, config=config)

# ==========================
# TEXT CLEANING
# ==========================

def clean_medical_text(text):
    text = text.lower()

    corrections = {
        "rn": "mg",
        "mq": "mg",
        "mgq": "mg",
        "cys": "days",
        " x ": " ",
        "|": "1"
    }

    for k, v in corrections.items():
        text = text.replace(k, v)

    # Remove tab/cap
    text = re.sub(r"\b(tab|cap)\b", "", text)

    # Brand → Generic
    for brand, generic in BRAND_GENERIC_MAP.items():
        text = re.sub(rf"\b{brand}\b", generic, text)

    text = re.sub(r"[^a-z0-9.\n /-]", "", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()

# ==========================
# REGEX
# ==========================

DOSE_PATTERN = r"\b\d+\s?(mg|ml|mcg|g)\b"
FREQ_PATTERN = r"\b(od|bd|tid|qid)\b"
DURATION_PATTERN = r"\b\d+\s?(day|days)\b"

# ==========================
# EXTRACTION (LINE-WISE)
# ==========================

def extract_drugs(text):
    results = []

    for line in text.split("\n"):
        for drug, info in DRUG_DATABASE.items():
            if re.search(rf"\b{re.escape(drug)}\b", line):

                results.append({
                    "drug": drug,
                    "class": info["class"],
                    "dose": re.findall(DOSE_PATTERN, line)[0],
                    "frequency": re.findall(FREQ_PATTERN, line)[0],
                    "duration": re.findall(DURATION_PATTERN, line)[0]
                    if re.findall(DURATION_PATTERN, line) else None
                })

    return results

# ==========================
# PIPELINE
# ==========================

def prescription_image_to_text(image_path):
    img = preprocess_image(image_path)
    raw_text = ocr_image(img)
    clean_text = clean_medical_text(raw_text)
    drugs = extract_drugs(clean_text)

    return {
        "text": clean_text,
        "drugs": drugs
    }
