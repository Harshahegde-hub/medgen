# ==========================================================
# OCR + NER ENSEMBLE SYSTEM FOR HANDWRITTEN PRESCRIPTIONS
# ==========================================================

import cv2
import pytesseract
import re
import json
import numpy as np
import easyocr
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
import spacy
from transformers import pipeline

# ==========================================================
# LOAD JSON FILES
# ==========================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

DRUG_DATABASE = load_json("drug_database.json")
BRAND_GENERIC_MAP = load_json("brand_generic.json")

# ==========================================================
# INITIALIZE MODELS (LOAD ONCE)
# ==========================================================

easyocr_reader = easyocr.Reader(['en'], gpu=False)
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Clinical + Biomedical NER models
spacy_ner = spacy.load("en_ner_bc5cdr_md")

transformer_ner = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple"
)

# ==========================================================
# IMAGE PREPROCESSING (HANDWRITING FRIENDLY)
# ==========================================================

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found")

    # Gentle denoising (do NOT over-process handwriting)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

# ==========================================================
# OCR MODELS
# ==========================================================

def ocr_tesseract(image):
    config = r"--oem 1 --psm 6 -l eng"
    return pytesseract.image_to_string(image, config=config).strip()


def ocr_easyocr(image_path):
    result = easyocr_reader.readtext(image_path, detail=0)
    return " ".join(result).strip()


def ocr_paddle(image_path):
    result = paddle_ocr.ocr(image_path, cls=True)
    lines = []
    for page in result:
        for line in page:
            lines.append(line[1][0])
    return " ".join(lines).strip()

# ==========================================================
# EMBEDDING-BASED OCR ENSEMBLE
# ==========================================================

def ensemble_ocr(image, image_path):
    outputs = [
        ocr_tesseract(image),
        ocr_easyocr(image_path),
        ocr_paddle(image_path)
    ]

    outputs = [t for t in outputs if len(t) > 0]
    if len(outputs) == 1:
        return outputs[0]

    embeddings = embed_model.encode(outputs)
    similarity_matrix = np.inner(embeddings, embeddings)
    scores = similarity_matrix.sum(axis=1)

    return outputs[int(np.argmax(scores))]

# ==========================================================
# MEDICAL TEXT CLEANING
# ==========================================================

def clean_medical_text(text):
    text = text.lower()

    corrections = {
        "rn": "mg",
        "mq": "mg",
        "mgq": "mg",
        "cys": "days",
        "|": "1"
    }

    for k, v in corrections.items():
        text = text.replace(k, v)

    text = re.sub(r"\b(tab|cap)\b", "", text)

    # Brand → Generic
    for brand, generic in BRAND_GENERIC_MAP.items():
        text = re.sub(rf"\b{brand}\b", generic, text)

    text = re.sub(r"[^a-z0-9.\n /-]", "", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()

# ==========================================================
# NER MODELS
# ==========================================================

def spacy_ner_extract(text):
    doc = spacy_ner(text)
    return {ent.text.lower() for ent in doc.ents if ent.label_ == "CHEMICAL"}


def transformer_ner_extract(text):
    entities = transformer_ner(text)
    return {
        ent["word"].lower()
        for ent in entities
        if ent["entity_group"] in ["CHEMICAL", "DRUG"]
    }


def dictionary_match(text):
    found = set()
    for drug in DRUG_DATABASE.keys():
        if re.search(rf"\b{re.escape(drug)}\b", text):
            found.add(drug)
    return found

# ==========================================================
# NER ENSEMBLE (MAJORITY VOTING)
# ==========================================================

def ner_ensemble(text):
    s1 = spacy_ner_extract(text)
    s2 = transformer_ner_extract(text)
    s3 = dictionary_match(text)

    final_drugs = set()
    for drug in s1 | s2 | s3:
        votes = sum([
            drug in s1,
            drug in s2,
            drug in s3
        ])
        if votes >= 2:
            final_drugs.add(drug)

    return final_drugs

# ==========================================================
# ATTRIBUTE EXTRACTION (DOSE / FREQ / DURATION)
# ==========================================================

def normalize_frequency(freq):
    mapping = {
        "od": "once daily",
        "bd": "twice daily",
        "tid": "three times daily",
        "qid": "four times daily"
    }
    return mapping.get(freq, freq)


def extract_attributes(line):
    dose = re.findall(r"\b\d+\s?(mg|ml|mcg|g)\b", line)
    freq = re.findall(r"\b(od|bd|tid|qid)\b", line)
    duration = re.findall(r"\b\d+\s?(day|days)\b", line)

    return (
        dose[0] if dose else None,
        normalize_frequency(freq[0]) if freq else None,
        duration[0] if duration else None
    )

# ==========================================================
# FINAL DRUG EXTRACTION (NER-BASED)
# ==========================================================

def extract_drugs(text):
    results = []
    detected_drugs = ner_ensemble(text)

    for drug in detected_drugs:
        for line in text.split("\n"):
            if drug in line:
                dose, freq, duration = extract_attributes(line)
                info = DRUG_DATABASE.get(drug, {})

                results.append({
                    "drug": drug,
                    "class": info.get("class"),
                    "dose": dose,
                    "frequency": freq,
                    "duration": duration
                })

    return results

# ==========================================================
# FULL PIPELINE
# ==========================================================

def prescription_image_to_text(image_path):
    image = preprocess_image(image_path)
    raw_text = ensemble_ocr(image, image_path)
    clean_text = clean_medical_text(raw_text)
    drugs = extract_drugs(clean_text)

    return {
        "extracted_text": clean_text,
        "identified_drugs": drugs
    }

# ==========================================================
# TEST RUN
# ==========================================================

if __name__ == "__main__":
    image_path = "test_prescription.jpg"  # change this
    output = prescription_image_to_text(image_path)
    print(json.dumps(output, indent=2))
