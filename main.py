import sys
import re
from ocr import extract_text
from ner_pipeline import MedicalNER
from matcher import DrugMatcher

def normalize(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9 mg]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

if len(sys.argv) < 2:
    print("Usage: python main.py <prescription_image>")
    sys.exit(1)

image_path = sys.argv[1]

print(f"\n📷 Processing image: {image_path}")

raw_text = extract_text(image_path)

print("\n--- RAW OCR TEXT ---")
print(raw_text)

text = normalize(raw_text)

ner = MedicalNER()
matcher = DrugMatcher()

entities = ner.extract(text)

print("\n--- EXTRACTED MEDICAL ENTITIES ---")
for e in entities:
    print(e)

print("\n--- DRUG MATCHING ---")

matched = False

# 1️⃣ Join medication tokens (handles paraceta + mol)
med_tokens = [e["text"] for e in entities if "med" in e["label"].lower()]
joined = "".join(med_tokens)

if joined:
    result = matcher.match(joined)
    if result:
        print(result)
        matched = True

# 2️⃣ Try individual medication tokens
if not matched:
    for token in med_tokens:
        result = matcher.match(token)
        if result:
            print(result)
            matched = True
            break

# 3️⃣ Final fallback: full OCR text
if not matched:
    print("⚠️ Fallback matching on full text")
    result = matcher.match(text)
    if result:
        print(result)
    else:
        print("❌ No known drug detected")
