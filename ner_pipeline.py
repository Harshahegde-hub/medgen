from transformers import pipeline

class MedicalNER:
    def __init__(self):
        self.ner = pipeline(
            "ner",
            model="d4data/biomedical-ner-all",
            aggregation_strategy="simple",
            device=-1
        )

    def extract(self, text):
        try:
            results = self.ner(text)
        except Exception:
            return []

        entities = []
        for r in results:
            entities.append({
                "text": r["word"],
                "label": r["entity_group"],
                "score": float(r["score"])
            })
        return entities
