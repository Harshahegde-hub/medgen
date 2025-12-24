import json
import re
from sentence_transformers import SentenceTransformer, util

class DrugMatcher:
    def __init__(self, db_path="drug_database.json"):
        with open(db_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)

        self.drugs = list(self.db.keys())
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = self.model.encode(
            self.drugs,
            convert_to_tensor=True
        )

    def _normalize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9]", "", text)
        return text

    def match(self, text, threshold=0.45):
        query = self._normalize(text)
        query_emb = self.model.encode(query, convert_to_tensor=True)

        scores = util.cos_sim(query_emb, self.embeddings)[0]
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()

        if best_score >= threshold:
            drug = self.drugs[best_idx]
            return {
                "matched_drug": drug,
                "similarity": round(best_score, 3),
                "details": self.db[drug]
            }

        return None
