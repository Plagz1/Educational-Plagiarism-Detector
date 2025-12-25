import json
import time
from pathlib import Path
from itertools import combinations

import pdfplumber

from src.preprocessing import preprocess_text
from src.similarity import (
    cosine_similarity_score,
    lcs_similarity,
    ngram_similarity
)


UPLOADS_DIR = Path("uploads")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_text(file_path: Path) -> str:
    if file_path.suffix == ".txt":
        return file_path.read_text(encoding="utf-8")

    if file_path.suffix == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            return " ".join(page.extract_text() for page in pdf.pages)

    raise ValueError("Unsupported file format")


def run_pipeline():
    files = list(UPLOADS_DIR.glob("*"))
    texts = [load_text(f) for f in files]
    processed = [preprocess_text(t) for t in texts]

    cosine_matrix = cosine_similarity_score(processed)

    pairwise = []
    for (i, a), (j, b) in combinations(enumerate(processed), 2):
        pairwise.append({
            "file_a": files[i].name,
            "file_b": files[j].name,
            "lcs": lcs_similarity(a, b),
            "ngram": ngram_similarity(a, b)
        })

    result = {
        "timestamp": time.time(),
        "files": [f.name for f in files],
        "cosine_matrix": cosine_matrix.tolist(),
        "pairwise": pairwise
    }

    output = RESULTS_DIR / f"result_{int(time.time())}.json"
    output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_pipeline()
