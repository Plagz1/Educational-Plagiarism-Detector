from pathlib import Path
import PyPDF2


def load_text(file_path: Path) -> str:
    if file_path.suffix == ".txt":
        return file_path.read_text(encoding="utf-8")

    if file_path.suffix == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text

    raise ValueError("Unsupported file format")
