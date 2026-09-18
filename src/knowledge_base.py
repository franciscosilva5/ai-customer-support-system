from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = ROOT / "data" / "knowledge_base"


def load_knowledge_base():
    documents = []

    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        lines = text.splitlines()

        title = (
            lines[0].replace("# ", "").strip()
            if lines[0].startswith("# ")
            else path.stem.replace("-", " ").title()
        )

        documents.append(
            {
                "id": path.stem,
                "title": title,
                "text": text,
                "source": path.name,
            }
        )

    return documents
