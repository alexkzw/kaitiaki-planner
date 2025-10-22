from pathlib import Path
import json, re

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "corpus_raw"
OUT = ROOT / "data" / "corpus.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

docs = []
for path in sorted(RAW_DIR.glob("*.txt")):
    fname = path.name
    lang = "mi" if fname.startswith("mi_") else "en"
    title = re.sub(r"\.txt$", "", fname).replace("_", " ")
    doc_id = fname[:-4]  # strip .txt
    text = path.read_text(encoding="utf-8").strip()
    if len(text) > 3000:
        text = text[:3000]
    docs.append({"id": doc_id, "lang": lang, "title": title, "text": text})

OUT.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(docs)} docs to {OUT}")
