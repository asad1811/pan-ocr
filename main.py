import json, time, re, unicodedata
from pathlib import Path
import pdfplumber, pandas as pd
from tqdm import tqdm
from huggingface_hub import InferenceClient

#Configurable Parameters 
PDF_FILE = "PDF for Python LLM.pdf"
CONFIG_FILE = "config.json"
OUTPUT_CSV = "mistral.csv"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

MAX_NEW_TOKENS = 1024
RETRIES = 2
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 400

# PROMPTS 
OCR_CLEAN_PROMPT = """
You are an OCR correction model.
The text below was extracted from a scanned PDF and may contain OCR errors such as:
- Broken words (A U T O L I T E)
- Missing or extra spaces
- Line breaks splitting text

Your job: Correct and normalize it to readable text.
Do not summarize or remove information. Return only corrected text.

OCR text:
---
{text}
---
"""

EXTRACTION_PROMPT = """
You are a structured information extraction model.

Some PAN-like tokens are pre-highlighted using [PAN:XXXXX].
Your task:
1. Confirm valid PANs (5 letters + 4 digits + 1 letter)
2. Link each to its entity (person or organization)
3. Return only JSON in this schema:

{{
  "relations": [
    {{"pan": "ABCDE1234F", "entity": "Mr. Agarwal", "confidence": "high"}},
    {{"pan": "AAECA1487G", "entity": "Autolite Agencies Pvt. Ltd.", "confidence": "low"}}
  ]
}}

Rules:
- Use "low" confidence if uncertain.
- Include all possible pairs.
- Do NOT include explanations.

Text:
---
{text}
---
"""
# HELPERS FUNCTIONS
def load_token(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    token = cfg.get("HF_API_TOKEN", "").strip()
    if not token.startswith("hf_"):
        raise ValueError("Invalid Hugging Face token.")
    print("Token loaded:", token[:10] + "...")
    return token

def read_pdf_text(pdf_path):
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            text_parts.append(normalize_text(raw))
    return "\n".join(text_parts)

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)
    text = re.sub(r"-\s*\n\s*", "-", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def highlight_pan_candidates(text):
    """Highlight potential PAN-like tokens for LLM attention."""
    pan_pattern = re.compile(r"\b[A-Z]{3,5}\s?[A-Z]?\d{4}\s?[A-Z]\b")
    return pan_pattern.sub(lambda m: f"[PAN:{m.group(0).replace(' ', '')}]", text)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    step = size - overlap
    return [text[i:i+size] for i in range(0, len(text), step)]

def call_llm(client, prompt):
    for attempt in range(1, RETRIES + 1):
        try:
            resp = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_NEW_TOKENS,
            )
            return resp.choices[0].message["content"]
        except Exception as e:
            print(f"Retry {attempt}/{RETRIES}: {e}")
            time.sleep(2**attempt)
    return ""

def safe_parse_json(text, client=None):
    """Try to extract and repair JSON from LLM output."""
    s, e = text.find("{"), text.rfind("}") + 1
    if s == -1 or e <= s:
        return {"relations": []}

    snippet = text[s:e]
    try:
        return json.loads(snippet)
    except Exception:
        cleaned = (
            snippet.replace("'", '"')
            .replace("\n", " ")
            .replace(",}", "}")
            .replace(",]", "]")
            .replace(" ,", ",")
            .replace("  ", " ")
        )
        try:
            return json.loads(cleaned)
        except Exception:
            pass

    # LLM self-repair attempt
    if client:
        fix_prompt = f"""
        The following JSON is invalid. Fix it to valid JSON with schema:
        {{ "relations": [{{"pan":"ABCDE1234F","entity":"Mr. Agarwal","confidence":"high"}}, ...] }}
        Return only JSON.
        ---
        {snippet[:1500]}
        ---
        """
        try:
            resp = client.chat_completion(
                messages=[{"role": "user", "content": fix_prompt}],
                max_tokens=MAX_NEW_TOKENS,
            )
            fixed = resp.choices[0].message["content"]
            s2, e2 = fixed.find("{"), fixed.rfind("}") + 1
            if s2 != -1 and e2 > s2:
                return json.loads(fixed[s2:e2])
        except Exception as e:
            print("JSON self-repair failed:", e)
    return {"relations": []}


# MAIN PIPELINE
def main():
    if not Path(PDF_FILE).exists():
        print("PDF not found:", PDF_FILE)
        return

    token = load_token(CONFIG_FILE)
    client = InferenceClient(model=HF_MODEL, token=token)

    print(f"Reading and normalizing PDF: {PDF_FILE}")
    text = read_pdf_text(PDF_FILE)
    chunks = chunk_text(text)

    # PASS 1: OCR Correction 
    print("Pass 1: OCR text correction (Mistral as OCR processor)")
    corrected_chunks = []
    for i, chunk in enumerate(tqdm(chunks, desc="Correcting chunks")):
        prompt = OCR_CLEAN_PROMPT.format(text=chunk)
        corrected = call_llm(client, prompt)
        corrected_chunks.append(corrected)
    corrected_text = "\n".join(corrected_chunks)
    print("OCR correction complete.")

    # PAN Highlighting 
    print("Highlighting PAN candidates")
    highlighted_text = highlight_pan_candidates(corrected_text)
    print("Highlighting complete.")

    # PASS 2: Extraction 
    print("Extracting PAN–Entity relations...")
    chunks = chunk_text(highlighted_text)
    all_relations = []

    for i, chunk in enumerate(tqdm(chunks, desc="Extracting relations")):
        prompt = EXTRACTION_PROMPT.format(text=chunk)
        out = call_llm(client, prompt)
        print(f"\n Chunk {i+1} output (first 200 chars):\n{out[:200]}\n")
        parsed = safe_parse_json(out, client)
        rels = parsed.get("relations", [])
        all_relations.extend(rels)

    if not all_relations:
        print("No relations found.")
        return

    df = pd.DataFrame(all_relations)
    if not {"pan", "entity"}.issubset(df.columns):
        print("Model output missing expected keys.")
        return

    # CLEANUP 
    df["pan"] = df["pan"].astype(str).str.replace(" ", "").str.upper()
    df.insert(1, "Relation", "PAN_Of")
    df.columns = ["Entity (PAN)", "Relation", "Entity (Name)", "Confidence"]

    # Keep valid PANs only
    df = df[df["Entity (PAN)"].str.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$")]

    # SAVE INITIAL OUTPUT -
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Raw output saved: {OUTPUT_CSV} ({len(df)} rows)")

    # DEDUPLICATION 
    df["Entity (Name)"] = df["Entity (Name)"].str.strip().str.upper()
    df["Entity (PAN)"] = df["Entity (PAN)"].str.strip().str.upper()
    df["confidence_rank"] = df["Confidence"].map({"high": 1, "low": 0}).fillna(0)

    df = (
        df.sort_values("confidence_rank", ascending=False)
          .drop_duplicates(subset=["Entity (PAN)", "Entity (Name)"])
          .drop(columns="confidence_rank")
    )

    CLEAN_CSV = OUTPUT_CSV.replace(".csv", "_final.csv")
    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")

    print(f"Cleaned CSV saved → {CLEAN_CSV}")
    print(f"Final unique relations: {len(df)}")
    print(df.head(10))


if __name__ == "__main__":
    main()
