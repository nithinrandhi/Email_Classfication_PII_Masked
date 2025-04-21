
# --- utils.py ---
import spacy
import re
from typing import Tuple, List, Dict
from langdetect import detect

# Load spaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_de = spacy.load("de_core_news_sm")

Entity = Dict[str, object]

PII_PATTERNS = {
    "email": re.compile(r"[\w\.-]+@[\w\.-]+"),
    "phone_number": re.compile(r"\+\d{1,3}(?:-\d{1,5}){2,5}"),
    "dob": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
    "aadhar_num": re.compile(r"\b\d{12}\b"),
    "credit_debit_no": re.compile(r"\b\d{13,}\b"),
    "cvv_no": re.compile(r"(?:CVV[:\s]*)?(\b\d{3}\b)", re.IGNORECASE),
    "expiry_no": re.compile(r"\b\d{2}/\d{2}\b")
}

NAME_PATTERN = re.compile(r"My name is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"

def extract_name(text: str) -> List[Tuple[int, int, str, str]]:
    matches = NAME_PATTERN.finditer(text)
    return [(m.start(1), m.end(1), "full_name", m.group(1)) for m in matches]

def mask_pii(text: str) -> Tuple[str, List[Entity]]:
    masked_text = text
    entities = []
    offset = 0

    lang = detect_language(text)
    doc = nlp_de(text) if lang == "de" else nlp_en(text)

    spacy_entities = [(ent.start_char, ent.end_char, "full_name", ent.text)
                      for ent in doc.ents if ent.label_ == "PERSON"]

    fallback_names = extract_name(text)

    regex_entities = []
    for entity_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            original = match.group(1) if entity_type == "cvv_no" and match.lastindex else match.group()
            regex_entities.append((match.start(), match.end(), entity_type, original))

    all_entities = sorted(spacy_entities + fallback_names + regex_entities, key=lambda x: x[0])

    for start, end, entity_type, original in all_entities:
        start += offset
        end += offset
        replacement = f"[{entity_type}]"

        masked_text = masked_text[:start] + replacement + masked_text[end:]

        entities.append({
            "position": [start, start + len(replacement)],
            "classification": entity_type,
            "entity": original
        })

        offset += len(replacement) - len(original)

    return masked_text, entities

def demask_pii(masked_text: str, entities: List[Entity]) -> str:
    restored_text = masked_text
    offset = 0
    for ent in entities:
        start, end = ent["position"]
        entity_val = ent["entity"]

        restored_text = (
            restored_text[:start + offset] + entity_val + restored_text[end + offset:]
        )
        offset += len(entity_val) - (end - start)

    return restored_text
