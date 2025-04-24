# Import spaCy for NLP and NER
import spacy

# Regular expressions for pattern matching PII
import re

# Typing for better code clarity
from typing import Tuple, List, Dict

# Language detection library
from langdetect import detect

# Load spaCy language models for English and German
nlp_en = spacy.load("en_core_web_sm")
nlp_de = spacy.load("de_core_news_sm")

# Define the structure of a detected entity
Entity = Dict[str, object]

# Define regular expression patterns for various PII types
PII_PATTERNS = {
    "email": re.compile(r"[\w\.-]+@[\w\.-]+"),  # Matches email addresses
    "phone_number": re.compile(r"\+\d{1,3}(?:-\d{1,5}){2,5}"),  # Matches international phone numbers
    "dob": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),  # Matches dates of birth (dd/mm/yyyy)
    "aadhar_num": re.compile(r"\b\d{12}\b"),  # Matches 12-digit Aadhar numbers
    "credit_debit_no": re.compile(r"\b\d{13,}\b"),  # Matches credit/debit card numbers with 13+ digits
    "cvv_no": re.compile(r"(?:CVV[:\s]*)?(\b\d{3}\b)", re.IGNORECASE),  # Matches CVV numbers (3 digits)
    "expiry_no": re.compile(r"\b\d{2}/\d{2}\b")  # Matches expiry dates (MM/YY)
}

# Regex to extract names from the phrase “My name is …”
NAME_PATTERN = re.compile(r"My name is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")

# Detect the language of the input text
def detect_language(text: str) -> str:
    try:
        return detect(text)  # Returns 'en', 'de', etc.
    except:
        return "en"  # Fallback to English on detection failure

# Extract names using the “My name is ...” pattern
def extract_name(text: str) -> List[Tuple[int, int, str, str]]:
    matches = NAME_PATTERN.finditer(text)
    return [(m.start(1), m.end(1), "full_name", m.group(1)) for m in matches]

# Mask all personally identifiable information in the email text
def mask_pii(text: str) -> Tuple[str, List[Entity]]:
    masked_text = text  # Copy of the original text
    entities = []  # Store details of masked entities
    offset = 0  # Track changes in string length after masking

    # Detect the language and choose the appropriate NER model
    lang = detect_language(text)
    doc = nlp_de(text) if lang == "de" else nlp_en(text)

    # Extract names using spaCy if labeled as PERSON
    spacy_entities = [(ent.start_char, ent.end_char, "full_name", ent.text)
                      for ent in doc.ents if ent.label_ == "PERSON"]

    # Extract names using the fallback regex method
    fallback_names = extract_name(text)

    # Detect PII using regex for all defined types
    regex_entities = []
    for entity_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            original = match.group(1) if entity_type == "cvv_no" and match.lastindex else match.group()
            regex_entities.append((match.start(), match.end(), entity_type, original))

    # Combine and sort all detected entities by position
    all_entities = sorted(spacy_entities + fallback_names + regex_entities, key=lambda x: x[0])

    # Replace each detected entity in the text with a placeholder
    for start, end, entity_type, original in all_entities:
        start += offset
        end += offset
        replacement = f"[{entity_type}]"

        masked_text = masked_text[:start] + replacement + masked_text[end:]

        # Record metadata about the replacement
        entities.append({
            "position": [start, start + len(replacement)],
            "classification": entity_type,
            "entity": original
        })

        # Adjust offset due to length difference between original and replacement
        offset += len(replacement) - len(original)

    return masked_text, entities

# Restore original PII entities back into the masked email (if needed)
def demask_pii(masked_text: str, entities: List[Entity]) -> str:
    restored_text = masked_text
    offset = 0

    # Replace placeholders with the original values using recorded positions
    for ent in entities:
        start, end = ent["position"]
        entity_val = ent["entity"]

        restored_text = (
            restored_text[:start + offset] + entity_val + restored_text[end + offset:]
        )
        offset += len(entity_val) - (end - start)

    return restored_text
