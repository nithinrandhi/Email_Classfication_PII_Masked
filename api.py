
# --- api.py ---
from models import load_model, predict_category
from utils import mask_pii, demask_pii

model = load_model()

def classify_email(email_body: str):
    masked_email, entities = mask_pii(email_body)
    category = predict_category(masked_email, model)
    demasked_email = demask_pii(masked_email, entities)

    return {
        "input_email_body": email_body,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }

