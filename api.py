# --- api.py ---

# Import model loading and prediction functions from models.py
from models import load_model, predict_category

# Import PII masking and demasking functions from utils.py
from utils import mask_pii, demask_pii

# Load the trained classification model once at startup
model = load_model()

# Function to classify a single email body
def classify_email(email_body: str):
    # Step 1: Mask PII (e.g., names, emails, phone numbers) in the input text
    masked_email, entities = mask_pii(email_body)

    # Step 2: Predict the email category using the masked version
    category = predict_category(masked_email, model)

    # Step 3: Optionally demask the PII to restore the original format (if needed)
    demasked_email = demask_pii(masked_email, entities)

    # Step 4: Return a structured response including raw, masked text and prediction
    return {
        "input_email_body": email_body,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
