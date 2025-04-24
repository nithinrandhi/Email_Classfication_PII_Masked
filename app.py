# Import FastAPI framework to create the API
from fastapi import FastAPI, Body

# Import Pydantic BaseModel to define data validation schema
from pydantic import BaseModel

# Import the email classification logic from api.py
from api import classify_email

# Used to redirect root requests to the documentation page
from fastapi.responses import RedirectResponse

# Initialize the FastAPI application
app = FastAPI()

# Define the structure of the expected request body using Pydantic
class EmailRequest(BaseModel):
    email_body: str  # Field to hold the raw email text

# Redirect root path '/' to the Swagger UI at '/docs'
@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

# Define the classification endpoint
@app.post("/classify")
def classify_endpoint(
    # Use Body(...) to embed example input into the Swagger UI
    request: EmailRequest = Body(
        example={
            "email_body": "Paste the entire email content here in a single line without line breaks."
        }
    )
):
    # Call the email classification function and return the result
    return classify_email(request.email_body)
