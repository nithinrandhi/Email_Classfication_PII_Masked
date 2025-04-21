# --- app.py ---
from fastapi import FastAPI
from pydantic import BaseModel, Field
from api import classify_email
from fastapi.responses import RedirectResponse

app = FastAPI()

class EmailRequest(BaseModel):
    email_body: str = Field(
        ..., 
        description="Paste your full email content here."
    )

@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post("/classify")
def classify_endpoint(request: EmailRequest):
    return classify_email(request.email_body)