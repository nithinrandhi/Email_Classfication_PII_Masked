# 📧 Email Classifier & PII Masking API

A FastAPI-based application that classifies emails into categories and masks personally identifiable information (PII) such as names, emails, phone numbers, credit card details, and more.

---

## 🚀 Features
- 🔒 PII masking using spaCy, regex, and custom rules
- 🧠 Email classification using Logistic Regression (TF-IDF)
- 🌐 Multilingual name detection (English & German)
- 🔎 Language detection with `langdetect`
- 📦 API deployment using FastAPI + Docker
- ☁️ Deployable on Hugging Face Spaces or any container platform

---

## 🧰 Requirements
- Python 3.10+
- pip
- Docker (for deployment)

### 🔧 Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

---

## 🏋️‍♂️ Training the Model
Make sure your dataset `emails.csv` is in the `data/` folder and contains:
- `email_body`: email content
- `pattern`: category label

```bash
python -c "from models import train_model; train_model('data/emails.csv')"
```
This saves the trained model to `saved_models/classifier.pkl`

---

## 🚀 Running the App Locally
```bash
uvicorn app:app --reload
```
Go to [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

---

## 🧪 API Endpoints

### POST `/classify`
```json
{
  "email_body": "Subject: Support Needed\nMy name is John Doe. Contact me at john@example.com."
}
```

### ✅ Response:
```json
{
  "input_email_body": "...",
  "list_of_masked_entities": [...],
  "masked_email": "...",
  "category_of_the_email": "Billing Issue"
}
```

---

## 🐳 Docker Deployment

### Build the Docker Image
```bash
docker build -t email-classifier .
```

### Run the Container
```bash
docker run -p 7860:7860 email-classifier
```

### Access it at:
[http://localhost:7860/docs](http://localhost:7860/docs)

---

## 🌍 Hugging Face Deployment

### Required Files:
- `Dockerfile`
- `.gitattributes` (if using Git LFS for models)
- `README.md` (with HF config block)

### Top of README:
```markdown
---
title: Email Classifier
emoji: 📧
colorFrom: pink
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---
```

### Push to HF:
```bash
git lfs install
# clone your HF space repo
cd email-classifier
# copy all files, then:
git add .
git commit -m "Initial push"
git push
```



---

## 📌 Project Structure
```
.
├── app.py                # FastAPI app
├── api.py                # API logic
├── models.py             # Training and inference
├── utils.py              # PII masking utilities
├── requirements.txt
├── Dockerfile
├── README.md
├── saved_models/
├── data/
└── .gitattributes
```

---

## 🔮 Future Improvements
- Replace Logistic Regression with BERT or DistilBERT
- Add more languages for name detection
- Create a frontend with Gradio or Streamlit
- Add `/train` API endpoint for retraining

---

## 👨‍💻 Author
**Nithin Randhi**  
**deployement link** https://huggingface.co/spaces/nithinrandhi/email-classifier


---

## 🛡️ License
MIT License

