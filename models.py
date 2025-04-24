# --- models.py ---

# Data handling and preprocessing
import pandas as pd

# TF-IDF vectorizer for text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Classifiers
from sklearn.naive_bayes import MultinomialNB  # (Not used but imported if needed later)
from sklearn.linear_model import LogisticRegression  # Classifier used

# Create a pipeline to combine TF-IDF and classifier
from sklearn.pipeline import Pipeline

# For saving/loading models
import joblib

# Train/test split for model evaluation
from sklearn.model_selection import train_test_split

# Metrics to evaluate performance
from sklearn.metrics import accuracy_score, classification_report

# File system utilities
import os

# Path to store the trained model
MODEL_PATH = "classifier.pkl"

# Train the email classification model
def train_model(csv_path: str):
    # Load the CSV dataset
    df = pd.read_csv(csv_path)
    X = df["email"]  # Input feature: email text
    y = df["type"]   # Target label: email category/type

    # Split the data into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a machine learning pipeline: TF-IDF vectorizer + Logistic Regression
    clf = Pipeline([
        ("tfidf", TfidfVectorizer()),  # Convert email text to numerical vectors
        ("clf", LogisticRegression(max_iter=1000))  # Train a logistic regression classifier
    ])
    
    # Train the model on the training set
    clf.fit(X_train, y_train)

    # Save the trained model to disk
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print("Model trained and saved.")

    # Evaluate the model using the test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Load the trained model from file
def load_model():
    return joblib.load(MODEL_PATH)

# Predict the category for a given email text
def predict_category(text: str, model) -> str:
    return model.predict([text])[0]
