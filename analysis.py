import joblib
import re

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def analyze_review(review):

    cleaned_review = clean_text(review)

    vector = vectorizer.transform([cleaned_review])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)

    confidence = probabilities.max()

    return prediction, confidence