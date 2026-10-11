import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(
    "model/food_reviews.csv",
    on_bad_lines="warn"
)

print("Dataset loaded successfully!")
print(df.head())


# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Clean reviews
df["clean_review"] = df["review"].apply(clean_text)


# Create sentiment
def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"


df["sentiment"] = df["rating"].apply(get_sentiment)

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())


# Input and output
X = df["clean_review"]
y = df["sentiment"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)


# Prediction
y_pred = model.predict(X_test_tfidf)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save model
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")