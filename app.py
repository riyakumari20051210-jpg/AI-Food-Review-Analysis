import streamlit as st
import pandas as pd
import joblib
import re
import plotly.express as px


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="AI Food Review Analysis",
    page_icon="🍕",
    layout="wide"
)


from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "sentiment_model.pkl"
)

vectorizer = joblib.load(
    BASE_DIR / "tfidf_vectorizer.pkl"
)

# -----------------------------
# Load model
# -----------------------------

model = joblib.load(
    "model/sentiment_model.pkl"
)

vectorizer = joblib.load(
    "model/tfidf_vectorizer.pkl"
)


# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv(
    "model/food_reviews.csv", on_bad_lines="warn")



# -----------------------------
# Title
# -----------------------------

st.title("🍕 AI-Based Food Review Analysis")

st.write(
    "Analyze food reviews using Natural Language "
    "Processing and Machine Learning."
)



total_reviews = len(df)

positive_reviews = len(
    df[df["rating"] >= 4]
)

negative_reviews = len(
    df[df["rating"] <= 2]
)

average_rating = df["rating"].mean()


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Reviews",
    total_reviews
)

col2.metric(
    "Positive Reviews",
    positive_reviews
)

col3.metric(
    "Negative Reviews",
    negative_reviews
)

col4.metric(
    "Average Rating",
    round(average_rating, 2)
)

def get_sentiment(rating):

    if rating >= 4:
        return "Positive"

    elif rating == 3:
        return "Neutral"

    else:
        return "Negative"


df["sentiment"] = df["rating"].apply(
    get_sentiment
)



st.subheader("📊 Sentiment Distribution")

sentiment_count = df["sentiment"].value_counts()

fig = px.pie(
    values=sentiment_count.values,
    names=sentiment_count.index,
    title="Food Review Sentiment"
)

st.plotly_chart(
    fig,
    use_container_width=True
)




st.subheader("⭐ Rating Distribution")

rating_count = (
    df["rating"]
    .value_counts()
    .sort_index()
)

fig = px.bar(
    x=rating_count.index,
    y=rating_count.values,
    labels={
        "x": "Rating",
        "y": "Number of Reviews"
    },
    title="Food Rating Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)





st.subheader("🍔 Product-wise Reviews")

product_count = (
    df["product"]
    .value_counts()
    .reset_index()
)

product_count.columns = [
    "Product",
    "Reviews"
]

fig = px.bar(
    product_count,
    x="Product",
    y="Reviews",
    title="Reviews by Food Product"
)

st.plotly_chart(
    fig,
    use_container_width=True
)






st.subheader("🤖 Analyze New Food Review")

review = st.text_area(
    "Enter your food review:"
)

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning(
            "Please enter a review."
        )

    else:

        cleaned = re.sub(
            r'[^a-zA-Z\s]',
            '',
            review.lower()
        )

        vector = vectorizer.transform(
            [cleaned]
        )

        prediction = model.predict(
            vector
        )[0]

        confidence = (
            model.predict_proba(vector)
            .max()
        )

        st.success(
            f"Sentiment: {prediction}"
        )

        st.info(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

