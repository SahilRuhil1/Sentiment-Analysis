import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load pre-trained model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.title("🧠 Sentiment Analysis App")
st.write("Enter a customer review below and get the sentiment prediction.")

user_input = st.text_area("✍️ Enter your review:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized.toarray())[0].capitalize()


        st.subheader("🔍 Prediction:")
        if prediction == 'Positive':
            st.success(f"Sentiment: {prediction} 😊")
        elif prediction == 'Negative':
            st.error(f"Sentiment: {prediction} 😞")
        else:
            st.info(f"Sentiment: {prediction} 😐")
