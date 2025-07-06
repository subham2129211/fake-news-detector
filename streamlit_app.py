import streamlit as st
import joblib
import re
import string

# ---------------------
# Load the trained model
# ---------------------
model = joblib.load("fake_news_model_v2.pkl")

# ---------------------
# Utility functions
# ---------------------

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def trim_text(text, max_len=500):
    return text[:max_len]

def predict_news(title, body, model):
    combined = title + " " + body
    trimmed = trim_text(combined)
    cleaned = clean_text(trimmed)
    pred = model.predict([cleaned])[0]
    prob = model.predict_proba([cleaned])[0]
    conf = round(max(prob) * 100, 2)
    return pred, conf

# ---------------------
# Streamlit UI
# ---------------------

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("Fake News Detection Web App")
st.markdown("A simple ML-powered tool to classify news articles as **FAKE** or **REAL**.")

st.write("---")

# Input fields
title_input = st.text_input("📝 News Title", placeholder="e.g. Elon Musk plans to build city on Mars")
text_input = st.text_area("📄 News Body", height=200, placeholder="Enter the full news article content here...")

if st.button("Detect"):
    if not title_input or not text_input:
        st.warning("Please enter both title and text to proceed.")
    else:
        label, confidence = predict_news(title_input, text_input, model)

        if label == "FAKE":
            st.error(f" Prediction: **FAKE News** ({confidence}% confidence)")
        else:
            st.success(f" Prediction: **REAL News** ({confidence}% confidence)")

        st.markdown("---")
        st.subheader(" Model Info")
        st.markdown("- Model: Logistic Regression + TF-IDF")
        st.markdown("- Input: Combined Title + Text (trimmed to 500 chars)")
        st.markdown("- Accuracy: ~98.5% on test data")
