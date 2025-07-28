import streamlit as st
import joblib
import re
import string

st.set_page_config(page_title="Tweet Fake News Detector", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('tweet_fake_news_model.pkl')

model = load_model()

# Clean tweet function (same as training)
def clean_tweet(text):
    if not text:
        return ''
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Example tweets for autofill
def get_example_tweets():
    return [
        "Breaking: Scientists discover water on Mars!",
        "COVID-19 vaccines contain microchips.",
        "The president announced a new economic plan today.",
        "Aliens have landed in New York City!",
        "@user Did you see the news about the new iPhone? #AppleEvent"
    ]

st.title("🕵️ Tweet Fake News Detector")
st.subheader("Check if a tweet is likely to be fake or real news.")
st.markdown("""
Enter a tweet below, select the actual label (if you know it), and click **Detect**. You can also autofill with example tweets for quick testing.
""")

col1, col2 = st.columns([2, 1])

with col1:
    tweet_input = st.text_area("Tweet", "", height=100)
    if st.button("Autofill Example Tweet"):
        examples = get_example_tweets()
        tweet_input = st.selectbox("Choose an example tweet:", examples, key="example_tweet")

with col2:
    actual_label = st.radio("Actual Label (optional)", ("Real", "Fake"), index=0, help="Select the true label if you know it.")

if st.button("Detect"):
    cleaned = clean_tweet(tweet_input)
    if not cleaned:
        st.warning("Please enter a tweet.")
    else:
        pred = model.predict([cleaned])[0]
        prob = model.predict_proba([cleaned])[0]
        conf = round(max(prob) * 100, 2)
        pred_label = "Fake" if pred == 1 else "Real"
        # Show prediction result
        if pred_label == "Fake":
            st.error(f"Prediction: **{pred_label}**")
        else:
            st.success(f"Prediction: **{pred_label}**")
        # Confidence progress bar
        st.progress(int(conf), text=f"Confidence: {conf}%")
        # Show actual label and comparison
        if actual_label:
            st.info(f"Actual Label: **{actual_label}**")
            if actual_label == pred_label:
                st.success("✅ Prediction matches actual label!")
            else:
                st.error("❌ Prediction does NOT match actual label.") 