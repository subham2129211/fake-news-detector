# 📰 Fake News Detection using Machine Learning

[![View on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fake-news-detector-tweets.streamlit.app/)

A Machine Learning-powered web app to detect fake news using TF-IDF + Logistic Regression. Built with Streamlit and trained on Kaggle’s Fake and True news dataset.

This project is a lightweight and fast fake news classifier that takes in the title and body of a news article and predicts whether it's FAKE or REAL. It uses a clean ML pipeline with TF-IDF and Logistic Regression, and features a Streamlit-based web app for interactive testing.

Built as a B.Tech CSE project to strengthen SDE-aligned ML profile and deploy-ready apps.

---

## 🔧 Features

- Combines `title + text` for better context understanding
- Trims input to 500 characters for consistent results
- Clean preprocessing (punctuation removal, lowercasing, stopword clean)
- TF-IDF feature extraction
- Logistic Regression classifier with 98.5% accuracy
- Fast and responsive Streamlit web interface
- Shows prediction with confidence score

---

## 🗂️ Folder Structure

fake-news-detector/
├── fake_news_detection.ipynb → Complete training notebook
├── fake_news_model_v2.pkl → Trained model (Logistic Regression + TF-IDF)
├── streamlit_app.py → Web app using Streamlit
├── requirements.txt → Python packages used
└── README.md → Project overview and setup (this file)


---

## ⚙️ How to Run Locally

1. Clone the repository

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

2. Install the dependencies

pip install -r requirements.txt

3. Launch the web app

streamlit run streamlit_app.py
