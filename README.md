# 🕵️ Tweet Fake News Detector

A machine learning-powered web app to detect fake news in tweets, built with Python, scikit-learn, and Streamlit.

---

## 🚀 Live Demo

Try the app here: [Streamlit Live App](YOUR_STREAMLIT_CLOUD_LINK_HERE)

---

## 📖 Project Overview

With the rise of misinformation on social media, this project aims to automatically detect fake news in tweets using machine learning. The app provides a simple interface for users to check if a tweet is likely to be fake or real.

---

## 🗂️ Dataset

- **Source:** `Truth_Seeker_Model_Dataset.csv`
- **Features:**  
  - Tweet text and various linguistic features (verb/adjective/pronoun counts, etc.)
  - User metadata (followers, friends, statuses, bot score, etc.)
  - Named Entity Recognition (NER) percentages (ORG, PERSON, GPE, etc.)
- **Target:**  
  - Binary label: 1 = True (real), 0 = Fake

---

## 🛠️ Technologies Used

- **Python**: Core programming language
- **pandas, numpy**: Data manipulation and numerical operations
- **scikit-learn**: Machine learning (Logistic Regression, TF-IDF)
- **joblib**: Model serialization
- **Streamlit**: Web app interface

---

## 🧑‍💻 How It Works

1. **Data Preprocessing:**  
   - Cleans tweet text (removes URLs, mentions, hashtags, punctuation, etc.)
   - Handles missing values

2. **Feature Engineering:**  
   - Uses TF-IDF vectorization for text features

3. **Model Training:**  
   - Logistic Regression classifier trained on cleaned tweet data

4. **Evaluation:**  
   - Model evaluated using accuracy and classification report

5. **Deployment:**  
   - Streamlit app for real-time predictions

---

## 🏃‍♂️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (optional)
```bash
python train_tweet_fake_news_model.py
```

### 4. Run the Streamlit app
```bash
streamlit run streamlit_tweet_detector.py
```

---

## 🌐 Usage

- Enter a tweet in the text box.
- Click **Detect** to see if it’s likely fake or real.
- View the confidence score and compare with the actual label (if known).

---

## 📊 Example

![App Screenshot](screenshot.png) <!-- Add a screenshot if available -->

---

## 📎 Live App

[Streamlit Live App](YOUR_STREAMLIT_CLOUD_LINK_HERE)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

- Your Name
- [Your LinkedIn](https://www.linkedin.com/in/yourprofile)
- [Your GitHub](https://github.com/yourusername) 