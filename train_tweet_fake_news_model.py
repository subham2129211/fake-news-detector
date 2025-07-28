import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load the dataset
DATA_PATH = 'Truth_Seeker_Model_Dataset.csv'
df = pd.read_csv(DATA_PATH)

# 2. Clean the tweets
def clean_tweet(text):
    if pd.isnull(text):
        return ''
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df = df.dropna(subset=['tweet', 'BinaryNumTarget'])
df['cleaned_tweet'] = df['tweet'].apply(clean_tweet)

# 3. Split into train/test
X = df['cleaned_tweet']
y = df['BinaryNumTarget']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=200))
])

pipeline.fit(X_train, y_train)

# 5. Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# 6. Save the model
joblib.dump(pipeline, 'tweet_fake_news_model.pkl')
print("Model saved as tweet_fake_news_model.pkl") 