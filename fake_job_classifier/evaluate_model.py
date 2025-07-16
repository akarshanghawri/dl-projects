import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.sequence import pad_sequences

# Load tokenizer and model
tokenizer = joblib.load("dl-projects/fake_job_classifier/saved_model/fake_job_tokenizer.pkl")
model = load_model("dl-projects/fake_job_classifier/saved_model/fake_job_lstm_model.keras")

# Load and preprocess data
df = pd.read_csv("dl-projects/fake_job_classifier/data/cleaned_fake_job_postings.csv")
df = df.dropna(subset=['description'])

def clean_text(text):
    import re, string
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_description'] = df['description'].apply(clean_text)

X = tokenizer.texts_to_sequences(df['clean_description'])
X = pad_sequences(X, maxlen=300)
y = df['fraudulent'].values

# Train-test split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict and evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
