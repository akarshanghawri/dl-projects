from flask import Flask, render_template, request
import numpy as np
import joblib
import re
import string
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os

# Setup
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'saved_model/fake_job_lstm_model.keras'))
tokenizer = joblib.load(os.path.join(BASE_DIR, 'saved_model/fake_job_tokenizer.pkl'))
MAX_LEN = 300

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        input_text = request.form['job_desc']
        clean = clean_text(input_text)
        seq = tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        pred = model.predict(padded)[0][0]

        label = 'Fake' if pred > 0.5 else 'Legitimate'
        confidence = f"{pred*100:.2f}%" if label == 'Fake' else f"{(1-pred)*100:.2f}%"
        prediction = f"🔎 This job posting is likely <b>{label}</b> with a confidence of <b>{confidence}</b>"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
