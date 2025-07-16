import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import re
import string
import joblib

# Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data/cleaned_fake_job_postings.csv')
model_dir = os.path.join(BASE_DIR, 'saved_model')
os.makedirs(model_dir, exist_ok=True)

# load the data
df = pd.read_csv(data_path)

# null descriptions dropped 
df = df.dropna(subset=['description'])

# Preprocessing text
def clean_text(text):
    text = text.lower()         # lowercase 
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)   #removes punctuations
    text = re.sub(r"\d+", "", text)     # removes numbers 
    text = re.sub(r"\s+", " ", text).strip()        # removes whitespaces
    return text

df['clean_description'] = df['description'].apply(clean_text)

# Tokenization
MAX_WORDS = 10000        # use only the 10,000 most frequent words
MAX_LEN = 300
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")       # OOV - Out Of Vocabulary
tokenizer.fit_on_texts(df['clean_description'])

X = tokenizer.texts_to_sequences(df['clean_description'])
X = pad_sequences(X, maxlen=MAX_LEN)

y = df['fraudulent'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Model
model = Sequential()
model.add(Embedding(MAX_WORDS, 64, input_length=MAX_LEN))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))       #sigmoid squashes output to a value between 0 and 1, representing the probability the job is fake

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with class weights
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early_stop],
    class_weight=class_weights
)

# Save model and tokenizer
model.save(os.path.join(model_dir, 'fake_job_lstm_model.keras'))
joblib.dump(tokenizer, os.path.join(model_dir, 'fake_job_tokenizer.pkl'))




