import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data/startup_funding.csv')
model_dir = os.path.join(BASE_DIR, 'saved_model')
os.makedirs(model_dir, exist_ok=True)

# Load & Clean Data
df = pd.read_csv(data_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# Drop irrelevant or problematic columns
df.drop(columns=['date_dd/mm/yyyy'], inplace=True)

# Dictionary to store label encoders for each column
encoders = {}
cat_cols = df.select_dtypes(include='object').columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    
# Save encoders
joblib.dump(encoders, os.path.join(model_dir, 'label_encoders.pkl'))

# Feature and target Split
X = df.drop(columns=['amount_in_usd'])
y = df['amount_in_usd']
print(X.columns.tolist())

# Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

# build ANN Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),           # relu: replaces negative values with 0
    Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# train the model
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluating the model
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest MAE: ₹{test_mae:,.2f}")
print(f"Test MSE: {test_loss:,.2f}")

# Save the Model
model.save(os.path.join(model_dir, 'ann_model.keras'))

# to plot the training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()
