from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, 'saved_model')
model = load_model(os.path.join(model_dir, 'ann_model.keras'))
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get form inputs
            sr_no_input = 999  # Placeholder
            startup_name_input = request.form['startup'].strip()
            industry_input = request.form['industry'].strip()
            subvertical_input = request.form['subvertical'].strip()
            city_input = request.form['city'].strip()
            investor_input = request.form['investor'].strip()
            inv_type_input = request.form['inv_type'].strip()
            remarks_input = "N/A"  # Placeholder

            # exception handling
            try:
                startup = encoders['startup_name'].transform([startup_name_input])[0]
            except:
                raise ValueError(f"Startup '{startup_name_input}' not recognized.")

            try:
                industry = encoders['industry_vertical'].transform([industry_input])[0]
            except:
                raise ValueError(f"Industry '{industry_input}' not recognized.")

            try:
                subvertical = encoders['subvertical'].transform([subvertical_input])[0]
            except:
                raise ValueError(f"Subvertical '{subvertical_input}' not recognized.")

            try:
                city = encoders['city__location'].transform([city_input])[0]
            except:
                raise ValueError(f"City '{city_input}' not recognized.")

            try:
                investor = encoders['investors_name'].transform([investor_input])[0]
            except:
                raise ValueError(f"Investor '{investor_input}' not recognized.")

            try:
                inv_type = encoders['investmentntype'].transform([inv_type_input])[0]
            except:
                raise ValueError(f"Investment Type '{inv_type_input}' not recognized.")

            try:
                remarks = encoders['remarks'].transform([remarks_input])[0]
            except:
                remarks = 0  # fallback if remarks unknown

            # final input in the form of [sr_no, startup_name, industry, subvertical, city, investor, inv_type, remarks]
            input_data = np.array([[sr_no_input, startup, industry, subvertical, city, investor, inv_type, remarks]])
            input_scaled = scaler.transform(input_data)

            # Predict
            pred = model.predict(input_scaled)[0][0]
            print("Raw prediction:", pred)

            # Conversion
            inr_value = pred
    
            # Conversion assuming prediction is in Lakhs
            inr_cr = inr_value / 100
            usd_k = (inr_value * 1e5) / 83 / 1e3

            inr_formatted = f"₹{inr_cr:.2f} Cr"
            usd_formatted = f"${usd_k:.0f}K"

            prediction = f"💰 Estimated Funding: {inr_formatted} (approx. {usd_formatted})"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

# run the program 
if __name__ == '__main__':
    app.run(debug=True, port=5000)
