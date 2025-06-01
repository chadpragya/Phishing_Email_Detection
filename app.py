from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from dotenv import load_dotenv
import os

load_dotenv()

# Example usage
secret_key = os.getenv('SECRET_KEY')

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('model.pkl')
cv = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        email_text = data['email_text']
        
        # Transform input using the same vectorizer used during training
        email_features = cv.transform([email_text]).toarray()
        
        # Make prediction
        prediction = model.predict(email_features)[0]
        
        # Get probability of prediction
        prediction_proba = model.predict_proba(email_features)[0]
        confidence = max(prediction_proba) * 100
        
        result = {
            'prediction': prediction,
            'confidence': float(confidence),
            'is_phishing': bool(prediction == 'Phishing Email')
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run(debug=True)
# This code is a Flask web application that serves as an interface for a phishing email detection model.
# It allows users to input email text and receive predictions on whether the email is phishing or legitimate.
