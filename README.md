# Phishing_Email_Detection
Phishing Email Detection using Machine Learning  This project is a Flask-based web application that detects phishing emails using a trained machine learning model. Users can input email text, and the model classifies it as either legitimate or phishing based on content analysis.
## Features

- Predicts whether an email is a **Phishing Email** or **Legitimate Email**
- Built using **Naive Bayes Classifier** with **CountVectorizer**
- Achieves up to **95% accuracy** on real-world email datasets
- Interactive **Flask-based backend** with a responsive **HTML frontend**
- Confidence score for predictions
- Clean, user-friendly interface
- Supports JSON-based API interaction (ideal for integration/testing)
---
##  Machine Learning Model

- **Algorithm**: Multinomial Naive Bayes
- **Text Preprocessing**: CountVectorizer with stopword removal
- **Dataset**: `Phishing_Email.csv`  
  - Cleaned and de-duplicated email text data labeled as `Phishing Email` or `Safe Email`
- **Train-Test Split**: 80-20
- **Model Accuracy**: ~95%
---
##  Tech Stack

- **Language**: Python
- **Framework**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **ML Libraries**: `scikit-learn`, `pandas`, `joblib`, `matplotlib`
- **API Testing**: Postman 
---
##  Project Structure

📁 phishing-email-detector/
├── model.pkl # Trained Naive Bayes model
├── vectorizer.pkl # CountVectorizer object
├── app.py # Flask backend
├── phishing_model.py # ML model training & prediction script
├── templates/
│ └── index1.html # Web frontend
└── Phishing_Email.csv # Dataset used for training
