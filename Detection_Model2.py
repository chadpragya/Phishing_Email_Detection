import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from sentence_transformers import SentenceTransformer

#cleaning the data
data = pd.read_csv('/Users/pragyatiwari/Desktop/phishing detection/Phishing_Email.csv')
data.drop_duplicates(inplace=True)

# Check for non-string types in the 'Email Text' column
print("Data types in Email Text column:", data['Email Text'].apply(type).value_counts())

#Convert all items to strings
data['Email Text'] = data['Email Text'].astype(str)
data['Email Type'] = data['Email Type'].astype(str)

email_type_counts = data['Email Type'].value_counts()

unique_email_types = email_type_counts.index.tolist()

#categorised the data
text = data['Email Text']
type = data['Email Type']
# Splitting the dataset into training and testing sets
text_train, text_test, type_train, type_test = train_test_split(text, type, test_size=0.2, random_state=42)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(text_train)

#Creating the model
model = MultinomialNB()
model.fit(features, type_train)

# Testing the model
features_test = cv.transform(text_test)
print('model score:', model.score(features_test, type_test))

# Saving the model
joblib.dump(model, 'model.pkl')
joblib.dump(cv, 'vectorizer.pkl')

# Predict Email Type
def predict_email(email_text):
    loaded_model = joblib.load('model.pkl')
    loaded_vectorizer = joblib.load('vectorizer.pkl')
    email_features = loaded_vectorizer.transform([email_text])
    prediction = loaded_model.predict(email_features)[0]
    return prediction

def user_interface():
    print("\n--- Phishing Email Detection System ---")
    print("Enter the email text to check if it's phishing or not(enter 'exit' to quit):")
    while True:
        user_input = input("Email Text: ")
        if  user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        result = predict_email(user_input)
        print(f"The email is classified as: {result}")
        
        if result == 'Phishing Email':
            print("Warning: This email is likely a phishing attempt!")
        else:
            print("This email seems to be safe.")

if __name__ == "__main__":
    user_interface()
