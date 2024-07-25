from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Initialize NLP tools
nltk.download('punkt')
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Ensuring the input is a string
    if not isinstance(text, str):
        text = str(text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Lowercase
    tokens = [token.lower() for token in tokens]
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    # Apply stemming
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

# Load the Logistic Regression model and the CountVectorizer
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

# Route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Preprocess text
        text_processed = preprocess_text(text)
        # Convert text to features using CountVectorizer
        text_vectorized = vectorizer.transform([text_processed])

        # Make prediction
        prediction_prob = model.predict_proba(text_vectorized)[0, 1]  
        threshold = 0.3  
        label = int(prediction_prob > threshold)

        # Return the result
        return jsonify({'label': label, 'probability': prediction_prob})

    except Exception as e:
        # Print error for debugging
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while processing the request'}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
