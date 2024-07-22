from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

# Load the model
model = load_model('mental_health_lstm_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Initialize Flask app
app = Flask(__name__)

# Route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    # Preprocess text
    max_sequence_length = 100
    text_sequence = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_sequence, maxlen=max_sequence_length)

    # Make prediction
    prediction = model.predict(text_padded)
    label = int(prediction[0][0] > 0.5)

    # Return the result
    return jsonify({'label': label})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
