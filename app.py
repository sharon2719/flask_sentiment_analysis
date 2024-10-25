from flask import Flask, request, jsonify
import joblib
import numpy as np
import nltk
import logging

# Download NLTK stopwords (one-time setup)
nltk.download('stopwords')
nltk.download('punkt')
nltk.data.path.append('/home/ona/nltk_data')

# Flask app setup
app = Flask(__name__)

# Load models and vectorizer
logistic_model = joblib.load('logistic_regression_model.pkl')
naive_bayes_model = joblib.load('naive_bayes_model.pkl')
decision_tree_model = joblib.load('decision_tree_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define a mapping of model names to their respective objects
models = {
    'logistic': logistic_model,
    'naive_bayes': naive_bayes_model,
    'decision_tree': decision_tree_model,
}

# Preprocessing function
def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return " ".join(tokens)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review_text = data.get('review', '')
    model_name = data.get('model', 'logistic')

    if not review_text:
        return jsonify({'error': 'Review text is required.'}), 400

    if model_name not in models:
        return jsonify({'error': f'Model {model_name} is not available.'}), 400

    try:
        # Preprocess review
        processed_text = preprocess_text(review_text)
        transformed_text = tfidf_vectorizer.transform([processed_text])
        
        model = models[model_name]
        prediction = model.predict(transformed_text)
        confidence = np.max(model.predict_proba(transformed_text)) * 100  # Convert to percentage

        # Determine sentiment based on prediction
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        
        # Return response with sentiment and confidence score in percentage
        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence, 2)  # Round to two decimal places for readability
        })

    except Exception as e:
        logging.error(f"Error predicting sentiment: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Start the Flask app
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
