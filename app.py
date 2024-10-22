from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import nltk
from flask_sqlalchemy import SQLAlchemy
import os
import logging

# Download NLTK stopwords (one-time setup)
nltk.download('stopwords')
nltk.download('punkt')
nltk.data.path.append('/home/ona/nltk_data')

# Flask app setup
app = Flask(__name__)

# SQLite database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "sqlite:///reviews.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

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

# Define the Review model
class Review(db.Model):
    __tablename__ = 'review'
    id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

# Create the database tables if they do not exist
@app.before_first_request
def create_tables():
    with app.app_context():
        db.create_all()

# Preprocessing function
def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return " ".join(tokens)

# Route for the home page
@app.route('/')
def home():
    reviews = Review.query.all()  # Load previous reviews from the database
    return render_template('index.html', reviews=reviews)

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
        confidence = np.max(model.predict_proba(transformed_text))

        # Determine sentiment based on prediction
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        
        # Save the review in the database
        new_review = Review(review=review_text, sentiment=sentiment, confidence=confidence)
        db.session.add(new_review)
        db.session.commit()

        # Return response with a thank you message
        return jsonify({
            'message': f'Thank you for your review! Your sentiment is {sentiment} with a confidence of {confidence:.2f}.',
            'sentiment': sentiment,
            'confidence': round(confidence, 3)
        })

    except Exception as e:
        logging.error(f"Error predicting sentiment: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route to clear all reviews
@app.route('/clear_reviews', methods=['POST'])
def clear_reviews():
    try:
        # Clear all reviews from the database
        db.session.query(Review).delete()  # This deletes all records in the Review table
        db.session.commit()

        return jsonify({'message': 'All reviews have been cleared successfully.'})

    except Exception as e:
        logging.error(f"Error clearing reviews: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Start the Flask app
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
