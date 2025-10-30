"""Load models dan predict"""

import joblib
from .preprocessing import preprocess_text


def load_models(config):
    """Load semua model dan preprocessing tools"""
    try:
        preprocessing = joblib.load(config.PREPROCESSING_PATH)
        model_word = joblib.load(config.MODEL_WORD_PATH)
        vectorizer_word = joblib.load(config.VECTORIZER_WORD_PATH)
        model_trigram = joblib.load(config.MODEL_TRIGRAM_PATH)
        vectorizer_trigram = joblib.load(config.VECTORIZER_TRIGRAM_PATH)
        
        return {
            'preprocessing': preprocessing,
            'model_word': model_word,
            'vectorizer_word': vectorizer_word,
            'model_trigram': model_trigram,
            'vectorizer_trigram': vectorizer_trigram,
            'loaded': True
        }
    except Exception as e:
        return {'loaded': False, 'error': str(e)}


def predict_sentiment(text, model, vectorizer, preprocessing):
    """Predict sentiment dari text"""
    try:
        # Preprocessing
        preprocessed = preprocess_text(
            text,
            preprocessing['combined_stopwords'],
            preprocessing['normalization_dict'],
            preprocessing['stemmer']
        )
        
        if not preprocessed:
            return {'success': False, 'error': 'Text kosong setelah preprocessing'}
        
        # Transform dan predict
        features = vectorizer.transform([preprocessed])
        prediction = int(model.predict(features)[0])  # Convert ke int
        probabilities = model.predict_proba(features)[0]
        
        sentiment = "Positif" if prediction == 1 else "Negatif"
        confidence = float(probabilities[prediction] * 100)  # Convert ke float
        
        return {
            'success': True,
            'sentiment': sentiment,
            'confidence': confidence,
            'prob_negatif': float(probabilities[0] * 100),
            'prob_positif': float(probabilities[1] * 100),
            'preprocessed': preprocessed
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
