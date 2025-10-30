"""Simple config untuk sentiment analysis app"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Paths untuk model files
MODEL_WORD_PATH = os.path.join(MODELS_DIR, 'model_word_based.joblib')
VECTORIZER_WORD_PATH = os.path.join(MODELS_DIR, 'vectorizer_word_based.joblib')
MODEL_TRIGRAM_PATH = os.path.join(MODELS_DIR, 'model_trigram.joblib')
VECTORIZER_TRIGRAM_PATH = os.path.join(MODELS_DIR, 'vectorizer_trigram.joblib')
PREPROCESSING_PATH = os.path.join(MODELS_DIR, 'preprocessing_tools.joblib')
