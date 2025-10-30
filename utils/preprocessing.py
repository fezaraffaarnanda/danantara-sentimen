"""Simple preprocessing functions"""

import re
import emoji
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)


def preprocess_text(text, stopwords, norm_dict, stemmer):
    """Preprocess text untuk sentiment analysis"""
    
    if not text:
        return ''
    
    text = str(text)
    
    # Cleaning
    text = emoji.demojize(text)
    text = re.sub(r':[a-z_]+:', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+', '', text)
    
    # Casefolding
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Normalization
    tokens = [norm_dict.get(word, word) for word in tokens]
    
    # Stopword removal
    tokens = [word for word in tokens if word not in stopwords and len(word) > 2]
    
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)
