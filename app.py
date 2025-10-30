"""
DANANTARA Sentiment Analysis - Simple Streamlit App
"""

import streamlit as st
import nltk

# Download NLTK data untuk deployment
@st.cache_resource
def download_nltk_resources():
    """Download NLTK data yang diperlukan"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        return True
    except:
        return False

# Download NLTK data
download_nltk_resources()

import config
from utils.model_loader import load_models, predict_sentiment

# Page config
st.set_page_config(
    page_title="DANANTARA Sentiment Analysis",
    page_icon="🎯",
    layout="wide"
)

# Title
st.title("🎯 DANANTARA Sentiment Analysis")
st.markdown("Analisis sentimen menggunakan 2 model Naive Bayes berbeda")
st.markdown("---")

# Load models
@st.cache_resource
def init_models():
    return load_models(config)

with st.spinner("Loading models..."):
    models_data = init_models()

if not models_data['loaded']:
    st.error(f"❌ Gagal load model: {models_data['error']}")
    st.info("Pastikan semua file model ada di folder 'models/'")
    st.stop()

st.success("✅ Models loaded!")

# Input text
st.subheader("📝 Input Text")
text_input = st.text_area(
    "Masukkan text untuk dianalisis:",
    height=150,
    placeholder="Contoh: DANANTARA sangat bagus dan pelayanannya memuaskan!"
)

# Predict button
if st.button("🔮 Analisis Sentimen", use_container_width=True):
    if not text_input or text_input.strip() == '':
        st.warning("⚠️ Masukkan text terlebih dahulu!")
    else:
        with st.spinner("Analyzing..."):
            # Predict dengan model 1 (word-based)
            result1 = predict_sentiment(
                text_input,
                models_data['model_word'],
                models_data['vectorizer_word'],
                models_data['preprocessing']
            )
            
            # Predict dengan model 2 (trigram)
            result2 = predict_sentiment(
                text_input,
                models_data['model_trigram'],
                models_data['vectorizer_trigram'],
                models_data['preprocessing']
            )
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")
        
        col1, col2 = st.columns(2)
        
        # Model 1 results
        with col1:
            st.markdown("### Model 1: Word-Based")
            if result1['success']:
                sentiment_emoji = "😊" if result1['sentiment'] == "Positif" else "😞"
                st.markdown(f"## {sentiment_emoji} {result1['sentiment']}")
                st.metric("Confidence", f"{result1['confidence']:.2f}%")
                
                st.write("**Detail Probabilitas:**")
                st.progress(result1['prob_positif'] / 100)
                st.caption(f"😊 Positif: {result1['prob_positif']:.2f}%")
                st.progress(result1['prob_negatif'] / 100)
                st.caption(f"😞 Negatif: {result1['prob_negatif']:.2f}%")
            else:
                st.error(f"Error: {result1['error']}")
        
        # Model 2 results
        with col2:
            st.markdown("### Model 2: Word + Tri-gram")
            if result2['success']:
                sentiment_emoji = "😊" if result2['sentiment'] == "Positif" else "😞"
                st.markdown(f"## {sentiment_emoji} {result2['sentiment']}")
                st.metric("Confidence", f"{result2['confidence']:.2f}%")
                
                st.write("**Detail Probabilitas:**")
                st.progress(result2['prob_positif'] / 100)
                st.caption(f"😊 Positif: {result2['prob_positif']:.2f}%")
                st.progress(result2['prob_negatif'] / 100)
                st.caption(f"😞 Negatif: {result2['prob_negatif']:.2f}%")
            else:
                st.error(f"Error: {result2['error']}")
        
        # Comparison
        if result1['success'] and result2['success']:
            st.markdown("---")
            st.subheader("📊 Perbandingan Model")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model 1 Confidence", f"{result1['confidence']:.1f}%")
            
            with col2:
                st.metric("Model 2 Confidence", f"{result2['confidence']:.1f}%")
            
            with col3:
                agreement = "✅ Sama" if result1['sentiment'] == result2['sentiment'] else "⚠️ Beda"
                st.metric("Kesepakatan", agreement)
            
            # Show preprocessed text
            with st.expander("🔍 Lihat Text Setelah Preprocessing"):
                preprocessed = result1.get('preprocessed', '')
                if preprocessed:
                    st.code(preprocessed)
                    st.caption(f"Original: {len(text_input)} karakter | Preprocessed: {len(preprocessed)} karakter")

# Footer
st.markdown("---")
st.caption("DANANTARA Sentiment Analysis © 2025")
