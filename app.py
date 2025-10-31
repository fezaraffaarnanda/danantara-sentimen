"""
DANANTARA Sentiment Analysis - Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

download_nltk_resources()

import config
from utils.model_loader import load_models, predict_sentiment

# Page config
st.set_page_config(
    page_title="DANANTARA Sentiment Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Mode
st.markdown("""
<style>
    /* Main Background - Dark */
    .main {
        background: #0e1117;
        padding: 0;
    }
    
    /* Container */
    .stApp {
        background: #0e1117;
    }
    
    /* Text Color */
    .stMarkdown, .stText, p, span, label {
        color: #e0e0e0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 0 0 30px 30px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    /* Cards */
    .metric-card {
        background: #1e1e2e;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .result-card-positive {
        background: linear-gradient(135deg, #0f7c6d 0%, #2dd881 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 30px rgba(45, 216, 129, 0.4);
    }
    
    .result-card-negative {
        background: linear-gradient(135deg, #c72d3c 0%, #e84855 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 30px rgba(232, 72, 85, 0.4);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Sidebar - Dark */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Input Fields - Dark */
    .stTextArea textarea, .stTextInput input {
        background-color: #1e1e2e !important;
        color: #e0e0e0 !important;
        border: 2px solid #2d2d3d !important;
        border-radius: 10px;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* File Uploader - Dark */
    [data-testid="stFileUploader"] {
        background-color: #1e1e2e;
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"] label {
        color: #e0e0e0 !important;
    }
    
    /* Dataframe - Dark */
    .stDataFrame {
        background-color: #1e1e2e;
        border-radius: 10px;
    }
    
    /* Metrics - Dark */
    [data-testid="stMetricValue"] {
        color: #667eea !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tab styling - Dark */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2e;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: #a0a0a0;
        border: 1px solid #2d2d3d;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
    }
    
    /* Expander - Dark */
    .streamlit-expanderHeader {
        background-color: #1e1e2e !important;
        color: #e0e0e0 !important;
        border-radius: 10px;
        border: 1px solid #2d2d3d;
    }
    
    .streamlit-expanderContent {
        background-color: #1a1a24 !important;
        border: 1px solid #2d2d3d;
        border-top: none;
    }
    
    /* Info/Warning/Error boxes - Dark */
    .stAlert {
        background-color: #1e1e2e !important;
        border-radius: 10px;
    }
    
    /* Download Button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #2dd881 0%, #0f7c6d 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(45, 216, 129, 0.3);
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(45, 216, 129, 0.5);
    }
    
    /* Caption - Dark */
    .caption {
        color: #a0a0a0 !important;
    }
    
    /* Code block - Dark */
    code {
        background-color: #1e1e2e !important;
        color: #2dd881 !important;
        padding: 0.2rem 0.4rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def init_models():
    return load_models(config)

# Header
st.markdown("""
<div class="header-container">
    <h1 style="font-size: 3rem; margin: 0;">🎯 DANANTARA Sentiment Analysis</h1>
    <p style="font-size: 1.2rem; opacity: 0.9; margin-top: 1rem;">
        Analisis Sentimen Menggunakan Dua Model Naive Bayes
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Model Info
with st.sidebar:
    st.markdown("### 🤖 Informasi Model")
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h4 style="color: white; margin: 0;">Model 1: Word-Based</h4>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0.5rem 0 0 0;">
        Naive Bayes dengan fitur TF-IDF menggunakan unigram (kata tunggal).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
        <h4 style="color: white; margin: 0;">Model 2: Word + Tri-gram</h4>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0.5rem 0 0 0;">
        Naive Bayes dengan fitur TF-IDF menggunakan unigram, bigram, dan trigram.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Cara Menggunakan")
    st.markdown("""
    <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">
    <b>Single Text:</b> Masukkan teks di tab "Single Text"<br><br>
    <b>Bulk Analysis:</b> Upload file CSV/Excel di tab "Upload File"<br><br>
    File harus memiliki kolom bernama <code>text</code> atau <code>review</code>
    </p>
    """, unsafe_allow_html=True)

# Load models
with st.spinner("Loading models..."):
    models_data = init_models()

if not models_data['loaded']:
    st.error(f"❌ Gagal load model: {models_data['error']}")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["📝 Single Text", "📁 Upload File"])

# TAB 1: Single Text Analysis
with tab1:
    st.markdown("### Analisis Sentimen untuk Satu Teks")
    
    text_input = st.text_area(
        "Masukkan teks:",
        height=150,
        placeholder="Contoh: DANANTARA sangat bagus dan pelayanannya memuaskan!"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🔮 Analisis Sentimen", key="single")
    
    if analyze_btn:
        if not text_input or text_input.strip() == '':
            st.warning("⚠️ Masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Analyzing..."):
                # Predict dengan kedua model
                result1 = predict_sentiment(
                    text_input,
                    models_data['model_word'],
                    models_data['vectorizer_word'],
                    models_data['preprocessing']
                )
                
                result2 = predict_sentiment(
                    text_input,
                    models_data['model_trigram'],
                    models_data['vectorizer_trigram'],
                    models_data['preprocessing']
                )
            
            st.markdown("---")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model 1: Word-Based")
                if result1['success']:
                    card_class = "result-card-positive" if result1['sentiment'] == "Positif" else "result-card-negative"
                    emoji = "😊" if result1['sentiment'] == "Positif" else "😞"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h2 style="margin: 0; font-size: 2.5rem;">{emoji} {result1['sentiment']}</h2>
                        <p style="margin: 1rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                            Confidence: {result1['confidence']:.2f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("**Detail Probabilitas:**")
                    st.progress(result1['prob_positif'] / 100)
                    st.caption(f"😊 Positif: {result1['prob_positif']:.2f}%")
                    st.progress(result1['prob_negatif'] / 100)
                    st.caption(f"😞 Negatif: {result1['prob_negatif']:.2f}%")
                else:
                    st.error(f"Error: {result1['error']}")
            
            with col2:
                st.markdown("#### Model 2: Word + Tri-gram")
                if result2['success']:
                    card_class = "result-card-positive" if result2['sentiment'] == "Positif" else "result-card-negative"
                    emoji = "😊" if result2['sentiment'] == "Positif" else "😞"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h2 style="margin: 0; font-size: 2.5rem;">{emoji} {result2['sentiment']}</h2>
                        <p style="margin: 1rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                            Confidence: {result2['confidence']:.2f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("**Detail Probabilitas:**")
                    st.progress(result2['prob_positif'] / 100)
                    st.caption(f"😊 Positif: {result2['prob_positif']:.2f}%")
                    st.progress(result2['prob_negatif'] / 100)
                    st.caption(f"😞 Negatif: {result2['prob_negatif']:.2f}%")
                else:
                    st.error(f"Error: {result2['error']}")

# TAB 2: Bulk Analysis
with tab2:
    st.markdown("### Analisis Sentimen untuk Multiple Teks")
    st.info("📌 Upload file CSV atau Excel dengan kolom **'text'** atau **'review'**")
    
    uploaded_file = st.file_uploader(
        "Upload file:",
        type=['csv', 'xlsx', 'xls'],
        help="File harus memiliki kolom 'text' atau 'review'"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Check kolom
            text_col = None
            if 'text' in df.columns:
                text_col = 'text'
            elif 'review' in df.columns:
                text_col = 'review'
            else:
                st.error("❌ File harus memiliki kolom 'text' atau 'review'")
                st.stop()
            
            st.success(f"✅ File berhasil diupload! Total: {len(df)} baris")
            
            # Show preview
            with st.expander("👀 Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_bulk_btn = st.button("🚀 Analisis Semua Data", key="bulk")
            
            if analyze_bulk_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_model1 = []
                results_model2 = []
                
                total = len(df)
                
                error_count = 0
                error_data = []  # Track data yang error
                
                for idx, row in df.iterrows():
                    # Update progress
                    progress = (idx + 1) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {idx + 1}/{total}...")
                    
                    text = str(row[text_col])
                    
                    # Skip teks kosong atau terlalu pendek
                    if not text or len(text.strip()) < 1:
                        results_model1.append({'success': False, 'sentiment': 'N/A', 'confidence': 0})
                        results_model2.append({'success': False, 'sentiment': 'N/A', 'confidence': 0})
                        error_count += 1
                        error_data.append({'index': idx, 'text': text, 'reason': 'Teks kosong atau terlalu pendek'})
                        continue
                    
                    # Predict dengan model 1
                    result1 = predict_sentiment(
                        text,
                        models_data['model_word'],
                        models_data['vectorizer_word'],
                        models_data['preprocessing']
                    )
                    results_model1.append(result1)
                    
                    if not result1['success']:
                        error_count += 1
                        error_data.append({
                            'index': idx, 
                            'text': text[:100] + '...' if len(text) > 100 else text,
                            'reason': result1.get('error', 'Gagal preprocessing/prediksi')
                        })
                    
                    # Predict dengan model 2
                    result2 = predict_sentiment(
                        text,
                        models_data['model_trigram'],
                        models_data['vectorizer_trigram'],
                        models_data['preprocessing']
                    )
                    results_model2.append(result2)
                
                progress_bar.empty()
                status_text.empty()
                
                # Compile results
                df['sentiment_model1'] = [r.get('sentiment', 'N/A') if r['success'] else 'N/A' for r in results_model1]
                df['confidence_model1'] = [r.get('confidence', 0) if r['success'] else 0 for r in results_model1]
                df['sentiment_model2'] = [r.get('sentiment', 'N/A') if r['success'] else 'N/A' for r in results_model2]
                df['confidence_model2'] = [r.get('confidence', 0) if r['success'] else 0 for r in results_model2]
                
                st.success("✅ Analisis selesai!")
                
                # Tampilkan warning jika ada error
                if error_count > 0:
                    st.warning(f"⚠️ {error_count} data gagal diproses (teks kosong atau terlalu pendek)")
                    
                    # Dropdown untuk lihat data error
                    with st.expander("🔍 Lihat data yang gagal diproses"):
                        if error_data:
                            error_df = pd.DataFrame(error_data)
                            error_df.columns = ['Index', 'Text', 'Alasan Error']
                            st.dataframe(error_df, use_container_width=True, height=300)
                            
                            # Info tambahan
                            st.caption(f"Total: {len(error_df)} data gagal diproses")
                        else:
                            st.info("Tidak ada detail error tersedia")
                
                st.markdown("---")
                st.markdown("## 📊 Hasil Analisis")
                
                # Filter out N/A untuk visualisasi
                df_valid = df[(df['sentiment_model1'] != 'N/A') & (df['sentiment_model2'] != 'N/A')]
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Data", len(df))
                
                with col2:
                    pos_model1 = (df_valid['sentiment_model1'] == 'Positif').sum()
                    st.metric("Model 1: Positif", pos_model1)
                
                with col3:
                    pos_model2 = (df_valid['sentiment_model2'] == 'Positif').sum()
                    st.metric("Model 2: Positif", pos_model2)
                
                with col4:
                    if len(df_valid) > 0:
                        agreement = (df_valid['sentiment_model1'] == df_valid['sentiment_model2']).sum()
                        agreement_pct = (agreement / len(df_valid)) * 100
                        st.metric("Kesepakatan", f"{agreement_pct:.1f}%")
                    else:
                        st.metric("Kesepakatan", "N/A")
                
                # Visualizations
                st.markdown("### 📈 Visualisasi")
                
                tab_viz1, tab_viz2, tab_viz3 = st.tabs(["Distribution", "Comparison", "Confidence"])
                
                with tab_viz1:
                    if len(df_valid) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Model 1 distribution
                            sentiment_counts1 = df_valid['sentiment_model1'].value_counts()
                            fig1 = px.pie(
                                values=sentiment_counts1.values,
                                names=sentiment_counts1.index,
                                title="Model 1: Distribusi Sentimen",
                                color=sentiment_counts1.index,
                                color_discrete_map={'Positif': '#38ef7d', 'Negatif': '#f45c43'}
                            )
                            fig1.update_traces(textinfo='percent+label', textfont_size=14)
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            # Model 2 distribution
                            sentiment_counts2 = df_valid['sentiment_model2'].value_counts()
                            fig2 = px.pie(
                                values=sentiment_counts2.values,
                                names=sentiment_counts2.index,
                                title="Model 2: Distribusi Sentimen",
                                color=sentiment_counts2.index,
                                color_discrete_map={'Positif': '#38ef7d', 'Negatif': '#f45c43'}
                            )
                            fig2.update_traces(textinfo='percent+label', textfont_size=14)
                            st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.warning("⚠️ Tidak ada data valid untuk divisualisasi")
                
                with tab_viz2:
                    if len(df_valid) > 0:
                        # Comparison bar chart
                        comparison_data = pd.DataFrame({
                            'Model': ['Model 1', 'Model 2'],
                            'Positif': [
                                (df_valid['sentiment_model1'] == 'Positif').sum(),
                                (df_valid['sentiment_model2'] == 'Positif').sum()
                            ],
                            'Negatif': [
                                (df_valid['sentiment_model1'] == 'Negatif').sum(),
                                (df_valid['sentiment_model2'] == 'Negatif').sum()
                            ]
                        })
                        
                        fig3 = go.Figure()
                        fig3.add_trace(go.Bar(
                            name='Positif',
                            x=comparison_data['Model'],
                            y=comparison_data['Positif'],
                            marker_color='#38ef7d'
                        ))
                        fig3.add_trace(go.Bar(
                            name='Negatif',
                            x=comparison_data['Model'],
                            y=comparison_data['Negatif'],
                            marker_color='#f45c43'
                        ))
                        
                        fig3.update_layout(
                            title="Perbandingan Hasil Kedua Model",
                            barmode='group',
                            xaxis_title="Model",
                            yaxis_title="Jumlah",
                            height=400
                        )
                        
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.warning("⚠️ Tidak ada data valid untuk divisualisasi")
                
                with tab_viz3:
                    if len(df_valid) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Model 1 confidence distribution (exclude 0)
                            df_conf1 = df_valid[df_valid['confidence_model1'] > 0]
                            if len(df_conf1) > 0:
                                fig4 = px.histogram(
                                    df_conf1,
                                    x='confidence_model1',
                                    nbins=20,
                                    title="Model 1: Distribusi Confidence",
                                    labels={'confidence_model1': 'Confidence (%)'},
                                    color_discrete_sequence=['#667eea']
                                )
                                fig4.update_layout(showlegend=False)
                                st.plotly_chart(fig4, use_container_width=True)
                            else:
                                st.info("Tidak ada data confidence untuk Model 1")
                        
                        with col2:
                            # Model 2 confidence distribution (exclude 0)
                            df_conf2 = df_valid[df_valid['confidence_model2'] > 0]
                            if len(df_conf2) > 0:
                                fig5 = px.histogram(
                                    df_conf2,
                                    x='confidence_model2',
                                    nbins=20,
                                    title="Model 2: Distribusi Confidence",
                                    labels={'confidence_model2': 'Confidence (%)'},
                                    color_discrete_sequence=['#764ba2']
                                )
                                fig5.update_layout(showlegend=False)
                                st.plotly_chart(fig5, use_container_width=True)
                            else:
                                st.info("Tidak ada data confidence untuk Model 2")
                    else:
                        st.warning("⚠️ Tidak ada data valid untuk divisualisasi")
                
                # Download results
                st.markdown("---")
                st.markdown("### 💾 Download Hasil")
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>DANANTARA Sentiment Analysis © 2025</p>
</div>
""", unsafe_allow_html=True)
