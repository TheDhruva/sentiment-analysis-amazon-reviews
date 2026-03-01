import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy import sparse
import io
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Sentiment Engine",
    page_icon="🚀",
    layout="wide",
)

# =========================
# CUSTOM NEUBRUTALISM UI
# =========================
st.markdown("""
<style>
/* Scale up global text size slightly */
html, body, [class*="css"] {
    font-size: 18px !important; 
}
body {
    background-color: #FFD166;
}
.main {
    background-color: #FFD166;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.brutal-card {
    background-color: #06D6A0;
    border: 5px solid #000000;
    box-shadow: 8px 8px 0px #000000;
    padding: 24px;
    margin-bottom: 24px;
    color: #FFFFFF;
}
.brutal-card p, .brutal-card span, .brutal-card label {
    color: #FFFFFF !important;
}
.brutal-title {
    font-size: 32px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 12px;
}
.brutal-subtitle {
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF; 
    margin-bottom: 15px;
}
.stButton>button {
    background-color: #EF476F;
    color: #FFFFFF;
    border: 4px solid #000000;
    box-shadow: 6px 6px 0px #000000;
    font-weight: 700;
}
.stDownloadButton>button {
    background-color: #118AB2;
    color: #000000;
    border: 4px solid #000000;
    box-shadow: 6px 6px 0px #000000;
    font-weight: 700;
}
.stTextArea textarea {
    border: 4px solid #000000 !important;
    box-shadow: 6px 6px 0px #000000 !important;
    color: #000000 !important;
}
.stFileUploader {
    border: 4px solid #000000;
    box-shadow: 6px 6px 0px #000000;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL LOADING (CACHED)
# =========================
@st.cache_resource
def load_artifacts():
    scaler = pickle.load(open("Models/scaler.pkl", "rb"))
    model = pickle.load(open("Models/model_xgb.pkl", "rb"))
    vectorizer = pickle.load(open("Models/CountVectorizer.pkl", "rb"))
    return scaler, model, vectorizer

scaler, model, vectorizer = load_artifacts()

# =========================
# UTILITIES
# =========================
def detect_text_columns(df: pd.DataFrame) -> list:
    keywords = ["review", "comment", "feedback", "text", "message"]
    text_cols = [
        col for col in df.columns
        if df[col].dtype == object and any(k in col.lower() for k in keywords)
    ]
    return text_cols if text_cols else df.select_dtypes(include="object").columns.tolist()

def predict_batch(text_batch: pd.Series):
    processed = text_batch.fillna("").str.lower().str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
    features_sparse = vectorizer.transform(processed) 
    features_dense = features_sparse.toarray()
    features_scaled = scaler.transform(features_dense)
    preds = model.predict(features_scaled)
    probs = model.predict_proba(features_scaled)[:, 1]
    return preds, probs

# =========================
# HEADER
# =========================
st.markdown('<div class="brutal-card">', unsafe_allow_html=True)
st.markdown('<div class="brutal-title">⚡ Sentiment Analysis Engine</div>', unsafe_allow_html=True)
st.markdown("<p>Production-ready. Scalable. Batch-optimized.</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# SINGLE PREDICTION
# =========================
st.markdown('<div class="brutal-card">', unsafe_allow_html=True)
st.markdown('<div class="brutal-subtitle">Single Review Prediction</div>', unsafe_allow_html=True)

review_input = st.text_area("Enter review")

if st.button("Analyze 🚀", key="single_analyze"):
    if review_input.strip():
        with st.spinner("Analyzing..."):
            preds, probs = predict_batch(pd.Series([review_input]))
            sentiment = "Positive" if preds[0] == 1 else "Negative"
            confidence = round(float(probs[0]) * 100, 2)

        st.success(f"Sentiment: {sentiment}")
        st.info(f"Confidence: {confidence}%")
    else:
        st.error("Input required.")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# BULK PREDICTION (CHUNKED)
# =========================
st.markdown('<div class="brutal-card">', unsafe_allow_html=True)
st.markdown('<div class="brutal-subtitle">Bulk CSV Prediction</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV (100k+ supported)", type=["csv"])

if uploaded_file:
    try:
        df_preview = pd.read_csv(uploaded_file, nrows=5)
        text_cols = detect_text_columns(df_preview)

        if not text_cols:
            st.error("No valid text column detected.")
        else:
            selected_col = st.selectbox("Select text column", text_cols)

            if st.button("Run Bulk Prediction", key="bulk_analyze"):
                uploaded_file.seek(0)
                
                output_buffer = io.StringIO()
                chunk_size = 10000
                first_chunk = True
                
                # Counters for charts
                total_pos = 0
                total_neg = 0

                with st.spinner("Processing large dataset in chunks to preserve RAM..."):
                    for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
                        preds, probs = predict_batch(chunk[selected_col])
                        
                        # Tally counts for graphs
                        pos_count = np.sum(preds == 1)
                        total_pos += pos_count
                        total_neg += (len(preds) - pos_count)
                        
                        chunk["sentiment"] = ["Positive" if p == 1 else "Negative" for p in preds]
                        chunk["confidence_%"] = np.round(probs * 100, 2)
                        
                        chunk.to_csv(output_buffer, index=False, header=first_chunk)
                        first_chunk = False

                st.success("Batch processing complete.")
                
                # Render the Graphs Card
                st.markdown('<div class="brutal-card">', unsafe_allow_html=True)
                st.markdown('<div class="brutal-subtitle">Distribution Breakdown</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    chart_data = pd.DataFrame(
                        {"Count": [total_pos, total_neg]}, 
                        index=["Positive", "Negative"]
                    )
                    st.bar_chart(chart_data, color="#118AB2")

                with col2:
                    fig, ax = plt.subplots()
                    # Styling the pie chart to match Neubrutalism
                    fig.patch.set_facecolor('none')
                    ax.pie(
                        [total_pos, total_neg], 
                        labels=["Positive", "Negative"], 
                        autopct='%1.1f%%', 
                        startangle=90,
                        colors=["#118AB2", "#EF476F"],
                        wedgeprops={"edgecolor": "black", "linewidth": 2},
                        textprops={'color': "white", 'weight': 'bold'}
                    )
                    st.pyplot(fig)
                
                st.markdown('</div>', unsafe_allow_html=True)

                st.download_button(
                    label="Download Results CSV",
                    data=output_buffer.getvalue().encode("utf-8"),
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)