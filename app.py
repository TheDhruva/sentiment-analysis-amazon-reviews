import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Enable Dark Theme
st.set_page_config(page_title="Sentiment Analysis", page_icon="📝", layout="centered")

# Load models and scaler
scaler = pickle.load(open('Models/scaler.pkl', 'rb'))
xgb_model = pickle.load(open('Models/model_xgb.pkl', 'rb'))
vectorizer = pickle.load(open('Models/CountVectorizer.pkl', 'rb'))

# Text preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    else:
        text = ''
    return text

# Interface
st.title('🛒 Amazon Review Sentiment Analysis')

# Single Review Sentiment Prediction
st.header('🌟 Enter a review')
review_input = st.text_area('Enter your review here')

if st.button('Analyze Sentiment 🚀'):
    if review_input:
        processed_review = preprocess_text(review_input)
        review_features = vectorizer.transform([processed_review]).toarray()
        review_features_scaled = scaler.transform(review_features)
        prediction = xgb_model.predict(review_features_scaled)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        
        if sentiment == 'Positive':
            st.success('The sentiment is positive! 🎉')
        else:
            st.error('The sentiment is negative. 😞')
    else:
        st.warning('Please enter a review!')

st.markdown("---")
st.markdown("<h4 style='text-align: center;'>⬇️ OR ⬇️</h2>", unsafe_allow_html=True)
st.markdown("---")

# CSV Upload and Bulk Sentiment Analysis
st.header('📂 Upload CSV for Bulk Sentiment Analysis')

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'review' in df.columns:
        df['review'] = df['review'].fillna('')
        df['processed_review'] = df['review'].apply(preprocess_text)
        review_features = vectorizer.transform(df['processed_review']).toarray()
        review_features_scaled = scaler.transform(review_features)
        predictions = xgb_model.predict(review_features_scaled)
        df['sentiment'] = ['Positive' if pred == 1 else 'Negative' for pred in predictions]
        st.write(df[['review', 'sentiment']].head())
        sentiment_counts = df['sentiment'].value_counts()

        # Two-column layout for charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Bar Chart')
            st.bar_chart(sentiment_counts)

        with col2:
            st.subheader('Pie Chart')
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
            st.pyplot(fig)
    else:
        st.error("CSV must have a 'review' column")

st.markdown("---")
st.write("Made with 💻 by YourName")
