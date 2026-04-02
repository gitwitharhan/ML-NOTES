import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import re
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Spam Detective", page_icon="mail", layout="centered")

# --- APP TITLE & INFO ---
st.title("Spam Detective")
st.markdown("Enter an email below to see if it's **Spam** or **Ham** using our AI models.")

# --- HELPERS ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# --- LOAD MODELS ---
@st.cache_resource
def load_assets():
    # Check if models exist first
    lr_path = 'spam_lr_model.joblib'
    rnn_path = 'spam_rnn_model.h5'
    
    if not (os.path.exists(lr_path) and os.path.exists(rnn_path)):
        st.error("Models not found! Please run the `spam_detection_lab.ipynb` notebook first to train and save the models.")
        st.stop()
        
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    lr_model = joblib.load(lr_path)
    rnn_model = tf.keras.models.load_model(rnn_path)
    return embed_model, lr_model, rnn_model

try:
    embed_model, lr_model, rnn_model = load_assets()
except Exception as e:
    st.info("Note: You need to run the notebook and generate `spam_lr_model.joblib` and `spam_rnn_model.h5` before using this app.")
    st.stop()

# --- USER INPUT ---
user_input = st.text_area("Paste Email Content Here:", height=150, placeholder="e.g., Congratulations! You've won a $1000 gift card...")

if st.button("Analyze Email"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # 1. Preprocess
        cleaned = clean_text(user_input)
        
        # 2. Embed
        with st.spinner("Generating Embeddings..."):
            embedding = embed_model.encode([cleaned])
            
        # 3. Predict - Logistic Regression
        lr_pred = lr_model.predict(embedding)[0]
        
        # 4. Predict - RNN
        rnn_embed = embedding.reshape((1, 1, 384))
        rnn_prob = rnn_model.predict(rnn_embed)[0][0]
        rnn_pred = 1 if rnn_prob > 0.5 else 0
        
        # 5. Sentiment
        sentiment = TextBlob(user_input).sentiment.polarity
        
        # --- RESULTS UI ---
        st.divider()
        st.subheader("Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Logistic Regression", "SPAM" if lr_pred == 1 else "HAM")
            
        with col2:
            st.metric("RNN Model", "SPAM" if rnn_pred == 1 else "HAM")
            st.caption(f"Confidence: {rnn_prob:.2%}" if rnn_pred == 1 else f"Confidence: {1-rnn_prob:.2%}")
            
        with col3:
            sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
            st.metric("Sentiment", sentiment_label)
            st.caption(f"Score: {sentiment:.2f}")

        # Final Verdict
        if lr_pred == 1 or rnn_pred == 1:
            st.error("Warning: This email looks like SPAM!")
        else:
            st.success("This email seems safe (HAM).")

# --- FOOTER ---
st.divider()
st.caption("Built with Streamlit, Sentence Transformers, and Deep Learning.")
