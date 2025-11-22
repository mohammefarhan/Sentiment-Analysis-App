import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import re
import nltk
from nltk.corpus import stopwords

# Load model + tokenizer
model = tf.keras.models.load_model("sentiment_model_small.h5")
tokenizer = load("tokenizer.joblib")

max_length = 120  # Must match training
labels = ["Negative", "Neutral", "Positive"]

# Preprocessing
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def predict_sentiment(text):
    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_length, truncating='post')

    pred = model.predict(padded)[0]

    sentiment = labels[pred.argmax()]
    confidence = round(float(max(pred)) * 100, 2)

    return sentiment, confidence, pred

# Streamlit UI
st.title(" Sentiment Analysis App")

text = st.text_area("Enter text:")

if st.button("Predict"):
    if text.strip():
        sentiment, confidence, raw = predict_sentiment(text)

        st.success(f"Prediction: **{sentiment}**")
        
    else:
        st.warning("Please enter text.")
st.write("----")
st.caption("Made By Farhan Thanks For Using This App.")