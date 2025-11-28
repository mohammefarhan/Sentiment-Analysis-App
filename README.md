# 📝 Sentiment Analysis App

A machine learning web application that predicts whether a given text is **Positive**, **Negative**, or **Neutral**.  
Built using **Python, NLP techniques, Machine Learning, and Streamlit**.

---

## 🚀 Overview

Sentiment analysis helps understand the emotional tone behind text such as product reviews, tweets, comments, and feedback.

This app:
- Takes user input text  
- Processes it using NLP  
- Predicts sentiment instantly  
- Provides a clean and simple UI  

---

## 🧠 How It Works

### **1. Data Preprocessing**
- Tokenization  
- Text cleaning (stopwords, punctuation, lowercase)  
- Lemmatization  
- Vectorization (TF-IDF / CountVectorizer)

### **2. Model Building**
Models used (you can modify based on your project):
- Logistic Regression  
- Naive Bayes  
- SVM  
- LSTM (Optional – Deep Learning version)

Final model used: **Logistic Regression / SVM / LSTM** (choose the one you used)

### **3. Prediction Pipeline**
- User enters text  
- Text is cleaned & vectorized  
- Model predicts sentiment  
- Output displayed in UI

---

## 🖥️ Features

✔ Predicts **Positive / Negative / Neutral**  
✔ Clean Streamlit interface  
✔ Real-time text classification  
✔ NLP preprocessing pipeline  
✔ Supports multiple ML/NLP models  
✔ Easy to run and modify  

---

## 📦 Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/mohammefarhan/sentiment-analysis-app.git
cd sentiment-analysis-app

pip install -r requirements.txt

streamlit run app.py
