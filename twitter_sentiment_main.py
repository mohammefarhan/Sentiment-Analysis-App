import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Load dataset
data = pd.read_csv('Twitter_Data.csv')

# Preprocessing
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

data['clean_text'] = data['text'].astype(str).apply(clean_text)

# Labels
y = data['sentiment'].replace({'negative': 0, 'neutral': 1, 'positive': 2})
X = data['clean_text']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = Tokenizer(num_words=15000)
tokenizer.fit_on_texts(X_train)

max_length = 120
X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_length)
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_length)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# One-hot encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=3)

# ---------------- SMALLER MODEL -------------------
model = tf.keras.Sequential([
    Embedding(input_dim=15000, output_dim=64, input_length=max_length),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train
model.fit(
    X_train_pad,
    y_train_onehot,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_pad, y_test_onehot),
    class_weight=class_weights,
    callbacks=[callback]
)

# Save
model.save("sentiment_model_small.h5")
dump(tokenizer, "tokenizer.joblib")

print("Smaller model training completed!")
