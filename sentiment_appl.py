# Import necessary libraries
import streamlit as st
import speech_recognition as sr
import os
import pickle
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# NLTK downloads (uncomment if you haven't downloaded them already)
nltk.download('stopwords')
nltk.download('punkt_tab')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# Initialize speech recognizer
r = sr.Recognizer()

# Function to record audio and convert to text
def record_audio():
    try:
        with sr.Microphone() as source:
            st.info("Recording... Please speak.")
            r.adjust_for_ambient_noise(source, duration=0.5)
            r.pause_threshold = 3.0  # Adjusted pause threshold for longer pauses
            audio = r.listen(source)
            st.success("Audio captured! Analyzing...")
            # Convert speech to text
            text = r.recognize_google(audio)
            return text.lower()
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
        return None
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return None

# Preprocessing functions
def remove_usernames(text):
    return re.sub(r'@\w+', '', text)

def tokenize(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words and len(word) > 1]

def stem_words(tokens):
    return [stemmer.stem(word) for word in tokens]

def preprocess_text(text):
    text = remove_usernames(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_words(tokens)
    return ' '.join(tokens)

# Load the trained Naive Bayes model
# Option 1: Use raw string
model_path = r'C:\\Users\\nagar\\OneDrive\\Desktop\\all\\CA1_project\\multi_nb_sentiment_model.data1'

# Option 2: Use forward slashes
model_path = 'C:/Users/heman/OneDrive/Documents/multi_nb_sentiment_model.data1'
if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))
else:
    st.error("Model file not found! Please train the model first.")

# Load the vectorizer
# Use raw string notation for Windows path
vectorizer_path = r'C:\\Users\\nagar\\OneDrive\\Desktop\\all\\CA1_project\\tfidf_vectorizer.pkl2'

# Or use forward slashes which work universally

if os.path.exists(vectorizer_path):
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))
else:
    st.error("Vectorizer file not found! Please train the model first.")

# Streamlit App
st.title("🎙 Speech-to-Text Sentiment Analysis 🌐")
st.markdown("Analyze sentiment from your voice input using Naive Bayes with emojis")

# Button to start recording
if st.button("Start Recording"):
    # Record and convert audio to text
    captured_text = record_audio()
    if captured_text:
        st.write(f"*You said:* {captured_text}")

        # Preprocess the captured text
        cleaned_input = preprocess_text(captured_text)
        input_vector = vectorizer.transform([cleaned_input])

        # Predict sentiment using the pre-trained model
        prediction = model.predict(input_vector)

        # Map the sentiment prediction to emojis
        sentiment_map = {0: 'Negative 😢', 1: 'Positive 😊'}
        sentiment = sentiment_map.get(int(prediction[0]), "Unknown Sentiment")
        
        # Display the sentiment
        st.markdown(f"### Sentiment: {sentiment}")

else:
    st.info("Click the button to start recording your voice.")
