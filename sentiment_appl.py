
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


nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

r = sr.Recognizer()

def record_audio():
    try:
        with sr.Microphone() as source:
            st.info("Recording... Please speak.")
            r.adjust_for_ambient_noise(source, duration=0.5)
            r.pause_threshold = 3.0  
            audio = r.listen(source)
            st.success("Audio captured! Analyzing...")
            text = r.recognize_google(audio)
            return text.lower()
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
        return None
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return None

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

model_path = r'C:\\Users\\nagar\\OneDrive\\Desktop\\all\\CA1_project\\multi_nb_sentiment_model.data1'

model_path = 'C:/Users/heman/OneDrive/Documents/multi_nb_sentiment_model.data1'
if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))
else:
    st.error("Model file not found! Please train the model first.")

vectorizer_path = r'C:\\Users\\nagar\\OneDrive\\Desktop\\all\\CA1_project\\tfidf_vectorizer.pkl2'


if os.path.exists(vectorizer_path):
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))
else:
    st.error("Vectorizer file not found! Please train the model first.")

st.title("🎙 Speech-to-Text Sentiment Analysis 🌐")
st.markdown("Analyze sentiment from your voice input using Naive Bayes with emojis")

if st.button("Start Recording"):
    captured_text = record_audio()
    if captured_text:
        st.write(f"*You said:* {captured_text}")

        cleaned_input = preprocess_text(captured_text)
        input_vector = vectorizer.transform([cleaned_input])

        prediction = model.predict(input_vector)

        sentiment_map = {0: 'Negative 😢', 1: 'Positive 😊'}
        sentiment = sentiment_map.get(int(prediction[0]), "Unknown Sentiment")
        
        st.markdown(f"### Sentiment: {sentiment}")

else:
    st.info("Click the button to start recording your voice.")
