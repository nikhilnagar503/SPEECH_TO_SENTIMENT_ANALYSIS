
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re

# sklearn for preprocessing and machine learning models
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Random seeds for consistent results
seed = 1234
np.random.seed(seed)

# Load and preprocess data
nltk.download('stopwords')
nltk.download('punkt_tab')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def remove_usernames(text):
    """Removes mentions like '@username' from the text."""
    return re.sub(r'@\w+', '', text)

def tokenize(text):
    """Tokenizes the text into individual words."""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Removes stopwords from the tokenized words."""
    return [word for word in tokens if word.lower() not in stop_words and len(word) > 1]

def stem_words(tokens):
    """Applies stemming to each word in the tokenized list."""
    return [stemmer.stem(word) for word in tokens]

def preprocess_text(text):
    """Full preprocessing pipeline."""
    if not isinstance(text, str):
        return ""
    text = remove_usernames(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_words(tokens)
    return ' '.join(tokens)

# Load dataset
df = pd.read_csv('/content/Sentiment Analysis Dataset.csv',on_bad_lines='skip')

# Preprocess the text data
df = df[df['SentimentText'].apply(lambda x: isinstance(x, str))]
df['cleaned_text'] = df['SentimentText'].apply(preprocess_text)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(df['cleaned_text'],
                                                  df['Sentiment'],
                                                  test_size=0.2,
                                                  random_state=seed)

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_val = vectorizer.transform(X_val)

# Train a Multinomial Naive Bayes model
multi_nb = MultinomialNB()
multi_nb.fit(X_train, y_train)


# Evaluate the model
train_pred = multi_nb.predict(X_train)
val_pred = multi_nb.predict(X_val)
print(f'Accuracy on training set: {round(accuracy_score(y_train, train_pred) * 100, 4)}%')
print(f'Accuracy on validation set: {round(accuracy_score(y_val, val_pred) * 100, 4)}%')


pickle.dump(multi_nb, open('multi_nb_sentiment_model_score.data1', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer_score.pkl2', 'wb'))
