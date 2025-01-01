import streamlit as st
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

def load_model():
    try:
        with open("model.pickle", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found. Train the model first.")
        return None
    
classifier = load_model()

def classify_tweet(tweet):
    words = word_tokenize(tweet)
    features = {word: (word in words) for word in words}
    return classifier.classify(features)

st.title("Suicidal Tweet Detection")

input_tweet = st.text_area("Enter your tweet:")

if st.button("Detect"):
    if len(input_tweet.split()) < 5:
        st.warning("Tweet must contain at least 5 words.")
    elif classifier is None:
        st.error("Model is not loaded. Please ensure the model is trained and available!")
    else:
        category = classify_tweet(input_tweet)
        if category.strip() == "Potential Suicide post":
            st.error(f"Category: {category}", icon="🚨")
        else:
            st.success(f"Category: {category}", icon="✅")


