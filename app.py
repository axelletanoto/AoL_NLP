import streamlit as st
import pickle
import pandas as pd
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist


def load_model():
    try:
        with open("best_model_rf.pickle", "rb") as file:
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

            st.write("### Words of Encouragement:")
            st.write("*You are not alone. There is always hope, and someone is ready to listen.*")
            st.write("*Reaching out for help is a sign of strength, not weakness.*")
            st.write("*We are not against you, we care about you.*")

            st.write("### Call Help:")
            st.write("- **Suicide Prevention Hotline (Indonesia)**: Call **119 ext. 8**")
            st.write("- **Into the Light Indonesia**: [Visit Website](https://intothelightid.org)")
            st.write("- **Save Yourselves Indonesia**: [Visit Instagram](https://www.instagram.com/saveyourselves.id)")
            st.write("- **Sehat Jiwa**: [Visit Website](https://sehatjiwa.kemkes.go.id)")
            
            st.write("### Immediate Action:")
            st.write("If you're in immediate danger or need urgent help, please contact the nearest mental health professional or call the hotline.")
        else:
            st.success(f"Category: {category}", icon="✅")


