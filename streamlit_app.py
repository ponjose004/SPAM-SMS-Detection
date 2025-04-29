import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

# Load model and vectorizer
model = pickle.load(open('spam_detector.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocessing function
def preprocess_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    return ' '.join(review)

# Streamlit UI
st.set_page_config(page_title="SMS Spam Detection", page_icon="📱")
st.title("SMS Spam Detection")
st.write("Enter a message below to check if it's **Spam** or **Not Spam**, with model confidence!")

user_input = st.text_area("✉️ Enter your SMS message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter a message!")
    else:
        cleaned_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input]).toarray()
        
        # Get prediction and probability
        prediction = model.predict(vectorized_input)[0]
        prediction_proba = model.predict_proba(vectorized_input)[0]
        
        spam_confidence = prediction_proba[1]  # Probability of being spam
        ham_confidence = prediction_proba[0]   # Probability of being not spam

        if prediction == 1:
            st.error(f"🚨 Spam Message Detected!\n\nConfidence: **{spam_confidence * 100:.2f}%**")
        else:
            st.success(f"✅ Not a Spam Message!\n\nConfidence: **{ham_confidence * 100:.2f}%**")
