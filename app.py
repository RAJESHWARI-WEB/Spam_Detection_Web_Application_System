import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Load model and vectorizer
model = pickle.load(open("model/scam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="AI Scam Detector", page_icon="🚨")

st.title("🚨 AI Scam & Fraud Detection System")
st.write("Enter suspicious message below to detect scam probability.")

user_input = st.text_area("✉ Enter Message Here")

if st.button("Analyze Message"):

    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized).max()

        st.subheader("🔎 Result:")

        if prediction == "general_scam":
            st.error(f"⚠ Scam Detected! (Confidence: {round(probability*100,2)}%)")
            st.write("🔐 Advice: Do NOT share OTP, bank details, or click unknown links.")
        else:
            st.success(f"✅ This message appears safe. (Confidence: {round(probability*100,2)}%)")