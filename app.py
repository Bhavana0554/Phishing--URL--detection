import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_parquet("Training.parquet")

# Preprocess
data['status'] = data['status'].str.lower()
data['status'] = data['status'].map({'legitimate': 0, 'phishing': 1})
data = data.dropna()

# Features and target
X = data.drop(['url', 'status'], axis=1)
y = data['status']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# ---------------- UI ----------------

st.title("🔐 Phishing URL Detector")
st.write("Enter a URL to check if it is safe or phishing")

# Input (IMPORTANT FIX)
url = st.text_input("Enter URL", placeholder="https://example.com")

# Feature extraction
def extract_features(url):
    return [
        len(url),
        url.count('.'),
        url.count('@'),
        1 if "https" in url else 0,
        url.count('-')
    ]

# Button
if st.button("Check URL"):
    if url.strip() == "":
        st.warning("⚠️ Please enter a URL")
    else:
        features = extract_features(url)

        # Match feature size
        features = features + [0] * (X.shape[1] - len(features))

        prediction = model.predict([features])

        if prediction[0] == 1:
            st.error("❌ Phishing URL Detected")
        else:
            st.success("✅ Safe URL")
