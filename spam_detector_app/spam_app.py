import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- Streamlit page config ---
st.set_page_config(page_title="Spam Detector AI", page_icon="📩", layout="centered")

# --- Clean background CSS ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8ff;
    }
    .main-overlay {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        color: #222;
    }
    .title {
        font-size: 2.5rem;
        text-align: center;
        color: #1a1a1a;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #777;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Dummy training data ---
data = {
    "label": ["ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam"],
    "message": [
        "Hi, how are you?",
        "Congratulations! You've won a free ticket.",
        "I'll call you later.",
        "WINNER! Claim your prize now.",
        "Are we still meeting today?",
        "Free entry in a weekly competition!",
        "Can we catch up tomorrow?",
        "URGENT! You have won a 1 week FREE membership!",
        "What time is dinner tonight?",
        "You’ve been selected for a cash reward!"
    ]
}
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']
model = MultinomialNB()
model.fit(X, y)

# --- Session state ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- App layout ---

st.markdown("<h1 class='title'>📩 Spam Detector AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter a message to check for spam</p>", unsafe_allow_html=True)

user_input = st.text_area("✍️ Your Message", height=150)

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        proba = model.predict_proba(input_vec)[0]
        confidence = round(np.max(proba) * 100, 2)

        # Display prediction
        if prediction == "spam":
            st.error(f"🚨 SPAM Detected with {confidence}% confidence.")
        else:
            st.success(f"✅ Message is safe with {confidence}% confidence.")

        # Save to history
        st.session_state.history.append({
            "Message": user_input,
            "Prediction": prediction,
            "Confidence": f"{confidence}%"
        })

        # Show chart
        fig = px.pie(
            values=[proba[0], proba[1]],
            names=["Ham", "Spam"],
            title="📊 Confidence Distribution",
            color_discrete_sequence=["#00cc96", "#ff4b4b"]
        )
        st.plotly_chart(fig)

st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction history ---
if st.session_state.history:
    st.markdown("### 🕘 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df[::-1], use_container_width=True)

# --- Footer ---
st.markdown("<div class='footer'>Built with ❤️ using Streamlit & Plotly</div>", unsafe_allow_html=True)
