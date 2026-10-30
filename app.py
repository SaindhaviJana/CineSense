import streamlit as st
import joblib
import re
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

# PAGE CONFIGURATION
st.set_page_config(
    page_title="CineSense | Movie Sentiment AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PROJECT PATHS
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "Models"
DB_PATH = BASE_DIR / "cinesense.db"

# DATABASE FUNCTIONS
def get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

def save_review(review, sentiment, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO reviews (review, sentiment, confidence, created_at)
            VALUES (?, ?, ?, ?)
        """, (review, sentiment, confidence, timestamp))
        conn.commit()

def get_all_reviews():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM reviews ORDER BY id DESC", conn
        )

def delete_all_reviews():
    with get_connection() as conn:
        conn.execute("DELETE FROM reviews")
        conn.commit()

initialize_database()

# LOAD TRAINED MODEL
@st.cache_resource
def load_models():
    model = joblib.load(MODEL_DIR / "sentiment_model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer

model, tfidf = load_models()

# TEXT PREPROCESSING
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# SESSION STATE
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "review_input" not in st.session_state:
    st.session_state.review_input = ""

def set_example(review):
    st.session_state.review_input = review

def clear_review():
    st.session_state.review_input = ""
    st.session_state.latest_result = None

# CUSTOM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
    --background:#09090f; --surface:#14141f; --border:#302e42;
    --purple:#a78bfa; --text:#f5f3ff; --muted:#a1a1b5;
}
.stApp {background:var(--background);color:var(--text);font-family:'DM Sans',sans-serif;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;color:var(--text)!important;}
.block-container {max-width:1450px;padding:2.5rem 3rem 4rem;}
section[data-testid="stSidebar"] {background:#11111b;border-right:1px solid var(--border);}
section[data-testid="stSidebar"] .block-container {padding:2rem 1.2rem;}
.sidebar-brand {font-family:'Space Grotesk',sans-serif;font-size:29px;font-weight:700;color:white;}
.sidebar-brand span,.purple {color:var(--purple);}
.sidebar-caption {color:var(--muted);font-size:12px;margin-top:5px;margin-bottom:25px;}
.page-label {color:var(--purple);font-size:12px;font-weight:700;letter-spacing:2px;margin-bottom:12px;}
.hero {padding:30px 0 20px;}
.hero-title {font-family:'Space Grotesk',sans-serif;font-size:clamp(36px,5vw,54px);font-weight:700;letter-spacing:-2px;}
.hero-subtitle {color:var(--muted);font-size:16px;margin-top:8px;}
.welcome-card {background:linear-gradient(120deg,#191426,#171322);border:1px solid #42335d;border-radius:18px;padding:25px;margin:15px 0 25px;}
.welcome-title {font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:600;color:white;}
.welcome-description {color:#b9b3c9;font-size:14px;margin-top:8px;line-height:1.7;}
.result-positive {background:linear-gradient(120deg,#10251d,#14231e);border:1px solid #276747;border-radius:16px;padding:22px;min-height:125px;}
.result-negative {background:linear-gradient(120deg,#2a151c,#24141b);border:1px solid #783c4d;border-radius:16px;padding:22px;min-height:125px;}
.result-title-positive {color:#4ade80;font-size:22px;font-weight:700;}
.result-title-negative {color:#fb7185;font-size:22px;font-weight:700;}
.result-description {color:#d4d4df;font-size:13px;margin-top:9px;}
div[data-testid="stTextArea"] textarea {background:var(--surface)!important;color:white!important;border:1px solid var(--border)!important;border-radius:13px!important;font-size:15px!important;padding:16px!important;}
div[data-testid="stTextArea"] textarea:focus {border-color:var(--purple)!important;box-shadow:0 0 0 1px var(--purple)!important;}
div.stButton>button {border-radius:10px;min-height:43px;font-weight:600;transition:0.2s ease;}
div.stButton>button[kind="primary"] {background:linear-gradient(100deg,#7c3aed,#a78bfa);color:white;border:none;}
div.stButton>button[kind="primary"]:hover {background:linear-gradient(100deg,#6d28d9,#8b5cf6);color:white;border:none;}
div[data-testid="stMetric"] {background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px;}
div[data-testid="stExpander"] {background:var(--surface);border:1px solid var(--border);border-radius:12px;}
hr {border-color:var(--border);margin:28px 0;}
.footer {text-align:center;color:#77778c;font-size:12px;padding-top:45px;line-height:1.8;}
.stProgress>div>div>div {background-color:var(--purple);}
</style>
""", unsafe_allow_html=True)

# SHARED FOOTER
def show_footer():
    st.markdown("""
    <div class="footer">
        CineSense · AI-Powered Movie Review Sentiment Analysis<br>
        Built using Python, Natural Language Processing and Machine Learning
    </div>
    """, unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">Cine<span>Sense</span> 🎬</div>
    <div class="sidebar-caption">AI-powered movie review analysis</div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("NAVIGATION", ["Home", "Analytics", "History", "About"],
                    label_visibility="visible")
    st.divider()
    st.caption("MODEL INFORMATION")
    st.caption("Algorithm: Logistic Regression")
    st.caption("Feature Extraction: TF-IDF")
    st.caption("Dataset: Movie Reviews")
    st.divider()
    st.caption("Storage: SQLite Database")
    st.caption("Built with Python and Streamlit")

# HOME PAGE
if page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Cine<span class="purple">Sense</span> 🎬</div>
        <div class="hero-subtitle">Understand the emotion behind every movie review.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">Every review tells a story.</div>
        <div class="welcome-description">
            Discover the sentiment hidden in your movie reviews.
            Our machine learning model analyzes your text and predicts
            whether the overall sentiment is positive or negative.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page-label">01 / SENTIMENT ANALYZER</div>',
                unsafe_allow_html=True)
    st.subheader("What's your take on the movie?")
    st.caption("Write your own review or try one of the examples below.")

    example1, example2, example3 = st.columns(3)
    with example1:
        st.button("😊 Positive example", use_container_width=True,
                  on_click=set_example,
                  args=("An absolutely brilliant movie! The acting was excellent and the story was wonderful.",))
    with example2:
        st.button("☹️ Negative example", use_container_width=True,
                  on_click=set_example,
                  args=("This was the worst movie I have ever watched. Terrible acting and boring storyline.",))
    with example3:
        st.button("⚖️ Mixed example", use_container_width=True,
                  on_click=set_example,
                  args=("Not as great as the hype, but still a good movie.",))

    review = st.text_area("Movie review",
                          placeholder="Write or paste your movie review here...",
                          height=170, label_visibility="collapsed",
                          key="review_input")

    analyze_col, clear_col = st.columns([1, 5])
    with analyze_col:
        analyze = st.button("✦ Analyze Review", type="primary",
                            use_container_width=True)
    with clear_col:
        st.button("Clear", on_click=clear_review)

    if analyze:
        if not review.strip():
            st.warning("Please enter a movie review.")
        else:
            cleaned = clean_text(review)
            if not cleaned:
                st.warning("Please enter readable text.")
            else:
                vectorized = tfidf.transform([cleaned])
                prediction = model.predict(vectorized)[0]
                probabilities = model.predict_proba(vectorized)[0]
                confidence = float(max(probabilities) * 100)
                save_review(review, prediction, confidence)
                st.session_state.latest_result = {
                    "review": review,
                    "sentiment": prediction,
                    "confidence": confidence
                }
                st.success("Review analyzed and saved successfully!")

    result = st.session_state.latest_result
    if result:
        st.divider()
        st.markdown('<div class="page-label">02 / ANALYSIS RESULT</div>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if result["sentiment"] == "positive":
                st.markdown("""
                <div class="result-positive">
                    <div class="result-title-positive">😊 Positive Sentiment</div>
                    <div class="result-description">Your review expresses an overall positive sentiment.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-negative">
                    <div class="result-title-negative">☹️ Negative Sentiment</div>
                    <div class="result-description">Your review expresses an overall negative sentiment.</div>
                </div>
                """, unsafe_allow_html=True)
        with col2:
            st.metric("Prediction Confidence", f'{result["confidence"]:.2f}%')
            st.metric("Review Length", f'{len(result["review"].split())} words')

        st.divider()
        st.subheader("Model Confidence")
        gauge_col, explanation_col = st.columns([1, 1])
        with gauge_col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=result["confidence"],
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#a78bfa"},
                    "bgcolor": "#191927", "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "#252331"},
                        {"range": [50, 100], "color": "#302744"}
                    ]
                }
            ))
            fig.update_layout(height=240, paper_bgcolor="#09090f",
                              font_color="#f5f3ff",
                              margin=dict(l=15, r=15, t=15, b=5))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        with explanation_col:
            st.markdown("#### Understanding the result")
            st.write("The model uses the words and patterns in your review to estimate its sentiment.")
            st.write("Confidence represents the model's predicted probability. It is not a guarantee that the prediction is correct.")

    show_footer()

# ANALYTICS PAGE
elif page == "Analytics":
    st.markdown('<div class="page-label">CINESENSE / INSIGHTS</div>',
                unsafe_allow_html=True)
    st.title("Analytics")
    st.caption("Explore sentiment patterns in your analyzed reviews.")
    history = get_all_reviews()

    if history.empty:
        st.info("No reviews analyzed yet. Visit Home and analyze some reviews.")
    else:
        total = len(history)
        positive = int((history["sentiment"] == "positive").sum())
        negative = total - positive
        positive_percentage = positive / total * 100
        negative_percentage = negative / total * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", total)
        col2.metric("Positive Reviews", positive)
        col3.metric("Negative Reviews", negative)
        st.divider()
        st.subheader("Sentiment Distribution")
        chart_col, info_col = st.columns([1.2, 1])

        with chart_col:
            fig = go.Figure(data=[go.Pie(
                labels=["Positive", "Negative"],
                values=[positive, negative], hole=0.68,
                marker=dict(colors=["#8b5cf6", "#fb7185"],
                            line=dict(color="#09090f", width=3)),
                textinfo="percent",
                hovertemplate="%{label}: %{value}<extra></extra>"
            )])
            fig.update_layout(paper_bgcolor="#09090f",
                              plot_bgcolor="#09090f",
                              font_color="#f5f3ff", height=320,
                              showlegend=True,
                              legend=dict(orientation="h", y=-0.08),
                              margin=dict(t=15, b=25, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

        with info_col:
            st.markdown("#### Sentiment Breakdown")
            st.metric("Positive", f"{positive_percentage:.1f}%")
            st.progress(positive / total)
            st.metric("Negative", f"{negative_percentage:.1f}%")
            st.progress(negative / total)

        st.caption("Statistics are calculated from all reviews saved in the database.")

    show_footer()

# HISTORY PAGE
elif page == "History":
    st.markdown('<div class="page-label">CINESENSE / ACTIVITY</div>',
                unsafe_allow_html=True)
    st.title("Review History")
    st.caption("Your previously analyzed movie reviews.")
    history = get_all_reviews()

    if history.empty:
        st.info("Your review history is currently empty.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.metric("Total Reviews Analyzed", len(history))
        with col2:
            st.write("")
            if st.button("Clear History"):
                delete_all_reviews()
                st.session_state.latest_result = None
                st.success("Review history cleared.")
                st.rerun()

        # CSV EXPORT
        st.divider()
        st.subheader("Export Your Reviews")
        st.caption("Download your saved review text, predictions, confidence scores and timestamps.")

        export_df = history[
            ["id", "review", "sentiment", "confidence", "created_at"]
        ].copy()
        export_df = export_df.rename(columns={
            "id": "Review ID",
            "review": "Review Text",
            "sentiment": "Predicted Sentiment",
            "confidence": "Prediction Confidence (%)",
            "created_at": "Date and Time"
        })
        csv_data = export_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="⬇️ Download Review History (CSV)",
            data=csv_data,
            file_name="CineSense_Review_History.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()
        for _, item in history.iterrows():
            sentiment = item["sentiment"].capitalize()
            with st.expander(
                f'Review #{item["id"]} · {sentiment} · {item["created_at"]}'
            ):
                st.write(item["review"])
                if item["sentiment"] == "positive":
                    st.success("Positive Sentiment")
                else:
                    st.error("Negative Sentiment")
                st.metric("Prediction Confidence",
                          f'{item["confidence"]:.2f}%')

    show_footer()

# ABOUT PAGE
elif page == "About":
    st.markdown('<div class="page-label">CINESENSE / PROJECT DETAILS</div>',
                unsafe_allow_html=True)
    st.title("About CineSense")
    st.write("""
    CineSense is an AI-powered movie review sentiment analysis
    application developed using Natural Language Processing
    and Machine Learning.

    It classifies textual movie reviews into positive or negative
    sentiment categories.
    """)
    st.divider()
    st.subheader("How CineSense Works")

    steps = [
        ("01", "Review Input", "The user submits a movie review."),
        ("02", "Text Preprocessing", "The review is cleaned and normalized."),
        ("03", "Feature Extraction", "TF-IDF converts text into numerical features."),
        ("04", "Sentiment Prediction", "Logistic Regression predicts sentiment."),
        ("05", "Result Visualization", "The application displays the prediction and confidence."),
        ("06", "Database Storage", "SQLite stores the analyzed review and its result.")
    ]
    for number, title, description in steps:
        with st.container(border=True):
            st.markdown(f"**{number} / {title}**")
            st.caption(description)

    st.divider()
    st.subheader("Technology Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Programming and Application**
        - Python
        - Streamlit
        - Pandas
        - Plotly
        """)
    with col2:
        st.markdown("""
        **Machine Learning and Storage**
        - Scikit-learn
        - TF-IDF Vectorization
        - Logistic Regression
        - SQLite
        """)

    st.divider()
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Logistic Regression", "88.46%")
    col2.metric("Naive Bayes", "86.44%")
    st.caption("Accuracy figures are from the held-out test dataset used during model evaluation.")
    show_footer()
