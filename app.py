import os
import tempfile
from pdf_generator import create_pdf
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
st.set_page_config(page_title="BizInsight AI", layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from database import insert_feedback, fetch_feedback, clear_data

from openai import OpenAI

# ---------- Chimera AI Client ----------

api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in Streamlit secrets or environment variables.")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

st.title("📊 BizInsight AI")
st.caption("AI-powered customer intelligence platform for business growth")

if "data_cleared" in st.session_state:
    st.success("All data removed successfully.")
    del st.session_state.data_cleared

tabs = st.tabs(["📊 Dashboard", "🤖 AI Assistant", "📂 Data Upload", "⚙ Controls"])

# ================= FUNCTIONS =================

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# ================= DATA UPLOAD =================

with tabs[2]:

    st.subheader("📂 Upload Customer Reviews")

    uploaded_file = st.file_uploader(
        "Upload CSV with review column",
        type="csv"
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.dataframe(df, width='stretch')

        if "review" not in df.columns:
            st.error("CSV must contain a 'review' column.")

        else:

            df = df.dropna(subset=["review"])

            df["review"] = df["review"].astype(str).str.strip()
            df = df[df["review"] != ""]

            if df.empty:
                st.warning("No valid reviews found after cleaning. Nothing to process.")

            else:

                df["sentiment"] = df["review"].apply(get_sentiment)

                inserted_count = 0

                for _, row in df.iterrows():
                    insert_feedback(row["review"], row["sentiment"])
                    inserted_count += 1

                st.success(f"{inserted_count} feedback entries successfully added!")

# ================= FETCH DATA =================

data = fetch_feedback()

if data:

    df = pd.DataFrame(
        data,
        columns=["review", "sentiment", "date"]
    )

    df["date"] = pd.to_datetime(df["date"])

    # Sentiment Counts
    positive = (df["sentiment"] > 0).sum()
    negative = (df["sentiment"] < 0).sum()
    neutral = (df["sentiment"] == 0).sum()

    total_reviews = len(df)

    # Sentiment Percentages
    positive_percent = round((positive / total_reviews) * 100, 2)
    negative_percent = round((negative / total_reviews) * 100, 2)
    neutral_percent = round((neutral / total_reviews) * 100, 2)

    # Trend Analysis
    trend = df.groupby(df["date"].dt.date)["sentiment"].mean()

    # Keyword Extraction

    reviews = df["review"].dropna()

    if reviews.empty or (
        reviews.apply(lambda x: isinstance(x, str)).all() and
        reviews.str.strip().eq("").all()
    ):
        keywords = []
        keyword_counts = []

    else:

        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=10
        )

        try:

            X = vectorizer.fit_transform(reviews)

            keywords = vectorizer.get_feature_names_out()
            keyword_counts = X.toarray().sum(axis=0)

        except ValueError as e:

            if "empty vocabulary" in str(e).lower():
                keywords = []
                keyword_counts = []

            else:
                raise

    keyword_df = pd.DataFrame({
        "Keyword": keywords,
        "Frequency": keyword_counts
    })

    # ================= DASHBOARD =================

    with tabs[0]:

        st.subheader("📈 Business Health Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Reviews", total_reviews)
        c2.metric("Positive %", f"{positive_percent}%")
        c3.metric("Negative %", f"{negative_percent}%")
        c4.metric("Neutral %", f"{neutral_percent}%")

        st.markdown("---")

        # Create chart first
        fig, ax = plt.subplots(figsize=(4,4))

        ax.bar(
            ["Positive", "Negative"],
            [positive, negative]
        )

        plt.tight_layout()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            chart_path = tmpfile.name

            fig.savefig(chart_path)

        if st.button("Generate PDF Report"):

            pdf_path = create_pdf(
                len(df),
                positive,
                negative,
                chart_path
            )

            with open(pdf_path, "rb") as pdf_file:

                st.download_button(
                    label="Download Report",
                    data=pdf_file,
                    file_name="bizinsight_report.pdf",
                    mime="application/pdf"
                )

        col1, col2 = st.columns([2,1])

        with col1:

            st.subheader("Customer Satisfaction Trend")
            st.area_chart(trend)

        with col2:

            fig3, ax3 = plt.subplots()

            ax3.pie(
                [positive, negative, neutral],
                labels=["Positive", "Negative", "Neutral"],
                autopct="%1.1f%%"
            )

            st.pyplot(fig3)
            plt.close(fig3)

            st.markdown("---")

        st.subheader("📊 Sentiment Score Distribution")

        col3, col4 = st.columns([1,2])

        with col3:

            fig2, ax2 = plt.subplots(figsize=(2.5,1.8))

            ax2.hist(df["sentiment"], bins=10)

            ax2.set_xlabel("Sentiment Score")
            ax2.set_ylabel("Frequency")

            st.pyplot(fig2)

        st.markdown("---")

        st.subheader("Top Customer Issues / Keywords")

        st.dataframe(keyword_df, use_container_width=True)

    # ================= CONTROLS =================

    with tabs[3]:

        st.subheader("⚙ System Controls")

        if st.button("🗑 Clear all stored feedback"):

            clear_data()

            st.session_state.data_cleared = True
            st.rerun()

        st.warning("This action cannot be undone.")

else:
    st.info("Upload feedback to start building insights.")