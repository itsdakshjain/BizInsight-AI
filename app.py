import streamlit as st
st.set_page_config(page_title="BizInsight AI", layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from database import insert_feedback, fetch_feedback, clear_data

st.title("📊 BizInsight AI")
st.caption("AI-powered customer intelligence platform for business growth")

# Tabs
tabs = st.tabs(["📊 Dashboard", "📂 Data Upload", "⚙ Controls"])


# ================= FUNCTIONS =================

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity


# ================= DATA UPLOAD =================

with tabs[1]:

    st.subheader("📂 Upload Customer Reviews")

    uploaded_file = st.file_uploader(
        "Upload CSV with review column",
        type="csv"
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        # Validation
        if "review" not in df.columns:
            st.error("CSV file must contain a 'review' column.")

        elif df.empty:
            st.error("Uploaded CSV is empty.")

        else:

            st.dataframe(df, use_container_width=True)

            # Sentiment Analysis
            df["sentiment"] = df["review"].astype(str).apply(get_sentiment)

            # Store in database
            for _, row in df.iterrows():
                insert_feedback(row["review"], row["sentiment"])

            st.success("Feedback successfully added!")


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
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=10
    )

    X = vectorizer.fit_transform(df["review"])

    keywords = vectorizer.get_feature_names_out()

    keyword_counts = X.toarray().sum(axis=0)

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

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Customer Satisfaction Trend")
            st.area_chart(trend)

        with col2:

            fig, ax = plt.subplots()

            ax.pie(
                [positive, negative, neutral],
                labels=["Positive", "Negative", "Neutral"],
                autopct="%1.1f%%"
            )

            st.pyplot(fig)

        st.markdown("---")

        st.subheader("📊 Sentiment Score Distribution")
        col3,col4=st.columns([1,2])
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

    with tabs[2]:

        st.subheader("⚙ System Controls")

        if st.button("🗑 Clear all stored feedback"):

            clear_data()

            st.success("All data removed successfully.")

        st.warning("This action cannot be undone.")

else:
    st.info("Upload feedback to start building insights.")