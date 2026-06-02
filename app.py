import os
import hashlib
import uuid
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
st.set_page_config(page_title="BizInsight AI", layout="wide")

from sklearn.feature_extraction.text import CountVectorizer
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import io
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from clustering.run_clustering import run_pipeline
from clustering.vectorize import load_model

from database import (
    initialize_database,
    insert_feedback,
    insert_feedback_bulk,
    fetch_feedback,
    fetch_all_feedback,
    fetch_all_users,
    clear_data,
    delete_user
)
from auth import is_logged_in, get_current_user, logout, show_auth_page, show_setup_wizard
from database import no_users_exist

# ─── Initialize DB ────────────────────────────────────────────────────────────

initialize_database()

# ─── Auth Gate ────────────────────────────────────────────────────────────────

if no_users_exist():
    show_setup_wizard()
    st.stop()

if not is_logged_in():
    show_auth_page()
    st.stop()

current_user = get_current_user()

# ─── OpenAI Client ───────────────────────────────────────────────────────────

@st.cache_resource
def get_ai_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

client = get_ai_client()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"👤 **{current_user['username']}**")
    st.caption(f"Role: `{current_user['role']}`")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()

# ─── Vader ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_vader_analyzer():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()

vader_analyzer = load_vader_analyzer()

# ─── Tabs ─────────────────────────────────────────────────────────────────────

if current_user["role"] == "admin":
    tabs = st.tabs(["📊 Dashboard", "🤖 AI Assistant", "📂 Data Upload", "⚙ Controls", "👑 Admin"])
else:
    tabs = st.tabs(["📊 Dashboard", "🤖 AI Assistant", "📂 Data Upload", "⚙ Controls"])

# ─── Helper Functions ─────────────────────────────────────────────────────────

def get_sentiment(text):
    return vader_analyzer.polarity_scores(text)['compound']

def clean_text_for_sentiment(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def ask_ai(question, reviews):
    context = "\n".join(reviews[:40])
    prompt = f"""You are a business intelligence assistant.

Customer reviews:
{context}

Question:
{question}
"""
    try:
        response = client.chat.completions.create(
            model="tngtech/deepseek-r1t2-chimera:free",
            messages=[
                {"role": "system", "content": "You provide business intelligence insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating AI response: {str(e)}"

def make_pdf(df, trend, keywords):
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("BizInsight AI Report", styles["Title"]))
    content.append(Paragraph("Generated: " + str(datetime.now().strftime("%Y-%m-%d %H:%M")), styles["Normal"]))
    content.append(Paragraph("Total Reviews: " + str(len(df)), styles["Normal"]))
    content.append(Paragraph("Positive: " + str((df["sentiment"] > 0).sum()), styles["Normal"]))
    content.append(Paragraph("Negative: " + str((df["sentiment"] < 0).sum()), styles["Normal"]))

    fig1, ax1 = plt.subplots()
    ax1.plot(trend.index, trend.values)
    ax1.set_title("Sentiment Trend")
    img1 = io.BytesIO()
    fig1.savefig(img1, format="png")
    plt.close(fig1)
    img1.seek(0)
    content.append(Image(img1, width=400, height=200))

    fig2, ax2 = plt.subplots()
    ax2.bar(["Positive", "Negative"], [(df["sentiment"] > 0).sum(), (df["sentiment"] < 0).sum()])
    ax2.set_title("Positive vs Negative")
    img2 = io.BytesIO()
    fig2.savefig(img2, format="png")
    plt.close(fig2)
    img2.seek(0)
    content.append(Image(img2, width=400, height=200))

    content.append(Paragraph("Top Keywords: " + ", ".join(keywords), styles["Normal"]))
    pdf.build(content)
    buffer.seek(0)
    return buffer.read()

# ================= DATA UPLOAD =================

with tabs[2]:
    st.subheader("📂 Upload Customer Reviews")
    uploaded_file = st.file_uploader("Upload CSV with review column", type="csv")

    if uploaded_file:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

        if st.session_state.get("last_upload_hash") != file_hash:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df, use_container_width=True)

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
                    reviews_data = list(zip(df["review"], df["sentiment"]))
                    insert_feedback_bulk(reviews_data, user_id=current_user["id"])
                    st.session_state["last_upload_hash"] = file_hash
                    st.success(f"{len(df)} feedback entries successfully added!")
        else:
            st.info("This file has already been uploaded in this session.")

# ─── Load Data ────────────────────────────────────────────────────────────────

data = fetch_feedback(user_id=current_user["id"])

df = pd.DataFrame(data, columns=["review", "sentiment", "date"])

if not df.empty:
    df["date"] = pd.to_datetime(df["date"])

    positive = (df["sentiment"] > 0).sum()
    negative = (df["sentiment"] < 0).sum()
    neutral  = (df["sentiment"] == 0).sum()
    total_reviews = len(df)

    # These are now safe from ZeroDivisionError because total_reviews > 0
    positive_percent = round((positive / total_reviews) * 100, 2)
    negative_percent = round((negative / total_reviews) * 100, 2)
    neutral_percent  = round((neutral  / total_reviews) * 100, 2)

    trend = df.groupby(df["date"].dt.date)["sentiment"].mean()

    reviews = df["review"].dropna()
    if reviews.empty or (reviews.apply(lambda x: isinstance(x, str)).all() and reviews.str.strip().eq("").all()):
        keywords = []
        keyword_counts = []
    else:
        vectorizer = CountVectorizer(stop_words="english", max_features=10)
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

    keyword_df = pd.DataFrame({"Keyword": keywords, "Frequency": keyword_counts})

    # ─── Dashboard Tab ────────────────────────────────────────────────────────

    with tabs[0]:
        st.subheader("📈 Business Health Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Reviews", total_reviews)
        c2.metric("Positive %", f"{positive_percent}%")
        c3.metric("Negative %", f"{negative_percent}%")
        c4.metric("Neutral %",  f"{neutral_percent}%")

        st.markdown("---")
        st.subheader("🔍 Smart Review Clustering")
        embedding_model = load_model()

        cluster_mode = st.radio(
            "Cluster by sentiment:",
            options=["Negative", "Positive"],
            horizontal=True,
            help="Negative: surfaces complaint themes. Positive: surfaces what customers love."
        )

        button_label = "Find Complaint Clusters" if cluster_mode == "Negative" else "Find Positive Themes"

        if st.button(button_label):
            if cluster_mode == "Negative":
                selected_reviews = df[df["sentiment"] < 0]["review"].tolist()
                spinner_text = "Clustering negative reviews..."
                empty_warning = f"Only {len(selected_reviews)} negative reviews. Need at least 10."
                success_label = "complaints"
                icon = "📌"
                mode_arg = "negative"
            else:
                selected_reviews = df[df["sentiment"] > 0]["review"].tolist()
                spinner_text = "Clustering positive reviews..."
                empty_warning = f"Only {len(selected_reviews)} positive reviews. Need at least 10."
                success_label = "positive themes"
                icon = "⭐"
                mode_arg = "positive"

            with st.spinner(spinner_text):
                if len(selected_reviews) < 10:
                    st.warning(empty_warning)
                else:
                    dynamic_min_topic_size = max(3, len(selected_reviews) // 5)
                    result = run_pipeline(
                        selected_reviews,
                        embedding_model,
                        min_topic_size=dynamic_min_topic_size,
                        similarity_threshold=0.4,
                        verbose=True,
                        mode=mode_arg
                    )
                    if result["success"]:
                        st.success(f"✅ Found {result['n_clusters']} {success_label} from {result['total_negative_reviews']} reviews")
                        for cluster in result["clusters"]:
                            with st.expander(f"{icon} {cluster['name']} ({cluster['percentage']:.1f}%) - {cluster['count']} reviews"):
                                st.write("**Example reviews:**")
                                for ex in cluster.get('example_reviews', [])[:3]:
                                    st.write(f"- \"{ex}\"")
                    else:
                        st.error(result["message"])

        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Customer Satisfaction Trend")
            st.area_chart(trend)
        with col2:
            fig3, ax3 = plt.subplots(figsize=(3.2, 3.2))
            ax3.pie(
                [positive, negative, neutral],
                labels=["Positive", "Negative", "Neutral"],
                autopct="%1.1f%%"
            )
            st.pyplot(fig3)
            plt.close(fig3)

        st.markdown("---")
        pdf_file = make_pdf(df, trend, list(keywords))
        st.download_button("Download PDF Report", pdf_file, file_name="report.pdf", mime="application/pdf")

        st.subheader("📊 Sentiment Score Distribution")
        col_small, _ = st.columns([1.5, 4])
        with col_small:
            fig2, ax2 = plt.subplots(figsize=(2.8, 2.1))
            ax2.hist(df["sentiment"], bins=10)
            ax2.set_xlabel("Score", fontsize=8)
            ax2.set_ylabel("Freq", fontsize=8)
            ax2.tick_params(axis='both', labelsize=7)
            st.pyplot(fig2)
            plt.close(fig2)

        st.markdown("---")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Feedback as CSV",
            data=csv_data,
            file_name="bizinsight_feedback.csv",
            mime="text/csv"
        )

        st.subheader("Top Customer Issues / Keywords")
        st.dataframe(keyword_df, use_container_width=True)

    # ─── AI Assistant Tab ─────────────────────────────────────────────────────

    with tabs[1]:
        st.subheader("🤖 AI Business Assistant")
        question = st.text_area("Ask a business insights question",
                                placeholder="Example: What are the major customer complaints?")
        if st.button("Generate AI Insight"):
            if client is None:
                st.warning("AI features unavailable because API key is missing.")
            elif question.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("Analyzing..."):
                    answer = ask_ai(question, df["review"].tolist())
                    st.success("AI Insight Generated")
                    st.write(answer)

    # ─── Controls Tab ─────────────────────────────────────────────────────────

    with tabs[3]:
        st.subheader("⚙ System Controls")
        st.warning("⚠️ Clearing data permanently deletes all your stored reviews. This cannot be undone.")

        if "confirm_clear" not in st.session_state:
            st.session_state["confirm_clear"] = False

        if not st.session_state["confirm_clear"]:
            if st.button("🗑️ Clear my feedback data"):
                st.session_state["confirm_clear"] = True
                st.rerun()
        else:
            st.error("Are you sure? All your reviews will be permanently deleted.")
            col1, col2 = st.columns(2)
            if col1.button("✅ Yes, delete everything"):
                clear_data(user_id=current_user["id"])
                st.session_state["confirm_clear"] = False
                st.session_state.pop("last_upload_hash", None)
                st.success("All your data has been removed.")
                st.rerun()
            if col2.button("❌ Cancel"):
                st.session_state["confirm_clear"] = False
                st.rerun()

else:
    st.info("Upload feedback to start building insights.")

# ─── Admin Tab ────────────────────────────────────────────────────────────────

if current_user["role"] == "admin":
    with tabs[4]:
        st.subheader("👑 Admin Panel")
        st.markdown("### Registered Users")
        users = fetch_all_users()

        if not users:
            st.info("No users found.")
        else:
            for user in users:
                user_id, username, role, created_at, review_count = user
                col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 1])
                col1.write(f"**{username}**")
                col2.write(f"`{role}`")
                col3.write(created_at)
                col4.write(f"{review_count} reviews")

                if user_id != current_user["id"]:
                    if col5.button("🗑️ Delete", key=f"del_user_{user_id}"):
                        delete_user(user_id)
                        st.success(f"User '{username}' and their data deleted.")
                        st.rerun()
                else:
                    col5.write("_(you)_")

        st.markdown("---")
        st.markdown("### All Feedback Across Users")
        all_feedback = fetch_all_feedback()

        if not all_feedback:
            st.info("No feedback in the system yet.")
        else:
            df_all = pd.DataFrame(all_feedback, columns=["review", "sentiment", "date", "uploaded_by"])
            st.dataframe(df_all, use_container_width=True)