# BizInsight AI - Dockerfile
# Builds a container image that runs the Streamlit app with all
# required dependencies (incl. NLTK/TextBlob corpora) baked in.

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies:
#  - build-essential: needed to build some ML packages (hdbscan, etc.) from source
#  - curl: used by the HEALTHCHECK below
# (must run as root, before we ever switch to the non-root user below)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# torch is pulled from PyPI's default index by `pip install -r requirements.txt`,
# which serves CUDA-enabled wheels on Linux (2GB+) -- unnecessary for this
# CPU-only Streamlit app and a major contributor to image size/build time.
# Installing the official CPU-only build first means the later
# `pip install -r requirements.txt` sees torch>=2.2.0 already satisfied
# (pip's default "only-if-needed" upgrade strategy) and leaves it alone.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies (still as root -- site-packages only need
# to be *readable* by appuser later, which the default 755/644 perms
# already allow, so no chown is needed for this layer).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK_DATA must be set BEFORE the downloader runs, otherwise the
# corpora land in the default location (e.g. /root/nltk_data) instead
# of the path below, and the app fails to find them at runtime and
# falls back to a slow/flaky download on first request.
# HF_HOME is set explicitly (rather than relying on the implicit
# ~/.cache/huggingface default) so the build-time download and the
# runtime lookup are guaranteed to use the same path.
ENV NLTK_DATA=/usr/local/share/nltk_data \
    HF_HOME=/usr/local/share/huggingface

# Create the non-root user now, and pre-create every directory the app
# writes to (app code, SQLite/Chroma data, NLTK/HF caches) with the
# correct ownership from the start -- all while they're still empty.
#
# Why now and not later: doing `useradd`+`mkdir`+`chown` *before*
# anything is downloaded into these paths means there's no existing
# file content for Docker's copy-on-write layer mechanism to duplicate.
# Running `chown -R` *after* ~600MB+ of NLTK/HF data already exists
# would force overlay2 to copy every one of those files into a new
# layer just to change ownership metadata, roughly doubling that
# portion of the image. Creating the dirs empty and owned correctly
# from the start avoids that entirely.
RUN useradd -u 10001 -m appuser \
    && mkdir -p /app /data /app/chroma_db "$NLTK_DATA" "$HF_HOME" \
    && chown -R appuser:appuser /app /data "$NLTK_DATA" "$HF_HOME"

# Everything from here on (model downloads, app code) is created
# directly as appuser, so it's correctly owned with zero extra chown.
USER appuser

# Pre-download everything the app needs to do inference offline:
#  - vader_lexicon / TextBlob corpora: used by sentiment.py / app.py
#  - SentenceTransformer models: used by clustering/vectorize.py and
#    sync_vectors.py / rag_api/config.py for embeddings
#  - CrossEncoder: used by rag_api/chains.py to rerank RAG results
# Baking these into the image means the first request doesn't have to
# pull ~hundreds of MB from the internet, and the app still works in
# network-restricted environments.
RUN python -m nltk.downloader vader_lexicon -d "$NLTK_DATA" \
    && python -m textblob.download_corpora \
    && python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-mpnet-base-v2'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy the application code, owned by appuser from the moment it lands
# in the image (no separate chown layer needed).
COPY --chown=appuser:appuser . .

# Symlink the SQLite database path into /data, which is mounted as a
# volume in docker-compose.yml. database.py does
# sqlite3.connect("bizinsight.db"), a path relative to /app, so
# /app/bizinsight.db now resolves through this symlink into
# /data/bizinsight.db.
#
# This avoids bind-mounting the .db file directly: a single-file bind
# mount doesn't give SQLite a real directory to create its sibling
# "-journal"/"-wal"/"-shm" files in, which can cause locking issues, and
# it requires the file to already exist on the host before the first
# `docker compose up` (otherwise Docker creates a directory in its place
# and the app crashes trying to open a directory as a database).
RUN ln -sf /data/bizinsight.db /app/bizinsight.db

# Streamlit's default port
EXPOSE 8501

# Container healthcheck against Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
