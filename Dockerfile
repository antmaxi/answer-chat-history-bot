# CPU-only torch: the CUDA wheel from PyPI is several GB and this bot embeds
# on CPU. Reinstall torch last so sentence-transformers cannot upgrade it.
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DB_PATH=/data/answerbot.db

RUN mkdir -p /data /opt/hf

COPY pyproject.toml .
COPY answerbot ./answerbot

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install . \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

# Default embed model, so the first `index` is not a surprise Hub download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

RUN useradd --create-home --uid 1000 answerbot \
    && chown -R answerbot:answerbot /app /data /opt/hf

# gosu is a late layer so a code/entrypoint change does not bust the torch cache.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN apt-get update \
    && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/* \
    && chmod +x /usr/local/bin/docker-entrypoint.sh \
    && gosu nobody true

VOLUME ["/data"]
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "answerbot.bot"]

