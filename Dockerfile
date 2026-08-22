# syntax=docker/dockerfile:1
# CPU-only torch: the CUDA wheel from PyPI is several GB and this bot embeds
# on CPU. Reinstall torch last so sentence-transformers cannot upgrade it.
#
# Cache-first layer order: apt, wheels, and the baked embed model sit above
# application code, so a Python edit does not reinstall torch or re-download
# the Hub snapshot.
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 gosu \
    && rm -rf /var/lib/apt/lists/* \
    && gosu nobody true

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DB_PATH=/data/answerbot.db

RUN mkdir -p /data /opt/hf \
    && useradd --create-home --uid 1000 answerbot \
    && chown answerbot:answerbot /data

# Dependency metadata only — the package itself is copied after the model
# download so source edits do not bust the wheel or Hub layers.
COPY pyproject.toml .

# cursor-sdk is an optional extra for venv installs, but Docker always includes
# it so LLM_PROVIDER=cursor works without a custom image. The local agent
# runtime is the bundled cursor-sdk-bridge binary, not the Cursor IDE.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -c "import subprocess, sys, tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; deps = p['dependencies'] + p.get('optional-dependencies', {}).get('cursor', []); subprocess.check_call([sys.executable, '-m', 'pip', 'install', *deps])" \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -c "from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions" \
    && cursor-sdk-bridge --help >/dev/null

# Default embed model, so the first `index` is not a surprise Hub download.
# chown here (not after COPY of app code) so a source change does not re-walk
# the snapshot.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')" \
    && chown -R answerbot:answerbot /opt/hf

COPY answerbot ./answerbot
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --no-build-isolation .

COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

VOLUME ["/data"]
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "answerbot.bot"]
