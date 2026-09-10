FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
    HF_HOME=/models/huggingface
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
RUN groupadd --gid 10001 rag && useradd --uid 10001 --gid rag --create-home rag \
    && mkdir -p /data /models/huggingface && chown -R rag:rag /data /models
COPY --chown=rag:rag app ./app
COPY --chown=rag:rag frontend ./frontend
COPY --chown=rag:rag scripts ./scripts
USER rag
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready',timeout=3)"
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
