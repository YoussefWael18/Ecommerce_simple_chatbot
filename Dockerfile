FROM python:3.10-slim
WORKDIR /srv/app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY prompts ./prompts
COPY Model_pickle ./Model_pickle
COPY data/raw ./data/raw
COPY scripts ./scripts
EXPOSE 8000
# Build the index once with `python scripts/build_vector_store.py` before serving.
CMD ["sh", "-c", "python scripts/build_vector_store.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
