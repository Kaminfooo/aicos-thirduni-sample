FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.lock.txt ./requirements.lock.txt
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY backend/ .
COPY VERSION ./VERSION
COPY dashboard/ ./dashboard/
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
