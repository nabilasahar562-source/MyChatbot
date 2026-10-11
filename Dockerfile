FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 1024
CMD ["gunicorn", "--bind", "0.0.0.0:1024", "--workers", "2", "--timeout", "120", "app:app"]