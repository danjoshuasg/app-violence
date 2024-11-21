# Dockerfile
FROM python:3.10.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y git

COPY . .

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]