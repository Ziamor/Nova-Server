FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/

ENTRYPOINT ["python", "voice_synthesis.py"]