FROM python:3.9-slim

# Install system dependencies required for PyAudio
RUN apt-get update && apt-get install -y \
	portaudio19-dev \
	python3-pyaudio \
	gcc \
	&& rm -rf /var/lib/apt/lists/*
	
WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/

ENTRYPOINT ["python", "command_processing.py"]
