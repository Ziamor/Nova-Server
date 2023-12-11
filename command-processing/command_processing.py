from flask import Flask
from flask_socketio import SocketIO
import pyaudio
import numpy as np
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import wave
import os
import json
import requests

app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent', path='/command-processing')

# Audio stream configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION_MS = 30
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)

# Load Vosk model
model = Model(os.path.join(os.path.dirname(__file__), 'models', 'vosk-model-small-en-us-0.15'))

recognizer = KaldiRecognizer(model, RATE)
recognizer.SetWords(True)
recognizer.SetPartialWords(True)

def process_audio_data(audio_data):
    recognizer.AcceptWaveform(audio_data.tobytes())

def send_text_to_llm(text):
    print("Sending text to LLM service")
    url = 'http://llm-processing:5003/process_text'
    data = {'text': text + '\n'}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print("Text sent to LLM service successfully")
        result = response.json()
        print("LLM result: ", result)        
        socketio.emit('command_detected', {'command': result['result']}) 
    else:
        print("Failed to send text to LLM service")
        
@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on_error()
def handle_error(e):
    print("An error occurred:")
    print(e)

@socketio.on('stream_audio')
def handle_stream_audio(frame):
    frame_id = frame['frame_id']
    data = frame['data']
    if isinstance(data, bytes):
        audio_data = np.frombuffer(data, dtype=np.int16)
        process_audio_data(audio_data)
    else:
        print("Received non-bytes data")
        print(data)

@socketio.on('process_command')
def handle_process_command(data):
    print("Received detect command event")
    result = recognizer.FinalResult()
    text = json.loads(result)['text']
    print("Transcribed text: ", text)
    print("Sending text to LLM service")
    send_text_to_llm(text)   

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5002)
