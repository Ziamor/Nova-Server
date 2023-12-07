from flask import Flask
from flask_socketio import SocketIO
import pyaudio
import numpy as np
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import wave
import sys
import os
import json

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

sys.stdout.flush()

def process_audio_data(audio_data):
    print("Processing audio data")
    if recognizer.AcceptWaveform(audio_data.tobytes()):
        result = json.loads(recognizer.Result())
        text = result.get('text', '')
        print("Transcribed text: ", text)
        socketio.emit('command_detected', {'command': text})
        return text
    else:
        # Handle partial results if needed
        print("Partial result")
        
    sys.stdout.flush()

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    sys.stdout.flush()

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")
    sys.stdout.flush()

@socketio.on_error()
def handle_error(e):
    print("An error occurred:")
    print(e)
    sys.stdout.flush()

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
        
    sys.stdout.flush()

@socketio.on('process_command')
def handle_process_command(data):
	print("Received detect command event")
	result = recognizer.FinalResult()
	print("Final result: ", result)
	sys.stdout.flush()
	socketio.emit('command_detected', {'command': result})

def process_wav_file(wav_file_path):
    """Process a WAV file for testing."""
    with wave.open(wav_file_path, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
        return process_audio_data(audio_data)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5002)
