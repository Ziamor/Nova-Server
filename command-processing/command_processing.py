from flask import Flask
from flask_socketio import SocketIO
import pyaudio
import numpy as np
import speech_recognition as sr
import wave
import sys

app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent', path='/command-processing')

# Audio stream configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION_MS = 30
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)

recognizer = sr.Recognizer()

def process_audio_data(audio_data):
    print("Processing audio data")
    """Process and transcribe audio data."""
    audio = sr.AudioData(audio_data.tobytes(), RATE, 2)
    try:
        text = recognizer.recognize_google(audio)
        print("Transcribed text: ", text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        
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
def handle_stream_audio(data):
    if isinstance(data, bytes):
        audio_data = np.frombuffer(data, dtype=np.int16)
        #process_audio_data(audio_data)
    else:
        print("Received non-bytes data")
        print(data)
        sys.stdout.flush()

def process_wav_file(wav_file_path):
    """Process a WAV file for testing."""
    with wave.open(wav_file_path, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
        return process_audio_data(audio_data)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5002)
