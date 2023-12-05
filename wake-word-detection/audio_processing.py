from flask import Flask, request
from flask_socketio import SocketIO
import numpy as np
from openwakeword.model import Model
import pyaudio
import os
import time
import sys

app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent', path='/wake-word-detection')

# Audio stream configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION_MS = 30
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)

# Pre-trained openwakeword model path
model_path = os.path.join(os.path.dirname(__file__), 'models', 'hey_nova.tflite')
inference_framework = 'tflite'

# Global variables
last_detection_times = {}
client_models = {}
DEBOUNCE_PERIOD = 1

def load_model():
    return Model(wakeword_models=[model_path], inference_framework=inference_framework)

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    sys.stdout.flush()
    client_sid = request.sid
    client_models[client_sid] = load_model()
    last_detection_times[client_sid] = 0
    print("Model loaded")
    sys.stdout.flush()

@socketio.on('disconnect')
def handle_disconnect():
    client_sid = request.sid
    if client_sid in client_models:
        del client_models[client_sid]
        del last_detection_times[client_sid]
    print("Client disconnected")
    sys.stdout.flush()

@socketio.on('stream_audio')
def handle_stream_audio(data):
    try:
        client_sid = request.sid
        if isinstance(data, bytes):
            audio_data = np.frombuffer(data, dtype=np.int16)
            prediction = client_models[client_sid].predict(audio_data)
            
            for mdl, scores in client_models[client_sid].prediction_buffer.items():
                curr_score = scores[-1]
                if curr_score > 0.5:
                    current_time = time.time()
                    if current_time - last_detection_times[client_sid] > DEBOUNCE_PERIOD:
                        last_detection_times[client_sid] = current_time
                        print(f"Wake word detected by model {mdl} with score {curr_score}")
                        sys.stdout.flush()
                        curr_score = float(curr_score)
                        socketio.emit('wake_word_detected', {'model': mdl, 'score': curr_score}, room=client_sid)
                    else:
                        print("Wake word detection ignored due to debounce period")
                        sys.stdout.flush()
        else:
            print("Received non-bytes data")
            print(data)
    except Exception as e:
        print(f"Error in handle_stream_audio: {e}")
        sys.stdout.flush()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001)
