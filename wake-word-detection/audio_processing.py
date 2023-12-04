from flask import Flask
from flask import request
from flask_socketio import SocketIO
import numpy as np
from openwakeword.model import Model
import pyaudio
import sys
import os
import time

app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent')

# Audio stream configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION_MS = 30  # Frame duration in ms for VAD (10, 20, or 30 ms)
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)  # Number of frames per buffer

# Load pre-trained openwakeword models
model_path = os.path.join(os.path.dirname(__file__), 'models', 'hey_nova.tflite')
inference_framework = 'tflite'  # Change to 'onnx' if you are using ONNX models
owwModel = Model(wakeword_models=[model_path], inference_framework=inference_framework)

# Global variable to track the last detection time
# Dictionary to track the last detection time for each client
last_detection_times = {}
DEBOUNCE_PERIOD = 1  # seconds

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    # Any additional code you want to run when a client connects

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('stream_audio')
def handle_stream_audio(data):
    global last_detection_time
    
    client_sid = request.sid
    if client_sid not in last_detection_times:
        last_detection_times[client_sid] = 0
    
    if isinstance(data, bytes):
        # Process the audio data
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio_data)
        
        # Process the prediction
        for mdl, scores in owwModel.prediction_buffer.items():
            curr_score = scores[-1]
            if curr_score > 0.5:
                current_time = time.time()

                # Check if the detection is within the debounce period for this client
                if current_time - last_detection_times[client_sid] > DEBOUNCE_PERIOD:
                    last_detection_times[client_sid] = current_time
                    print(f"Wake word detected by model {mdl} with score {curr_score}")

                    # Convert curr_score to native Python float for JSON serialization
                    curr_score = float(curr_score)

                    # Emitting an event to the specific client
                    socketio.emit('wake_word_detected', {'model': mdl, 'score': curr_score}, room=client_sid)
                else:
                    print("Wake word detection ignored due to debounce period")
    else:
        print("Received non-bytes data")
        print(data)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001)
