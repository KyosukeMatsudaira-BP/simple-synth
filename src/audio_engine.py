# audio_engine.py
import numpy as np
import sounddevice as sd
from synth_core import PolySynth, SAMPLE_RATE
import threading

# グローバル波形バッファ
WAVEFORM_BUFFER_SIZE = 2048
waveform_buffer = np.zeros(WAVEFORM_BUFFER_SIZE)
buffer_index = 0
buffer_lock = threading.Lock()

def audio_callback(outdata: np.ndarray, frames: int, time_info, status, poly_synth: PolySynth):
    global buffer_index
    if status:
        print("オーディオステータス:", status)
    outdata.fill(0.0)
    for i in range(frames):
        sample = poly_synth.process()
        outdata[i, 0] = sample
        with buffer_lock:
            waveform_buffer[buffer_index] = sample
            buffer_index = (buffer_index + 1) % WAVEFORM_BUFFER_SIZE

def start_audio(poly_synth: PolySynth):
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=512,
                             callback=lambda outdata, frames, time_info, status:
                             audio_callback(outdata, frames, time_info, status, poly_synth))
    stream.start()
    return stream
