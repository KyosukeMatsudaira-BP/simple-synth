# main.py
import threading
import sounddevice as sd
from synth_core import PolySynth, SAMPLE_RATE
from midi_input import midi_input_thread
from audio_engine import audio_callback, start_audio
from gui import SynthGUI
import tkinter as tk

if __name__ == "__main__":
    poly_synth = PolySynth(max_voices=6, sample_rate=SAMPLE_RATE)
    stream = start_audio(poly_synth)
    if poly_synth is not None:
        if __import__("mido", globals(), locals(), [], 0):
            midi_thread = threading.Thread(target=midi_input_thread, args=(poly_synth,), daemon=True)
            midi_thread.start()
    app = SynthGUI(poly_synth)
    app.mainloop()
    stream.stop()
    stream.close()
