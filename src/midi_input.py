# midi_input.py
import mido
from synth_core import PolySynth
from synth_core import SAMPLE_RATE
import threading

def midi_input_thread(poly_synth: PolySynth):
    try:
        ports = mido.get_input_names()
        if not ports:
            print("MIDI 入力ポートが見つかりません")
            return
        print("使用する MIDI ポート:", ports[0])
        with mido.open_input(ports[0]) as port:
            for msg in port:
                if msg.type == 'note_on' and msg.velocity > 0:
                    print("MIDI ノートオン:", msg.note)
                    poly_synth.note_on(msg.note)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    print("MIDI ノートオフ:", msg.note)
                    poly_synth.note_off(msg.note)
    except Exception as e:
        print("MIDI 入力エラー:", e)

if __name__ == "__main__":
    pass
