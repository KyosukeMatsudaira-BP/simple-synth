"""
Tkinter GUI + MIDI/PCキーボード入力で操作するポリフォニックシンセの最小例
- ADSR (Attack, Decay, Sustain, Release) とフィルタカットオフをスライダーで調整
- オンスクリーン鍵盤ボタン、PCキーボード、MIDIキーボードからの入力で音を出す
- オシレーターの波形種別（sine, triangle, square, sawtooth）を選択可能
- 同時発音数（ポリフォニック）はデフォルト6音
　
必要なパッケージ:
    pip install sounddevice numpy tkinter mido python-rtmidi
※ tkinter は通常標準ライブラリに含まれています。
"""

import sys
import math
import threading
import time
from typing import Any
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk

# MIDI 入力用ライブラリ（なければ MIDI 入力は無効）
try:
    import mido
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False

SAMPLE_RATE = 44100

# ============================
# シンセ本体（単音シンセ）: SimpleSynth
# ============================
class SimpleSynth:
    def __init__(self):
        # ADSR パラメータ
        self.attack = 0.1
        self.decay = 0.2
        self.sustain = 0.7
        self.release = 0.5

        # フィルタカットオフ
        self.cutoff = 1000.0

        # エンベロープ状態
        self.env_value = 0.0
        self.env_state = 'idle'  # 'idle', 'attack', 'decay', 'sustain', 'release'
        self.env_inc = 0.0
        self.env_release_start = 0.0

        # 現在のノート（単音なのでひとつ）
        self.current_note = None

        # オシレーター関連
        self.phase = 0.0
        self.frequency = 0.0
        # オシレーターの波形種別 ("sine", "triangle", "square", "sawtooth")
        self.osc_type = "sine"

        # フィルタ（1次ローパス）の内部状態
        self.y_prev = 0.0

    def note_on(self, note_number: int):
        self.current_note = note_number
        self.set_frequency(self.note2freq(note_number))
        self.env_state = 'attack'
        if self.attack > 0:
            self.env_inc = 1.0 / (self.attack * SAMPLE_RATE)
        else:
            self.env_value = 1.0
            self._start_decay()

    def note_off(self, note_number: int):
        if self.current_note == note_number:
            self.env_release_start = self.env_value
            self.env_state = 'release'
            if self.release > 0:
                self.env_inc = self.env_release_start / (self.release * SAMPLE_RATE)
            else:
                self.env_value = 0.0
                self.env_state = 'idle'
            self.current_note = None

    def set_frequency(self, freq: float):
        self.frequency = freq

    def note2freq(self, note_number: int) -> float:
        # A4 (MIDI 69) -> 440 Hz
        return 440.0 * (2.0 ** ((note_number - 69) / 12.0))

    def set_adsr(self, a: float, d: float, s: float, r: float):
        self.attack = a
        self.decay = d
        self.sustain = s
        self.release = r

    def set_cutoff(self, cutoff: float):
        self.cutoff = cutoff

    def set_osc_type(self, osc_type: str):
        # osc_type: "sine", "triangle", "square", "sawtooth"
        self.osc_type = osc_type

    def adsr_process(self) -> float:
        if self.env_state == 'idle':
            return 0.0
        elif self.env_state == 'attack':
            self.env_value += self.env_inc
            if self.env_value >= 1.0:
                self.env_value = 1.0
                self._start_decay()
        elif self.env_state == 'decay':
            self.env_value -= self.env_inc
            if self.env_value <= self.sustain:
                self.env_value = self.sustain
                self.env_state = 'sustain'
        elif self.env_state == 'sustain':
            pass
        elif self.env_state == 'release':
            self.env_value -= self.env_inc
            if self.env_value <= 0.0:
                self.env_value = 0.0
                self.env_state = 'idle'
        return self.env_value

    def _start_decay(self):
        self.env_state = 'decay'
        if self.decay > 0:
            self.env_inc = (1.0 - self.sustain) / (self.decay * SAMPLE_RATE)
        else:
            self.env_value = self.sustain
            self.env_state = 'sustain'

    def oscillator(self) -> float:
        # 波形生成
        if self.osc_type == "sine":
            sample = math.sin(self.phase)
        elif self.osc_type == "triangle":
            sample = 2 * abs(2 * ((self.phase / (2 * math.pi)) - math.floor(self.phase / (2 * math.pi) + 0.5))) - 1
        elif self.osc_type == "square":
            sample = 1.0 if math.sin(self.phase) >= 0 else -1.0
        elif self.osc_type == "sawtooth":
            sample = 2 * (self.phase / (2 * math.pi)) - 1
        else:
            sample = math.sin(self.phase)
        phase_inc = (2.0 * math.pi * self.frequency) / SAMPLE_RATE
        self.phase += phase_inc
        if self.phase >= 2.0 * math.pi:
            self.phase -= 2.0 * math.pi
        return sample

    def filter_process(self, x: float, cutoff: float) -> float:
        if cutoff < 20:
            cutoff = 20
        elif cutoff > SAMPLE_RATE / 2:
            cutoff = SAMPLE_RATE / 2
        alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff / SAMPLE_RATE)
        out = alpha * x + (1.0 - alpha) * self.y_prev
        self.y_prev = out
        return out

# ============================
# Voice クラス：１つの発音（ノート）を管理
# ============================
class Voice:
    def __init__(self, note_number: int, master_params: dict, sample_rate=SAMPLE_RATE):
        self.note_number = note_number
        # 新たにシンセ本体を生成し、マスターのパラメータをコピー
        self.synth = SimpleSynth()
        self.synth.attack = master_params.get("attack", 0.1)
        self.synth.decay = master_params.get("decay", 0.2)
        self.synth.sustain = master_params.get("sustain", 0.7)
        self.synth.release = master_params.get("release", 0.5)
        self.synth.cutoff = master_params.get("cutoff", 1000.0)
        self.synth.osc_type = master_params.get("osc_type", "sine")
        self.start_time = time.time()
        self.active = True
        self.synth.note_on(note_number)

    def note_off(self):
        self.synth.note_off(self.note_number)

    def process(self) -> float:
        sample = self.synth.adsr_process() * self.synth.oscillator()
        filtered = self.synth.filter_process(sample, self.synth.cutoff)
        if self.synth.env_state == 'idle':
            self.active = False
        return filtered

# ============================
# PolySynth クラス：複数の Voice を管理（ポリフォニック化）
# ============================
class PolySynth:
    def __init__(self, max_voices=6, sample_rate=SAMPLE_RATE):
        self.max_voices = max_voices
        self.voices = []  # Voice のリスト
        self.sample_rate = sample_rate
        # マスターパラメータ（GUIからの更新で変更される）
        self.attack = 0.1
        self.decay = 0.2
        self.sustain = 0.7
        self.release = 0.5
        self.cutoff = 1000.0
        self.osc_type = "sine"

    def update_parameters(self):
        for voice in self.voices:
            voice.synth.attack = self.attack
            voice.synth.decay = self.decay
            voice.synth.sustain = self.sustain
            voice.synth.release = self.release
            voice.synth.cutoff = self.cutoff
            voice.synth.osc_type = self.osc_type

    def set_attack(self, a: float):
        self.attack = a
        self.update_parameters()

    def set_decay(self, d: float):
        self.decay = d
        self.update_parameters()

    def set_sustain(self, s: float):
        self.sustain = s
        self.update_parameters()

    def set_release(self, r: float):
        self.release = r
        self.update_parameters()

    def set_cutoff(self, c: float):
        self.cutoff = c
        self.update_parameters()

    def set_osc_type(self, osc: str):
        self.osc_type = osc
        self.update_parameters()

    def note_on(self, note_number: int):
        # すでに同じノートが鳴っていればリトリガー
        for voice in self.voices:
            if voice.note_number == note_number and voice.active:
                voice.synth.note_on(note_number)
                return
        # 空いている場合は新たにVoiceを追加
        master_params = {
            "attack": self.attack,
            "decay": self.decay,
            "sustain": self.sustain,
            "release": self.release,
            "cutoff": self.cutoff,
            "osc_type": self.osc_type
        }
        if len(self.voices) < self.max_voices:
            new_voice = Voice(note_number, master_params, self.sample_rate)
            self.voices.append(new_voice)
        else:
            # 全てのVoiceが使われている場合、古いものを奪う（Voice stealing）
            oldest_voice = min(self.voices, key=lambda v: v.start_time)
            oldest_voice.note_off()
            oldest_voice.synth.note_on(note_number)
            oldest_voice.note_number = note_number
            oldest_voice.start_time = time.time()

    def note_off(self, note_number: int):
        for voice in self.voices:
            if voice.note_number == note_number and voice.active:
                voice.note_off()

    def process(self) -> float:
        sample_sum = 0.0
        active_voices = []
        for voice in self.voices:
            sample_sum += voice.process()
            if voice.active:
                active_voices.append(voice)
        self.voices = active_voices
        # ミキシング：複数Voiceの合計を平均化（クリッピング防止）
        return sample_sum / max(1, self.max_voices)

# グローバルな PolySynth インスタンス（デフォルト6音）
poly_synth = PolySynth(max_voices=6, sample_rate=SAMPLE_RATE)

# ============================
# オーディオコールバック（PolySynth版）
# ============================
def audio_callback(outdata: np.ndarray, frames: int, time_info, status) -> None:
    if status:
        print("Audio status:", status)
    outdata.fill(0.0)
    for i in range(frames):
        outdata[i, 0] = poly_synth.process()

# ============================
# Tkinter GUI部分
# ============================
class SynthGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Polyphonic Synth")
        self.geometry("400x600")
        self.create_controls()
        self.create_keyboard()

        # PCキーボード入力（ウィンドウ全体で受け付ける）
        self.bind("<KeyPress>", self.on_key_press)
        self.bind("<KeyRelease>", self.on_key_release)
        # PCキーボード用のキー割り当て例
        self.pc_key_map = {
            'z': 60,
            'x': 62,
            'c': 64,
            'v': 65,
            'b': 67,
            'n': 69,
            'm': 71
        }
        self.active_keys = set()

    def create_controls(self):
        frame = tk.Frame(self)
        frame.pack(pady=10)

        # ADSR スライダー群
        self.attack_var = tk.DoubleVar(value=poly_synth.attack)
        tk.Label(frame, text="Attack").grid(row=0, column=0, sticky="w")
        attack_slider = tk.Scale(frame, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL,
                                 variable=self.attack_var, command=self.update_attack)
        attack_slider.grid(row=0, column=1)

        self.decay_var = tk.DoubleVar(value=poly_synth.decay)
        tk.Label(frame, text="Decay").grid(row=1, column=0, sticky="w")
        decay_slider = tk.Scale(frame, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL,
                                variable=self.decay_var, command=self.update_decay)
        decay_slider.grid(row=1, column=1)

        self.sustain_var = tk.DoubleVar(value=poly_synth.sustain)
        tk.Label(frame, text="Sustain").grid(row=2, column=0, sticky="w")
        sustain_slider = tk.Scale(frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                  variable=self.sustain_var, command=self.update_sustain)
        sustain_slider.grid(row=2, column=1)

        self.release_var = tk.DoubleVar(value=poly_synth.release)
        tk.Label(frame, text="Release").grid(row=3, column=0, sticky="w")
        release_slider = tk.Scale(frame, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL,
                                  variable=self.release_var, command=self.update_release)
        release_slider.grid(row=3, column=1)

        # Filter Cutoff スライダー
        self.cutoff_var = tk.DoubleVar(value=poly_synth.cutoff)
        tk.Label(frame, text="Cutoff (Hz)").grid(row=4, column=0, sticky="w")
        cutoff_slider = tk.Scale(frame, from_=20.0, to=5000.0, resolution=1, orient=tk.HORIZONTAL,
                                 variable=self.cutoff_var, command=self.update_cutoff)
        cutoff_slider.grid(row=4, column=1)

        # オシレーター選択 (OptionMenu)
        tk.Label(frame, text="Oscillator").grid(row=5, column=0, sticky="w")
        self.osc_types = ["sine", "triangle", "square", "sawtooth"]
        self.osc_var = tk.StringVar(value=poly_synth.osc_type)
        osc_menu = tk.OptionMenu(frame, self.osc_var, *self.osc_types, command=self.update_osc_type)
        osc_menu.grid(row=5, column=1)

    def create_keyboard(self):
        kb_frame = tk.Frame(self)
        kb_frame.pack(pady=20)
        # オンスクリーン鍵盤（例: C, D, E, F, G, A, B）
        self.key_buttons = {}
        notes = [60, 62, 64, 65, 67, 69, 71]
        labels = ["C", "D", "E", "F", "G", "A", "B"]
        self.button_vars = {}
        for i, note in enumerate(notes):
            var = tk.IntVar(value=0)
            btn = tk.Checkbutton(
                kb_frame, text=labels[i], width=4,
                variable=var, indicatoron=False
            )
            btn.grid(row=0, column=i, padx=5)
            btn.bind("<ButtonPress-1>", lambda event, n=note: self.on_note_press(n))
            btn.bind("<ButtonRelease-1>", lambda event, n=note: self.on_note_release(n))
            self.key_buttons[note] = btn
            self.button_vars[note] = var

    # スライダー更新コールバック
    def update_attack(self, val):
        poly_synth.set_attack(float(val))
    def update_decay(self, val):
        poly_synth.set_decay(float(val))
    def update_sustain(self, val):
        poly_synth.set_sustain(float(val))
    def update_release(self, val):
        poly_synth.set_release(float(val))
    def update_cutoff(self, val):
        poly_synth.set_cutoff(float(val))
    def update_osc_type(self, val):
        poly_synth.set_osc_type(val)

    # オンスクリーン鍵盤
    def on_note_press(self, note):
        poly_synth.note_on(note)
    def on_note_release(self, note):
        poly_synth.note_off(note)

    # PCキーボード入力イベント
    def on_key_press(self, event):
        key = event.char
        if key in self.pc_key_map and key not in self.active_keys:
            self.active_keys.add(key)
            note = self.pc_key_map[key]
            poly_synth.note_on(note)
    def on_key_release(self, event):
        key = event.char
        if key in self.pc_key_map and key in self.active_keys:
            self.active_keys.remove(key)
            note = self.pc_key_map[key]
            poly_synth.note_off(note)

# ============================
# MIDI入力用スレッド
# ============================
def midi_input_thread():
    if not MIDI_AVAILABLE:
        print("MIDI input not available.")
        return
    try:
        ports = mido.get_input_names()
        if not ports:
            print("No MIDI input ports found.")
            return
        print("Using MIDI port:", ports[0])
        with mido.open_input(ports[0]) as port:
            for msg in port:
                if msg.type == 'note_on' and msg.velocity > 0:
                    print("MIDI Note On:", msg.note)
                    poly_synth.note_on(msg.note)
                elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                    print("MIDI Note Off:", msg.note)
                    poly_synth.note_off(msg.note)
    except Exception as e:
        print("MIDI input error:", e)

# ============================
# メイン関数
# ============================
def main():
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=512, callback=audio_callback)
    stream.start()

    if MIDI_AVAILABLE:
        midi_thread = threading.Thread(target=midi_input_thread, daemon=True)
        midi_thread.start()

    app = SynthGUI()
    app.mainloop()

    stream.stop()
    stream.close()

if __name__ == "__main__":
    main()
