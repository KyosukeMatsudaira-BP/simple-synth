"""
Tkinter GUI + MIDI/PCキーボード入力で操作するポリフォニックシンセ
（プリセット管理、一時保存、ピアノ鍵盤、スペクトル表示追加版）

【機能】
- ADSR（アンプエンベロープ）、フィルタカットオフ、レゾナンス、LFO、ノイズミックス、矩形波デューティー比をスライダーで調整
- オシレーターの波形種別（sine, triangle, square, sawtooth）に加え、ホワイト／ピンクノイズを混合可能
- ノイズと通常波形の混合比を調整可能
- 左側にコントロール＋プリセット管理＋一時保存、右側に波形表示・スペクトル表示、下部にピアノ鍵盤を配置
- PCキーボード（白鍵: z,x,c,v,b,n,m、黒鍵: s,d,g,h,j）およびMIDI入力に対応
- 同時発音数はデフォルト6音

必要なパッケージ:
    pip install sounddevice numpy tkinter mido python-rtmidi
※ tkinter は標準ライブラリです。
"""

import sys, math, threading, time, random
from typing import Any
import numpy as np
import numpy.fft as fft
import sounddevice as sd
import tkinter as tk
from tkinter import ttk

try:
    import mido
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False

SAMPLE_RATE = 44100

# ---------------------------
# グローバル：波形表示用リングバッファ
# ---------------------------
WAVEFORM_BUFFER_SIZE = 2048
waveform_buffer = np.zeros(WAVEFORM_BUFFER_SIZE)
buffer_index = 0
buffer_lock = threading.Lock()

# ---------------------------
# LFO クラス（低周波オシレーター）
# ---------------------------
class LFO:
    def __init__(self, rate=5.0, depth=0.0, sample_rate=SAMPLE_RATE):
        self.rate = rate          
        self.depth = depth        
        self.phase = 0.0
        self.sample_rate = sample_rate

    def process(self) -> float:
        value = math.sin(self.phase)
        self.phase += (2.0 * math.pi * self.rate) / self.sample_rate
        if self.phase >= 2.0 * math.pi:
            self.phase -= 2.0 * math.pi
        return value

# ---------------------------
# ResonantLPF クラス（レゾナンス付きバイクワッド低域通過フィルタ）
# ---------------------------
class ResonantLPF:
    def __init__(self, sample_rate=SAMPLE_RATE, cutoff=1000.0, q=0.707):
        self.sample_rate = sample_rate
        self.cutoff = cutoff
        self.q = q
        self.b0 = self.b1 = self.b2 = self.a1 = self.a2 = 0.0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0
        self.update_coefficients()

    def update_coefficients(self):
        omega = 2 * math.pi * self.cutoff / self.sample_rate
        alpha = math.sin(omega) / (2 * self.q)
        cos_omega = math.cos(omega)
        a0 = 1 + alpha
        self.b0 = ((1 - cos_omega) / 2) / a0
        self.b1 = (1 - cos_omega) / a0
        self.b2 = ((1 - cos_omega) / 2) / a0
        self.a1 = -2 * cos_omega / a0
        self.a2 = (1 - alpha) / a0

    def set_cutoff(self, cutoff: float):
        self.cutoff = cutoff
        self.update_coefficients()

    def set_q(self, q: float):
        self.q = q
        self.update_coefficients()

    def process(self, x: float) -> float:
        y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y

# ---------------------------
# SimpleSynth クラス（単音シンセ）
# ---------------------------
class SimpleSynth:
    def __init__(self):
        self.attack = 0.1
        self.decay = 0.2
        self.sustain = 0.7
        self.release = 0.5
        self.cutoff = 1000.0
        self.env_value = 0.0
        self.env_state = 'idle'
        self.env_inc = 0.0
        self.env_release_start = 0.0
        self.current_note = None
        self.phase = 0.0
        self.frequency = 0.0
        self.osc_type = "sine"
        self.duty_cycle = 0.5
        self.noise_mix = 0.0
        self.noise_type = "white"
        self.pink_b0 = self.pink_b1 = self.pink_b2 = self.pink_b3 = self.pink_b4 = self.pink_b5 = self.pink_b6 = 0.0
        self.y_prev = 0.0
        self.resonance = 0.0
        self.filter = None
        self.smoothed_cutoff = self.cutoff
        self.lfo = LFO(rate=5.0, depth=0.0, sample_rate=SAMPLE_RATE)

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
        return 440.0 * (2.0 ** ((note_number - 69) / 12.0))

    def set_adsr(self, a: float, d: float, s: float, r: float):
        self.attack = a
        self.decay = d
        self.sustain = s
        self.release = r

    def set_cutoff(self, cutoff: float):
        self.cutoff = cutoff

    def set_osc_type(self, osc_type: str):
        self.osc_type = osc_type

    def set_resonance(self, res: float):
        self.resonance = res
        if self.resonance > 0:
            self.filter = ResonantLPF(sample_rate=SAMPLE_RATE, cutoff=self.cutoff, q=self.resonance)
        else:
            self.filter = None

    def set_lfo_rate(self, rate: float):
        self.lfo.rate = rate

    def set_lfo_depth(self, depth: float):
        self.lfo.depth = depth

    def set_noise_mix(self, mix: float):
        self.noise_mix = mix

    def set_duty_cycle(self, duty: float):
        self.duty_cycle = duty

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
        lfo_val = self.lfo.process()
        modulated_freq = self.frequency * (1 + self.lfo.depth * lfo_val)
        if self.osc_type == "sine":
            waveform = math.sin(self.phase)
        elif self.osc_type == "triangle":
            waveform = 2 * abs(2 * ((self.phase / (2 * math.pi)) - math.floor(self.phase / (2 * math.pi) + 0.5))) - 1
        elif self.osc_type == "square":
            waveform = 1.0 if (self.phase / (2 * math.pi)) < self.duty_cycle else -1.0
        elif self.osc_type == "sawtooth":
            waveform = 2 * (self.phase / (2 * math.pi)) - 1
        else:
            waveform = math.sin(self.phase)
        phase_inc = (2.0 * math.pi * modulated_freq) / SAMPLE_RATE
        self.phase += phase_inc
        if self.phase >= 2.0 * math.pi:
            self.phase -= 2.0 * math.pi
        if self.noise_type == "white":
            noise = random.uniform(-1, 1)
        elif self.noise_type == "pink":
            white = random.uniform(-1, 1)
            self.pink_b0 = 0.99886 * self.pink_b0 + white * 0.0555179
            self.pink_b1 = 0.99332 * self.pink_b1 + white * 0.0750759
            self.pink_b2 = 0.96900 * self.pink_b2 + white * 0.1538520
            self.pink_b3 = 0.86650 * self.pink_b3 + white * 0.3104856
            self.pink_b4 = 0.55000 * self.pink_b4 + white * 0.5329522
            self.pink_b5 = -0.7616 * self.pink_b5 - white * 0.0168980
            noise = (self.pink_b0 + self.pink_b1 + self.pink_b2 +
                     self.pink_b3 + self.pink_b4 + self.pink_b5 + self.pink_b6 +
                     white * 0.5362)
            self.pink_b6 = white * 0.115926
        else:
            noise = 0.0
        mixed_wave = (1 - self.noise_mix) * waveform + self.noise_mix * noise
        return mixed_wave

    def filter_process(self, x: float, cutoff: float) -> float:
        if cutoff < 20:
            cutoff = 20
        elif cutoff > SAMPLE_RATE / 2:
            cutoff = SAMPLE_RATE / 2
        smoothing_factor = 0.01
        self.smoothed_cutoff += smoothing_factor * (cutoff - self.smoothed_cutoff)
        cutoff_used = self.smoothed_cutoff
        if self.resonance > 0 and self.filter is not None:
            self.filter.set_cutoff(cutoff_used)
            self.filter.set_q(self.resonance)
            return self.filter.process(x)
        else:
            alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_used / SAMPLE_RATE)
            out = alpha * x + (1.0 - alpha) * self.y_prev
            self.y_prev = out
            return out

# ---------------------------
# Voice クラス：1つの発音を管理
# ---------------------------
class Voice:
    def __init__(self, note_number: int, master_params: dict, sample_rate=SAMPLE_RATE):
        self.note_number = note_number
        self.synth = SimpleSynth()
        self.synth.set_adsr(master_params.get("attack", 0.1),
                           master_params.get("decay", 0.2),
                           master_params.get("sustain", 0.7),
                           master_params.get("release", 0.5))
        self.synth.set_cutoff(master_params.get("cutoff", 1000.0))
        self.synth.set_osc_type(master_params.get("osc_type", "sine"))
        self.synth.set_resonance(master_params.get("resonance", 0.0))
        self.synth.set_lfo_rate(master_params.get("lfo_rate", 5.0))
        self.synth.set_lfo_depth(master_params.get("lfo_depth", 0.0))
        self.synth.set_noise_mix(master_params.get("noise_mix", 0.0))
        self.synth.set_duty_cycle(master_params.get("duty_cycle", 0.5))
        self.synth.noise_type = master_params.get("noise_type", "white")
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

# ---------------------------
# PolySynth クラス：複数 Voice の管理（ポリフォニック化）
# ---------------------------
class PolySynth:
    def __init__(self, max_voices=6, sample_rate=SAMPLE_RATE):
        self.max_voices = max_voices
        self.voices = []
        self.sample_rate = sample_rate
        self.attack = 0.1
        self.decay = 0.2
        self.sustain = 0.7
        self.release = 0.5
        self.cutoff = 1000.0
        self.osc_type = "sine"
        self.resonance = 0.0
        self.lfo_rate = 5.0
        self.lfo_depth = 0.0
        self.noise_mix = 0.0
        self.duty_cycle = 0.5
        self.noise_type = "white"

    def update_parameters(self):
        for voice in self.voices:
            voice.synth.set_adsr(self.attack, self.decay, self.sustain, self.release)
            voice.synth.set_cutoff(self.cutoff)
            voice.synth.set_osc_type(self.osc_type)
            voice.synth.set_resonance(self.resonance)
            voice.synth.set_lfo_rate(self.lfo_rate)
            voice.synth.set_lfo_depth(self.lfo_depth)
            voice.synth.set_noise_mix(self.noise_mix)
            voice.synth.set_duty_cycle(self.duty_cycle)
            voice.synth.noise_type = self.noise_type

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
    def set_resonance(self, res: float):
        self.resonance = res
        self.update_parameters()
    def set_lfo_rate(self, rate: float):
        self.lfo_rate = rate
        self.update_parameters()
    def set_lfo_depth(self, depth: float):
        self.lfo_depth = depth
        self.update_parameters()
    def set_noise_mix(self, mix: float):
        self.noise_mix = mix
        self.update_parameters()
    def set_duty_cycle(self, duty: float):
        self.duty_cycle = duty
        self.update_parameters()
    def set_noise_type(self, ntype: str):
        self.noise_type = ntype
        self.update_parameters()

    def note_on(self, note_number: int):
        for voice in self.voices:
            if voice.note_number == note_number and voice.active:
                voice.synth.note_on(note_number)
                return
        master_params = {
            "attack": self.attack,
            "decay": self.decay,
            "sustain": self.sustain,
            "release": self.release,
            "cutoff": self.cutoff,
            "osc_type": self.osc_type,
            "resonance": self.resonance,
            "lfo_rate": self.lfo_rate,
            "lfo_depth": self.lfo_depth,
            "noise_mix": self.noise_mix,
            "duty_cycle": self.duty_cycle,
            "noise_type": self.noise_type
        }
        if len(self.voices) < self.max_voices:
            new_voice = Voice(note_number, master_params, self.sample_rate)
            self.voices.append(new_voice)
        else:
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
        return sample_sum / max(1, self.max_voices)

# グローバルな PolySynth インスタンス（デフォルト6音）
poly_synth = PolySynth(max_voices=6, sample_rate=SAMPLE_RATE)

# ---------------------------
# オーディオコールバック（PolySynth版）
# ---------------------------
def audio_callback(outdata: np.ndarray, frames: int, time_info, status) -> None:
    if status:
        print("Audio status:", status)
    outdata.fill(0.0)
    for i in range(frames):
        sample = poly_synth.process()
        outdata[i, 0] = sample
        global buffer_index
        with buffer_lock:
            waveform_buffer[buffer_index] = sample
            buffer_index = (buffer_index + 1) % WAVEFORM_BUFFER_SIZE

# ---------------------------
# プリセット管理用（プリセットは固定、上書き不可）
# ---------------------------
default_presets = {
    1: {"attack": 1.0, "decay": 1.0, "sustain": 0.8, "release": 2.0,
        "cutoff": 800, "osc_type": "sine", "resonance": 0.0,
        "lfo_rate": 3.0, "lfo_depth": 0.0, "noise_mix": 0.0,
        "duty_cycle": 0.5, "noise_type": "white"},
    2: {"attack": 0.05, "decay": 0.1, "sustain": 0.9, "release": 0.3,
        "cutoff": 3000, "osc_type": "sawtooth", "resonance": 2.0,
        "lfo_rate": 5.0, "lfo_depth": 0.02, "noise_mix": 0.0,
        "duty_cycle": 0.5, "noise_type": "white"},
    3: {"attack": 0.01, "decay": 0.05, "sustain": 0.8, "release": 0.2,
        "cutoff": 400, "osc_type": "square", "resonance": 3.0,
        "lfo_rate": 2.0, "lfo_depth": 0.01, "noise_mix": 0.2,
        "duty_cycle": 0.3, "noise_type": "white"},
    4: {"attack": 0.2, "decay": 0.3, "sustain": 0.6, "release": 1.0,
        "cutoff": 1500, "osc_type": "triangle", "resonance": 1.0,
        "lfo_rate": 4.0, "lfo_depth": 0.03, "noise_mix": 0.7,
        "duty_cycle": 0.5, "noise_type": "pink"},
    5: {"attack": 0.01, "decay": 0.2, "sustain": 0.0, "release": 0.5,
        "cutoff": 2500, "osc_type": "sawtooth", "resonance": 0.5,
        "lfo_rate": 6.0, "lfo_depth": 0.01, "noise_mix": 0.1,
        "duty_cycle": 0.5, "noise_type": "white"}
}
# プリセットは上書き不可（固定）
presets = default_presets.copy()

# 一時保存用の音色スロット（A, B：上書き可能）
temp_tones = {"A": None, "B": None}

# ---------------------------
# Tkinter GUI部分
# ---------------------------
class SynthGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ポリフォニックシンセ")
        self.geometry("800x850")
        # メインフレームを2列に分割（左：コントロール＋プリセット、一時保存、右：波形・スペクトル表示）
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.control_frame = ttk.Frame(main_frame)
        self.control_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        self.display_frame = ttk.Frame(main_frame)
        self.display_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        # 左側： コントロールとプリセット、一時保存
        self.create_controls(self.control_frame)
        self.create_preset_controls(self.control_frame)
        self.create_temp_save_controls(self.control_frame)
        # 右側： 波形表示とスペクトル表示
        self.create_waveform_display(self.display_frame)
        self.create_spectrum_display(self.display_frame)
        # 下部： ピアノ鍵盤
        self.create_keyboard()
        self.bind("<KeyPress>", self.on_key_press)
        self.bind("<KeyRelease>", self.on_key_release)
        # PCキーボード用マッピング（白鍵＋黒鍵）
        self.pc_key_map = {
            'z': 60, 's': 61, 'x': 62, 'd': 63, 'c': 64,
            'v': 65, 'g': 66, 'b': 67, 'h': 68, 'n': 69, 'j': 70, 'm': 71
        }
        self.active_keys = set()
        self.after(50, self.update_waveform)
        self.after(100, self.update_spectrum)

    def create_controls(self, parent):
        container = ttk.LabelFrame(parent, text="コントロール")
        container.pack(fill="x", padx=5, pady=5)
        # ADSR セクション
        adsr_frame = ttk.LabelFrame(container, text="ADSR（アンプエンベロープ）")
        adsr_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.attack_var = tk.DoubleVar(value=poly_synth.attack)
        ttk.Label(adsr_frame, text="Attack").grid(row=0, column=0, sticky="w")
        attack_slider = tk.Scale(adsr_frame, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL,
                                 variable=self.attack_var, command=self.update_attack)
        attack_slider.grid(row=0, column=1)
        self.decay_var = tk.DoubleVar(value=poly_synth.decay)
        ttk.Label(adsr_frame, text="Decay").grid(row=1, column=0, sticky="w")
        decay_slider = tk.Scale(adsr_frame, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL,
                                variable=self.decay_var, command=self.update_decay)
        decay_slider.grid(row=1, column=1)
        self.sustain_var = tk.DoubleVar(value=poly_synth.sustain)
        ttk.Label(adsr_frame, text="Sustain").grid(row=2, column=0, sticky="w")
        sustain_slider = tk.Scale(adsr_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                  variable=self.sustain_var, command=self.update_sustain)
        sustain_slider.grid(row=2, column=1)
        self.release_var = tk.DoubleVar(value=poly_synth.release)
        ttk.Label(adsr_frame, text="Release").grid(row=3, column=0, sticky="w")
        release_slider = tk.Scale(adsr_frame, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL,
                                  variable=self.release_var, command=self.update_release)
        release_slider.grid(row=3, column=1)
        # Filter セクション
        filter_frame = ttk.LabelFrame(container, text="フィルター")
        filter_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.cutoff_var = tk.DoubleVar(value=poly_synth.cutoff)
        ttk.Label(filter_frame, text="カットオフ（Hz）").grid(row=0, column=0, sticky="w")
        cutoff_slider = tk.Scale(filter_frame, from_=20.0, to=5000.0, resolution=1, orient=tk.HORIZONTAL,
                                 variable=self.cutoff_var, command=self.update_cutoff)
        cutoff_slider.grid(row=0, column=1)
        ttk.Label(filter_frame, text="レゾナンス").grid(row=1, column=0, sticky="w")
        self.resonance_var = tk.DoubleVar(value=poly_synth.resonance)
        resonance_slider = tk.Scale(filter_frame, from_=0.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
                                    variable=self.resonance_var, command=self.update_resonance)
        resonance_slider.grid(row=1, column=1)
        # Oscillator セクション
        osc_frame = ttk.LabelFrame(container, text="オシレーター")
        osc_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        ttk.Label(osc_frame, text="波形種別").grid(row=0, column=0, sticky="w")
        self.osc_types = ["sine", "triangle", "square", "sawtooth"]
        self.osc_var = tk.StringVar(value=poly_synth.osc_type)
        osc_menu = tk.OptionMenu(osc_frame, self.osc_var, *self.osc_types, command=self.update_osc_type)
        osc_menu.grid(row=0, column=1)
        ttk.Label(osc_frame, text="ノイズミックス").grid(row=1, column=0, sticky="w")
        self.noise_mix_var = tk.DoubleVar(value=poly_synth.noise_mix)
        noise_mix_slider = tk.Scale(osc_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                    variable=self.noise_mix_var, command=self.update_noise_mix)
        noise_mix_slider.grid(row=1, column=1)
        ttk.Label(osc_frame, text="デューティー比").grid(row=2, column=0, sticky="w")
        self.duty_cycle_var = tk.DoubleVar(value=poly_synth.duty_cycle)
        duty_cycle_slider = tk.Scale(osc_frame, from_=0.1, to=0.9, resolution=0.01, orient=tk.HORIZONTAL,
                                     variable=self.duty_cycle_var, command=self.update_duty_cycle)
        duty_cycle_slider.grid(row=2, column=1)
        ttk.Label(osc_frame, text="ノイズタイプ").grid(row=3, column=0, sticky="w")
        self.noise_types = ["white", "pink"]
        self.noise_type_var = tk.StringVar(value=poly_synth.noise_type)
        noise_menu = tk.OptionMenu(osc_frame, self.noise_type_var, *self.noise_types, command=self.update_noise_type)
        noise_menu.grid(row=3, column=1)
        # LFO セクション
        lfo_frame = ttk.LabelFrame(container, text="LFO（ビブラート）")
        lfo_frame.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        ttk.Label(lfo_frame, text="レート（Hz）").grid(row=0, column=0, sticky="w")
        self.lfo_rate_var = tk.DoubleVar(value=poly_synth.lfo_rate)
        lfo_rate_slider = tk.Scale(lfo_frame, from_=0.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
                                   variable=self.lfo_rate_var, command=self.update_lfo_rate)
        lfo_rate_slider.grid(row=0, column=1)
        ttk.Label(lfo_frame, text="デプス").grid(row=1, column=0, sticky="w")
        self.lfo_depth_var = tk.DoubleVar(value=poly_synth.lfo_depth)
        lfo_depth_slider = tk.Scale(lfo_frame, from_=0.0, to=0.05, resolution=0.001, orient=tk.HORIZONTAL,
                                    variable=self.lfo_depth_var, command=self.update_lfo_depth)
        lfo_depth_slider.grid(row=1, column=1)

    def create_preset_controls(self, parent):
        preset_frame = ttk.LabelFrame(parent, text="プリセット（上書き不可）")
        preset_frame.pack(fill="x", padx=5, pady=5)
        button_frame = ttk.Frame(preset_frame)
        button_frame.pack(side="top", padx=5, pady=5)
        self.preset_buttons = {}
        for i in range(1, 6):
            btn = ttk.Button(button_frame, text=f"プリセット {i}", command=lambda i=i: self.load_preset_slot(i))
            btn.grid(row=0, column=i-1, padx=3)
            self.preset_buttons[i] = btn
        self.preset_label = ttk.Label(preset_frame, text="読み込み中のプリセット: なし")
        self.preset_label.pack(side="top", padx=5, pady=5)

    def create_temp_save_controls(self, parent):
        temp_frame = ttk.LabelFrame(parent, text="一時保存（上書き可能）")
        temp_frame.pack(fill="x", padx=5, pady=5)
        self.tempA = None
        self.tempB = None
        btn_frame = ttk.Frame(temp_frame)
        btn_frame.pack(side="top", padx=5, pady=5)
        self.save_tempA_btn = ttk.Button(btn_frame, text="一時保存 A 保存", command=lambda: self.save_temp("A"))
        self.save_tempA_btn.grid(row=0, column=0, padx=3)
        self.load_tempA_btn = ttk.Button(btn_frame, text="一時保存 A 読み込み", command=lambda: self.load_temp("A"))
        self.load_tempA_btn.grid(row=0, column=1, padx=3)
        self.save_tempB_btn = ttk.Button(btn_frame, text="一時保存 B 保存", command=lambda: self.save_temp("B"))
        self.save_tempB_btn.grid(row=0, column=2, padx=3)
        self.load_tempB_btn = ttk.Button(btn_frame, text="一時保存 B 読み込み", command=lambda: self.load_temp("B"))
        self.load_tempB_btn.grid(row=0, column=3, padx=3)
        self.temp_label = ttk.Label(temp_frame, text="一時保存状態: A: なし, B: なし")
        self.temp_label.pack(side="top", padx=5, pady=5)

    def create_waveform_display(self, parent):
        title = ttk.Label(parent, text="波形表示", font=("Arial", 10, "bold"))
        title.pack()
        self.wave_canvas = tk.Canvas(parent, width=450, height=150, bg="black")
        self.wave_canvas.pack(pady=5)

    def create_spectrum_display(self, parent):
        title = ttk.Label(parent, text="スペクトル表示", font=("Arial", 10, "bold"))
        title.pack()
        self.spec_canvas = tk.Canvas(parent, width=450, height=200, bg="black")
        self.spec_canvas.pack(pady=5)

    def create_keyboard(self):
        kb_frame = tk.Frame(self)
        kb_frame.pack(pady=10)
        self.piano_canvas = tk.Canvas(kb_frame, width=800, height=150, bg="gray")
        self.piano_canvas.pack()
        # 白鍵（C, D, E, F, G, A, B）
        white_keys = [60, 62, 64, 65, 67, 69, 71]
        white_labels = ["C", "D", "E", "F", "G", "A", "B"]
        white_width = 100
        white_height = 150
        self.white_key_ids = {}
        for i, note in enumerate(white_keys):
            x = i * white_width
            rect = self.piano_canvas.create_rectangle(x, 0, x+white_width, white_height, fill="white", outline="black")
            self.piano_canvas.create_text(x+white_width/2, white_height-20, text=white_labels[i], font=("Arial", 12))
            self.white_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note: self.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note: self.on_note_release(n))
        # 黒鍵（C#, D#, F#, G#, A#）
        black_keys = [61, 63, 66, 68, 70]
        black_labels = ["C#", "D#", "F#", "G#", "A#"]
        black_width = 60
        black_height = 90
        black_positions = [75, 175, 375, 475, 575]  # 調整済みの位置
        self.black_key_ids = {}
        for pos, note, label in zip(black_positions, black_keys, black_labels):
            rect = self.piano_canvas.create_rectangle(pos, 0, pos+black_width, black_height, fill="black", outline="black")
            self.piano_canvas.create_text(pos+black_width/2, black_height-20, text=label, fill="white", font=("Arial", 10))
            self.black_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note: self.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note: self.on_note_release(n))

    def update_attack(self, val): poly_synth.set_attack(float(val))
    def update_decay(self, val): poly_synth.set_decay(float(val))
    def update_sustain(self, val): poly_synth.set_sustain(float(val))
    def update_release(self, val): poly_synth.set_release(float(val))
    def update_cutoff(self, val): poly_synth.set_cutoff(float(val))
    def update_osc_type(self, val): poly_synth.set_osc_type(val)
    def update_resonance(self, val): poly_synth.set_resonance(float(val))
    def update_lfo_rate(self, val): poly_synth.set_lfo_rate(float(val))
    def update_lfo_depth(self, val): poly_synth.set_lfo_depth(float(val))
    def update_noise_mix(self, val): poly_synth.set_noise_mix(float(val))
    def update_duty_cycle(self, val): poly_synth.set_duty_cycle(float(val))
    def update_noise_type(self, val): poly_synth.set_noise_type(val)

    def on_note_press(self, note): poly_synth.note_on(note)
    def on_note_release(self, note): poly_synth.note_off(note)

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

    def update_waveform(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        xdata = np.linspace(0, 450, WAVEFORM_BUFFER_SIZE)
        ydata = 75 - (buf * 75)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.wave_canvas.delete("all")
        self.wave_canvas.create_line(points, fill="green")
        self.after(50, self.update_waveform)

    def update_spectrum(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        window = np.hamming(len(buf))
        fft_result = fft.rfft(buf * window)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-6)
        freqs = fft.rfftfreq(len(buf), 1.0 / SAMPLE_RATE)
        max_disp_freq = 5000
        idx_max = np.searchsorted(freqs, max_disp_freq)
        freqs = freqs[:idx_max]
        magnitude_db = magnitude_db[:idx_max]
        canvas_width = 450
        canvas_height = 200
        xdata = np.linspace(0, canvas_width, len(freqs))
        # 縦軸を -120 dB ～ 0 dB に設定
        ydata = canvas_height - ((magnitude_db + 120) / 120 * canvas_height)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.spec_canvas.delete("all")
        self.spec_canvas.create_line(points, fill="yellow")
        for f in range(0, max_disp_freq+1, 1000):
            x = f / max_disp_freq * canvas_width
            self.spec_canvas.create_line(x, canvas_height, x, canvas_height-10, fill="white")
            self.spec_canvas.create_text(x, canvas_height-15, text=str(f)+"Hz", fill="white", font=("Arial",9))
        self.after(100, self.update_spectrum)

    def create_preset_controls(self, parent):
        preset_frame = ttk.LabelFrame(parent, text="プリセット（上書き不可）")
        preset_frame.pack(fill="x", padx=5, pady=5)
        button_frame = ttk.Frame(preset_frame)
        button_frame.pack(side="top", padx=5, pady=5)
        self.preset_buttons = {}
        for i in range(1, 6):
            btn = ttk.Button(button_frame, text=f"プリセット {i}", command=lambda i=i: self.load_preset_slot(i))
            btn.grid(row=0, column=i-1, padx=3)
            self.preset_buttons[i] = btn
        self.preset_label = ttk.Label(preset_frame, text="読み込み中のプリセット: なし")
        self.preset_label.pack(side="top", padx=5, pady=5)

    def create_temp_save_controls(self, parent):
        temp_frame = ttk.LabelFrame(parent, text="一時保存（上書き可能）")
        temp_frame.pack(fill="x", padx=5, pady=5)
        btn_frame = ttk.Frame(temp_frame)
        btn_frame.pack(side="top", padx=5, pady=5)
        self.tempA = None
        self.tempB = None
        self.save_tempA_btn = ttk.Button(btn_frame, text="一時保存 A 保存", command=lambda: self.save_temp("A"))
        self.save_tempA_btn.grid(row=0, column=0, padx=3)
        self.load_tempA_btn = ttk.Button(btn_frame, text="一時保存 A 読み込み", command=lambda: self.load_temp("A"))
        self.load_tempA_btn.grid(row=0, column=1, padx=3)
        self.save_tempB_btn = ttk.Button(btn_frame, text="一時保存 B 保存", command=lambda: self.save_temp("B"))
        self.save_tempB_btn.grid(row=0, column=2, padx=3)
        self.load_tempB_btn = ttk.Button(btn_frame, text="一時保存 B 読み込み", command=lambda: self.load_temp("B"))
        self.load_tempB_btn.grid(row=0, column=3, padx=3)
        self.temp_label = ttk.Label(temp_frame, text="一時保存状態: A: なし, B: なし")
        self.temp_label.pack(side="top", padx=5, pady=5)

    def create_waveform_display(self, parent):
        title = ttk.Label(parent, text="波形表示", font=("Arial", 10, "bold"))
        title.pack()
        self.wave_canvas = tk.Canvas(parent, width=450, height=150, bg="black")
        self.wave_canvas.pack(pady=5)

    def create_spectrum_display(self, parent):
        title = ttk.Label(parent, text="スペクトル表示", font=("Arial", 10, "bold"))
        title.pack()
        self.spec_canvas = tk.Canvas(parent, width=450, height=200, bg="black")
        self.spec_canvas.pack(pady=5)

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

    def update_waveform(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        xdata = np.linspace(0, 450, WAVEFORM_BUFFER_SIZE)
        ydata = 75 - (buf * 75)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.wave_canvas.delete("all")
        self.wave_canvas.create_line(points, fill="green")
        self.after(50, self.update_waveform)

    def update_spectrum(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        window = np.hamming(len(buf))
        fft_result = fft.rfft(buf * window)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-6)
        freqs = fft.rfftfreq(len(buf), 1.0 / SAMPLE_RATE)
        max_disp_freq = 5000
        idx_max = np.searchsorted(freqs, max_disp_freq)
        freqs = freqs[:idx_max]
        magnitude_db = magnitude_db[:idx_max]
        canvas_width = 450
        canvas_height = 200
        xdata = np.linspace(0, canvas_width, len(freqs))
        ydata = canvas_height - ((magnitude_db + 120) / 120 * canvas_height)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.spec_canvas.delete("all")
        self.spec_canvas.create_line(points, fill="yellow")
        for f in range(0, max_disp_freq+1, 1000):
            x = f / max_disp_freq * canvas_width
            self.spec_canvas.create_line(x, canvas_height, x, canvas_height-10, fill="white")
            self.spec_canvas.create_text(x, canvas_height-15, text=f"{f}Hz", fill="white", font=("Arial",9))
        self.after(100, self.update_spectrum)

    def save_temp(self, slot: str):
        preset = {
            "attack": poly_synth.attack,
            "decay": poly_synth.decay,
            "sustain": poly_synth.sustain,
            "release": poly_synth.release,
            "cutoff": poly_synth.cutoff,
            "osc_type": poly_synth.osc_type,
            "resonance": poly_synth.resonance,
            "lfo_rate": poly_synth.lfo_rate,
            "lfo_depth": poly_synth.lfo_depth,
            "noise_mix": poly_synth.noise_mix,
            "duty_cycle": poly_synth.duty_cycle,
            "noise_type": poly_synth.noise_type
        }
        temp_tones[slot] = preset
        self.update_temp_label()

    def load_temp(self, slot: str):
        if temp_tones[slot]:
            preset = temp_tones[slot]
            poly_synth.set_attack(preset["attack"])
            poly_synth.set_decay(preset["decay"])
            poly_synth.set_sustain(preset["sustain"])
            poly_synth.set_release(preset["release"])
            poly_synth.set_cutoff(preset["cutoff"])
            poly_synth.set_osc_type(preset["osc_type"])
            poly_synth.set_resonance(preset["resonance"])
            poly_synth.set_lfo_rate(preset["lfo_rate"])
            poly_synth.set_lfo_depth(preset["lfo_depth"])
            poly_synth.set_noise_mix(preset["noise_mix"])
            poly_synth.set_duty_cycle(preset["duty_cycle"])
            poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            self.osc_var.set(preset["osc_type"])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            self.noise_type_var.set(preset["noise_type"])
            self.temp_label.config(text=f"一時保存: A: { 'あり' if temp_tones['A'] else 'なし' }, B: { 'あり' if temp_tones['B'] else 'なし' }")
        else:
            self.temp_label.config(text=f"一時保存スロット {slot} に保存された音色はありません。")

    def create_preset_controls(self, parent):
        preset_frame = ttk.LabelFrame(parent, text="プリセット（上書き不可）")
        preset_frame.pack(fill="x", padx=5, pady=5)
        button_frame = ttk.Frame(preset_frame)
        button_frame.pack(side="top", padx=5, pady=5)
        self.preset_buttons = {}
        for i in range(1, 6):
            btn = ttk.Button(button_frame, text=f"プリセット {i}", command=lambda i=i: self.load_preset_slot(i))
            btn.grid(row=0, column=i-1, padx=3)
            self.preset_buttons[i] = btn
        self.preset_label = ttk.Label(preset_frame, text="読み込み中のプリセット: なし")
        self.preset_label.pack(side="top", padx=5, pady=5)

    def create_temp_save_controls(self, parent):
        temp_frame = ttk.LabelFrame(parent, text="一時保存（上書き可能）")
        temp_frame.pack(fill="x", padx=5, pady=5)
        btn_frame = ttk.Frame(temp_frame)
        btn_frame.pack(side="top", padx=5, pady=5)
        self.save_tempA_btn = ttk.Button(btn_frame, text="一時保存 A 保存", command=lambda: self.save_temp("A"))
        self.save_tempA_btn.grid(row=0, column=0, padx=3)
        self.load_tempA_btn = ttk.Button(btn_frame, text="一時保存 A 読み込み", command=lambda: self.load_temp("A"))
        self.load_tempA_btn.grid(row=0, column=1, padx=3)
        self.save_tempB_btn = ttk.Button(btn_frame, text="一時保存 B 保存", command=lambda: self.save_temp("B"))
        self.save_tempB_btn.grid(row=0, column=2, padx=3)
        self.load_tempB_btn = ttk.Button(btn_frame, text="一時保存 B 読み込み", command=lambda: self.load_temp("B"))
        self.load_tempB_btn.grid(row=0, column=3, padx=3)
        self.temp_label = ttk.Label(temp_frame, text="一時保存状態: A: なし, B: なし")
        self.temp_label.pack(side="top", padx=5, pady=5)

    def create_waveform_display(self, parent):
        title = ttk.Label(parent, text="波形表示", font=("Arial", 10, "bold"))
        title.pack()
        self.wave_canvas = tk.Canvas(parent, width=450, height=150, bg="black")
        self.wave_canvas.pack(pady=5)

    def create_spectrum_display(self, parent):
        title = ttk.Label(parent, text="スペクトル表示", font=("Arial", 10, "bold"))
        title.pack()
        self.spec_canvas = tk.Canvas(parent, width=450, height=200, bg="black")
        self.spec_canvas.pack(pady=5)

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

    def update_waveform(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        xdata = np.linspace(0, 450, WAVEFORM_BUFFER_SIZE)
        ydata = 75 - (buf * 75)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.wave_canvas.delete("all")
        self.wave_canvas.create_line(points, fill="green")
        self.after(50, self.update_waveform)

    def update_spectrum(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        window = np.hamming(len(buf))
        fft_result = fft.rfft(buf * window)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-6)
        freqs = fft.rfftfreq(len(buf), 1.0 / SAMPLE_RATE)
        max_disp_freq = 5000
        idx_max = np.searchsorted(freqs, max_disp_freq)
        freqs = freqs[:idx_max]
        magnitude_db = magnitude_db[:idx_max]
        canvas_width = 450
        canvas_height = 200
        xdata = np.linspace(0, canvas_width, len(freqs))
        ydata = canvas_height - ((magnitude_db + 120) / 120 * canvas_height)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.spec_canvas.delete("all")
        self.spec_canvas.create_line(points, fill="yellow")
        for f in range(0, max_disp_freq+1, 1000):
            x = f / max_disp_freq * canvas_width
            self.spec_canvas.create_line(x, canvas_height, x, canvas_height-10, fill="white")
            self.spec_canvas.create_text(x, canvas_height-15, text=f"{f}Hz", fill="white", font=("Arial",9))
        self.after(100, self.update_spectrum)

    def save_preset(self):
        # プリセットは上書き不可（固定）
        pass  # 既定のプリセットは default_presets を使用するため保存処理はなし

    def load_preset_slot(self, slot: int):
        if slot in presets:
            preset = presets[slot]
            poly_synth.set_attack(preset["attack"])
            poly_synth.set_decay(preset["decay"])
            poly_synth.set_sustain(preset["sustain"])
            poly_synth.set_release(preset["release"])
            poly_synth.set_cutoff(preset["cutoff"])
            poly_synth.set_osc_type(preset["osc_type"])
            poly_synth.set_resonance(preset["resonance"])
            poly_synth.set_lfo_rate(preset["lfo_rate"])
            poly_synth.set_lfo_depth(preset["lfo_depth"])
            poly_synth.set_noise_mix(preset["noise_mix"])
            poly_synth.set_duty_cycle(preset["duty_cycle"])
            poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            self.osc_var.set(preset["osc_type"])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            self.noise_type_var.set(preset["noise_type"])
            self.preset_label.config(text=f"プリセット {slot} を読み込みました")
        else:
            self.preset_label.config(text=f"プリセット {slot} は存在しません")

    def save_temp(self, slot: str):
        preset = {
            "attack": poly_synth.attack,
            "decay": poly_synth.decay,
            "sustain": poly_synth.sustain,
            "release": poly_synth.release,
            "cutoff": poly_synth.cutoff,
            "osc_type": poly_synth.osc_type,
            "resonance": poly_synth.resonance,
            "lfo_rate": poly_synth.lfo_rate,
            "lfo_depth": poly_synth.lfo_depth,
            "noise_mix": poly_synth.noise_mix,
            "duty_cycle": poly_synth.duty_cycle,
            "noise_type": poly_synth.noise_type
        }
        temp_tones[slot] = preset
        self.update_temp_label()

    def load_temp(self, slot: str):
        if temp_tones[slot]:
            preset = temp_tones[slot]
            poly_synth.set_attack(preset["attack"])
            poly_synth.set_decay(preset["decay"])
            poly_synth.set_sustain(preset["sustain"])
            poly_synth.set_release(preset["release"])
            poly_synth.set_cutoff(preset["cutoff"])
            poly_synth.set_osc_type(preset["osc_type"])
            poly_synth.set_resonance(preset["resonance"])
            poly_synth.set_lfo_rate(preset["lfo_rate"])
            poly_synth.set_lfo_depth(preset["lfo_depth"])
            poly_synth.set_noise_mix(preset["noise_mix"])
            poly_synth.set_duty_cycle(preset["duty_cycle"])
            poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            self.osc_var.set(preset["osc_type"])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            self.noise_type_var.set(preset["noise_type"])
            self.temp_label.config(text=f"一時保存 {slot} を読み込みました")
        else:
            self.temp_label.config(text=f"一時保存 {slot} に保存された音色はありません")

    def create_waveform_display(self, parent):
        title = ttk.Label(parent, text="波形表示", font=("Arial", 10, "bold"))
        title.pack()
        self.wave_canvas = tk.Canvas(parent, width=450, height=150, bg="black")
        self.wave_canvas.pack(pady=5)

    def create_spectrum_display(self, parent):
        title = ttk.Label(parent, text="スペクトル表示", font=("Arial", 10, "bold"))
        title.pack()
        self.spec_canvas = tk.Canvas(parent, width=450, height=200, bg="black")
        self.spec_canvas.pack(pady=5)

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

    def update_waveform(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        xdata = np.linspace(0, 450, WAVEFORM_BUFFER_SIZE)
        ydata = 75 - (buf * 75)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.wave_canvas.delete("all")
        self.wave_canvas.create_line(points, fill="green")
        self.after(50, self.update_waveform)

    def update_spectrum(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        window = np.hamming(len(buf))
        fft_result = fft.rfft(buf * window)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-6)
        freqs = fft.rfftfreq(len(buf), 1.0 / SAMPLE_RATE)
        max_disp_freq = 5000
        idx_max = np.searchsorted(freqs, max_disp_freq)
        freqs = freqs[:idx_max]
        magnitude_db = magnitude_db[:idx_max]
        canvas_width = 450
        canvas_height = 200
        xdata = np.linspace(0, canvas_width, len(freqs))
        ydata = canvas_height - ((magnitude_db + 120) / 120 * canvas_height)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.spec_canvas.delete("all")
        self.spec_canvas.create_line(points, fill="yellow")
        for f in range(0, max_disp_freq+1, 1000):
            x = f / max_disp_freq * canvas_width
            self.spec_canvas.create_line(x, canvas_height, x, canvas_height-10, fill="white")
            self.spec_canvas.create_text(x, canvas_height-15, text=f"{f}Hz", fill="white", font=("Arial",9))
        self.after(100, self.update_spectrum)

    def create_preset_controls(self, parent):
        preset_frame = ttk.LabelFrame(parent, text="プリセット（上書き不可）")
        preset_frame.pack(fill="x", padx=5, pady=5)
        button_frame = ttk.Frame(preset_frame)
        button_frame.pack(side="top", padx=5, pady=5)
        self.preset_buttons = {}
        for i in range(1, 6):
            btn = ttk.Button(button_frame, text=f"プリセット {i}", command=lambda i=i: self.load_preset_slot(i))
            btn.grid(row=0, column=i-1, padx=3)
            self.preset_buttons[i] = btn
        self.preset_label = ttk.Label(preset_frame, text="読み込み中のプリセット: なし")
        self.preset_label.pack(side="top", padx=5, pady=5)

    def create_temp_save_controls(self, parent):
        temp_frame = ttk.LabelFrame(parent, text="一時保存（上書き可能）")
        temp_frame.pack(fill="x", padx=5, pady=5)
        btn_frame = ttk.Frame(temp_frame)
        btn_frame.pack(side="top", padx=5, pady=5)
        self.save_tempA_btn = ttk.Button(btn_frame, text="一時保存 A 保存", command=lambda: self.save_temp("A"))
        self.save_tempA_btn.grid(row=0, column=0, padx=3)
        self.load_tempA_btn = ttk.Button(btn_frame, text="一時保存 A 読み込み", command=lambda: self.load_temp("A"))
        self.load_tempA_btn.grid(row=0, column=1, padx=3)
        self.save_tempB_btn = ttk.Button(btn_frame, text="一時保存 B 保存", command=lambda: self.save_temp("B"))
        self.save_tempB_btn.grid(row=0, column=2, padx=3)
        self.load_tempB_btn = ttk.Button(btn_frame, text="一時保存 B 読み込み", command=lambda: self.load_temp("B"))
        self.load_tempB_btn.grid(row=0, column=3, padx=3)
        self.temp_label = ttk.Label(temp_frame, text="一時保存状態: A: なし, B: なし")
        self.temp_label.pack(side="top", padx=5, pady=5)

    def update_temp_label(self):
        a_status = "あり" if temp_tones["A"] else "なし"
        b_status = "あり" if temp_tones["B"] else "なし"
        self.temp_label.config(text=f"一時保存状態: A: {a_status}, B: {b_status}")

    def save_preset(self):
        # プリセットは上書き不可（既定のプリセットを使用）
        pass

    def load_preset_slot(self, slot: int):
        if slot in presets:
            preset = presets[slot]
            poly_synth.set_attack(preset["attack"])
            poly_synth.set_decay(preset["decay"])
            poly_synth.set_sustain(preset["sustain"])
            poly_synth.set_release(preset["release"])
            poly_synth.set_cutoff(preset["cutoff"])
            poly_synth.set_osc_type(preset["osc_type"])
            poly_synth.set_resonance(preset["resonance"])
            poly_synth.set_lfo_rate(preset["lfo_rate"])
            poly_synth.set_lfo_depth(preset["lfo_depth"])
            poly_synth.set_noise_mix(preset["noise_mix"])
            poly_synth.set_duty_cycle(preset["duty_cycle"])
            poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            self.osc_var.set(preset["osc_type"])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            self.noise_type_var.set(preset["noise_type"])
            self.preset_label.config(text=f"プリセット {slot} を読み込みました")
        else:
            self.preset_label.config(text=f"プリセット {slot} は存在しません")

    def save_temp(self, slot: str):
        preset = {
            "attack": poly_synth.attack,
            "decay": poly_synth.decay,
            "sustain": poly_synth.sustain,
            "release": poly_synth.release,
            "cutoff": poly_synth.cutoff,
            "osc_type": poly_synth.osc_type,
            "resonance": poly_synth.resonance,
            "lfo_rate": poly_synth.lfo_rate,
            "lfo_depth": poly_synth.lfo_depth,
            "noise_mix": poly_synth.noise_mix,
            "duty_cycle": poly_synth.duty_cycle,
            "noise_type": poly_synth.noise_type
        }
        temp_tones[slot] = preset
        self.update_temp_label()

    def load_temp(self, slot: str):
        if temp_tones[slot]:
            preset = temp_tones[slot]
            poly_synth.set_attack(preset["attack"])
            poly_synth.set_decay(preset["decay"])
            poly_synth.set_sustain(preset["sustain"])
            poly_synth.set_release(preset["release"])
            poly_synth.set_cutoff(preset["cutoff"])
            poly_synth.set_osc_type(preset["osc_type"])
            poly_synth.set_resonance(preset["resonance"])
            poly_synth.set_lfo_rate(preset["lfo_rate"])
            poly_synth.set_lfo_depth(preset["lfo_depth"])
            poly_synth.set_noise_mix(preset["noise_mix"])
            poly_synth.set_duty_cycle(preset["duty_cycle"])
            poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            self.osc_var.set(preset["osc_type"])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            self.noise_type_var.set(preset["noise_type"])
            self.preset_label.config(text=f"一時保存 {slot} を読み込みました")
        else:
            self.preset_label.config(text=f"一時保存 {slot} に保存された音色はありません")

    def create_waveform_display(self, parent):
        title = ttk.Label(parent, text="波形表示", font=("Arial", 10, "bold"))
        title.pack()
        self.wave_canvas = tk.Canvas(parent, width=450, height=150, bg="black")
        self.wave_canvas.pack(pady=5)

    def create_spectrum_display(self, parent):
        title = ttk.Label(parent, text="スペクトル表示", font=("Arial", 10, "bold"))
        title.pack()
        self.spec_canvas = tk.Canvas(parent, width=450, height=200, bg="black")
        self.spec_canvas.pack(pady=5)

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

    def update_waveform(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        xdata = np.linspace(0, 450, WAVEFORM_BUFFER_SIZE)
        ydata = 75 - (buf * 75)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.wave_canvas.delete("all")
        self.wave_canvas.create_line(points, fill="green")
        self.after(50, self.update_waveform)

    def update_spectrum(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        window = np.hamming(len(buf))
        fft_result = fft.rfft(buf * window)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-6)
        freqs = fft.rfftfreq(len(buf), 1.0 / SAMPLE_RATE)
        max_disp_freq = 5000
        idx_max = np.searchsorted(freqs, max_disp_freq)
        freqs = freqs[:idx_max]
        magnitude_db = magnitude_db[:idx_max]
        canvas_width = 450
        canvas_height = 200
        xdata = np.linspace(0, canvas_width, len(freqs))
        ydata = canvas_height - ((magnitude_db + 120) / 120 * canvas_height)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.spec_canvas.delete("all")
        self.spec_canvas.create_line(points, fill="yellow")
        for f in range(0, max_disp_freq+1, 1000):
            x = f / max_disp_freq * canvas_width
            self.spec_canvas.create_line(x, canvas_height, x, canvas_height-10, fill="white")
            self.spec_canvas.create_text(x, canvas_height-15, text=f"{f}Hz", fill="white", font=("Arial",9))
        self.after(100, self.update_spectrum)

    def create_keyboard(self):
        kb_frame = tk.Frame(self)
        kb_frame.pack(pady=10)
        self.piano_canvas = tk.Canvas(kb_frame, width=800, height=150, bg="gray")
        self.piano_canvas.pack()
        # 白鍵
        white_keys = [60, 62, 64, 65, 67, 69, 71]
        white_labels = ["C", "D", "E", "F", "G", "A", "B"]
        white_width = 100
        white_height = 150
        self.white_key_ids = {}
        for i, note in enumerate(white_keys):
            x = i * white_width
            rect = self.piano_canvas.create_rectangle(x, 0, x+white_width, white_height, fill="white", outline="black")
            self.piano_canvas.create_text(x+white_width/2, white_height-20, text=white_labels[i], font=("Arial", 12))
            self.white_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note: self.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note: self.on_note_release(n))
        # 黒鍵
        black_keys = [61, 63, 66, 68, 70]
        black_labels = ["C#", "D#", "F#", "G#", "A#"]
        black_width = 60
        black_height = 90
        black_positions = [75, 175, 375, 475, 575]
        self.black_key_ids = {}
        for pos, note, label in zip(black_positions, black_keys, black_labels):
            rect = self.piano_canvas.create_rectangle(pos, 0, pos+black_width, black_height, fill="black", outline="black")
            self.piano_canvas.create_text(pos+black_width/2, black_height-20, text=label, fill="white", font=("Arial",10))
            self.black_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note: self.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note: self.on_note_release(n))

    def save_preset(self):
        # プリセットは固定（上書き不可）のでここでは何もしません
        pass

    def load_preset_slot(self, slot: int):
        if slot in presets:
            preset = presets[slot]
            poly_synth.set_attack(preset["attack"])
            poly_synth.set_decay(preset["decay"])
            poly_synth.set_sustain(preset["sustain"])
            poly_synth.set_release(preset["release"])
            poly_synth.set_cutoff(preset["cutoff"])
            poly_synth.set_osc_type(preset["osc_type"])
            poly_synth.set_resonance(preset["resonance"])
            poly_synth.set_lfo_rate(preset["lfo_rate"])
            poly_synth.set_lfo_depth(preset["lfo_depth"])
            poly_synth.set_noise_mix(preset["noise_mix"])
            poly_synth.set_duty_cycle(preset["duty_cycle"])
            poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            self.osc_var.set(preset["osc_type"])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            self.noise_type_var.set(preset["noise_type"])
            self.preset_label.config(text=f"プリセット {slot} を読み込みました")
        else:
            self.preset_label.config(text=f"プリセット {slot} は存在しません")

    def save_temp(self, slot: str):
        preset = {
            "attack": poly_synth.attack,
            "decay": poly_synth.decay,
            "sustain": poly_synth.sustain,
            "release": poly_synth.release,
            "cutoff": poly_synth.cutoff,
            "osc_type": poly_synth.osc_type,
            "resonance": poly_synth.resonance,
            "lfo_rate": poly_synth.lfo_rate,
            "lfo_depth": poly_synth.lfo_depth,
            "noise_mix": poly_synth.noise_mix,
            "duty_cycle": poly_synth.duty_cycle,
            "noise_type": poly_synth.noise_type
        }
        temp_tones[slot] = preset
        self.update_temp_label()

    def load_temp(self, slot: str):
        if temp_tones[slot]:
            preset = temp_tones[slot]
            poly_synth.set_attack(preset["attack"])
            poly_synth.set_decay(preset["decay"])
            poly_synth.set_sustain(preset["sustain"])
            poly_synth.set_release(preset["release"])
            poly_synth.set_cutoff(preset["cutoff"])
            poly_synth.set_osc_type(preset["osc_type"])
            poly_synth.set_resonance(preset["resonance"])
            poly_synth.set_lfo_rate(preset["lfo_rate"])
            poly_synth.set_lfo_depth(preset["lfo_depth"])
            poly_synth.set_noise_mix(preset["noise_mix"])
            poly_synth.set_duty_cycle(preset["duty_cycle"])
            poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            self.osc_var.set(preset["osc_type"])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            self.noise_type_var.set(preset["noise_type"])
            self.preset_label.config(text=f"一時保存 {slot} を読み込みました")
        else:
            self.preset_label.config(text=f"一時保存 {slot} に保存された音色はありません")

    def create_waveform_display(self, parent):
        title = ttk.Label(parent, text="波形表示", font=("Arial", 10, "bold"))
        title.pack()
        self.wave_canvas = tk.Canvas(parent, width=450, height=150, bg="black")
        self.wave_canvas.pack(pady=5)

    def create_spectrum_display(self, parent):
        title = ttk.Label(parent, text="スペクトル表示", font=("Arial", 10, "bold"))
        title.pack()
        self.spec_canvas = tk.Canvas(parent, width=450, height=200, bg="black")
        self.spec_canvas.pack(pady=5)

    def create_keyboard(self):
        kb_frame = tk.Frame(self)
        kb_frame.pack(pady=10)
        self.piano_canvas = tk.Canvas(kb_frame, width=800, height=150, bg="gray")
        self.piano_canvas.pack()
        # 白鍵
        white_keys = [60, 62, 64, 65, 67, 69, 71]
        white_labels = ["C", "D", "E", "F", "G", "A", "B"]
        white_width = 100
        white_height = 150
        self.white_key_ids = {}
        for i, note in enumerate(white_keys):
            x = i * white_width
            rect = self.piano_canvas.create_rectangle(x, 0, x+white_width, white_height, fill="white", outline="black")
            self.piano_canvas.create_text(x+white_width/2, white_height-20, text=white_labels[i], font=("Arial", 12))
            self.white_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note: self.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note: self.on_note_release(n))
        # 黒鍵
        black_keys = [61, 63, 66, 68, 70]
        black_labels = ["C#", "D#", "F#", "G#", "A#"]
        black_width = 60
        black_height = 90
        black_positions = [75, 175, 375, 475, 575]
        self.black_key_ids = {}
        for pos, note, label in zip(black_positions, black_keys, black_labels):
            rect = self.piano_canvas.create_rectangle(pos, 0, pos+black_width, black_height, fill="black", outline="black")
            self.piano_canvas.create_text(pos+black_width/2, black_height-20, text=label, fill="white", font=("Arial",10))
            self.black_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note: self.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note: self.on_note_release(n))

    def update_waveform(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        xdata = np.linspace(0, 450, WAVEFORM_BUFFER_SIZE)
        ydata = 75 - (buf * 75)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.wave_canvas.delete("all")
        self.wave_canvas.create_line(points, fill="green")
        self.after(50, self.update_waveform)

    def update_spectrum(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        window = np.hamming(len(buf))
        fft_result = fft.rfft(buf * window)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-6)
        freqs = fft.rfftfreq(len(buf), 1.0 / SAMPLE_RATE)
        max_disp_freq = 5000
        idx_max = np.searchsorted(freqs, max_disp_freq)
        freqs = freqs[:idx_max]
        magnitude_db = magnitude_db[:idx_max]
        canvas_width = 450
        canvas_height = 200
        xdata = np.linspace(0, canvas_width, len(freqs))
        ydata = canvas_height - ((magnitude_db + 120) / 120 * canvas_height)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.spec_canvas.delete("all")
        self.spec_canvas.create_line(points, fill="yellow")
        for f in range(0, max_disp_freq+1, 1000):
            x = f / max_disp_freq * canvas_width
            self.spec_canvas.create_line(x, canvas_height, x, canvas_height-10, fill="white")
            self.spec_canvas.create_text(x, canvas_height-15, text=f"{f}Hz", fill="white", font=("Arial",9))
        self.after(100, self.update_spectrum)

    def save_preset(self):
        # プリセットは固定（上書き不可）ので処理は行わない
        pass

    def load_preset_slot(self, slot: int):
        if slot in presets:
            preset = presets[slot]
            poly_synth.set_attack(preset["attack"])
            poly_synth.set_decay(preset["decay"])
            poly_synth.set_sustain(preset["sustain"])
            poly_synth.set_release(preset["release"])
            poly_synth.set_cutoff(preset["cutoff"])
            poly_synth.set_osc_type(preset["osc_type"])
            poly_synth.set_resonance(preset["resonance"])
            poly_synth.set_lfo_rate(preset["lfo_rate"])
            poly_synth.set_lfo_depth(preset["lfo_depth"])
            poly_synth.set_noise_mix(preset["noise_mix"])
            poly_synth.set_duty_cycle(preset["duty_cycle"])
            poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            self.osc_var.set(preset["osc_type"])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            self.noise_type_var.set(preset["noise_type"])
            self.preset_label.config(text=f"プリセット {slot} を読み込みました")
        else:
            self.preset_label.config(text=f"プリセット {slot} は存在しません")

    def save_temp(self, slot: str):
        preset = {
            "attack": poly_synth.attack,
            "decay": poly_synth.decay,
            "sustain": poly_synth.sustain,
            "release": poly_synth.release,
            "cutoff": poly_synth.cutoff,
            "osc_type": poly_synth.osc_type,
            "resonance": poly_synth.resonance,
            "lfo_rate": poly_synth.lfo_rate,
            "lfo_depth": poly_synth.lfo_depth,
            "noise_mix": poly_synth.noise_mix,
            "duty_cycle": poly_synth.duty_cycle,
            "noise_type": poly_synth.noise_type
        }
        temp_tones[slot] = preset
        self.update_temp_label()

    def load_temp(self, slot: str):
        if temp_tones[slot]:
            preset = temp_tones[slot]
            poly_synth.set_attack(preset["attack"])
            poly_synth.set_decay(preset["decay"])
            poly_synth.set_sustain(preset["sustain"])
            poly_synth.set_release(preset["release"])
            poly_synth.set_cutoff(preset["cutoff"])
            poly_synth.set_osc_type(preset["osc_type"])
            poly_synth.set_resonance(preset["resonance"])
            poly_synth.set_lfo_rate(preset["lfo_rate"])
            poly_synth.set_lfo_depth(preset["lfo_depth"])
            poly_synth.set_noise_mix(preset["noise_mix"])
            poly_synth.set_duty_cycle(preset["duty_cycle"])
            poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            self.osc_var.set(preset["osc_type"])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            self.noise_type_var.set(preset["noise_type"])
            self.preset_label.config(text=f"一時保存 {slot} を読み込みました")
        else:
            self.preset_label.config(text=f"一時保存 {slot} に保存された音色はありません")

    def create_waveform_display(self, parent):
        title = ttk.Label(parent, text="波形表示", font=("Arial", 10, "bold"))
        title.pack()
        self.wave_canvas = tk.Canvas(parent, width=450, height=150, bg="black")
        self.wave_canvas.pack(pady=5)

    def create_spectrum_display(self, parent):
        title = ttk.Label(parent, text="スペクトル表示", font=("Arial", 10, "bold"))
        title.pack()
        self.spec_canvas = tk.Canvas(parent, width=450, height=200, bg="black")
        self.spec_canvas.pack(pady=5)

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

    def update_waveform(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        xdata = np.linspace(0, 450, WAVEFORM_BUFFER_SIZE)
        ydata = 75 - (buf * 75)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.wave_canvas.delete("all")
        self.wave_canvas.create_line(points, fill="green")
        self.after(50, self.update_waveform)

    def update_spectrum(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        window = np.hamming(len(buf))
        fft_result = fft.rfft(buf * window)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-6)
        freqs = fft.rfftfreq(len(buf), 1.0 / SAMPLE_RATE)
        max_disp_freq = 5000
        idx_max = np.searchsorted(freqs, max_disp_freq)
        freqs = freqs[:idx_max]
        magnitude_db = magnitude_db[:idx_max]
        canvas_width = 450
        canvas_height = 200
        xdata = np.linspace(0, canvas_width, len(freqs))
        ydata = canvas_height - ((magnitude_db + 120) / 120 * canvas_height)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.spec_canvas.delete("all")
        self.spec_canvas.create_line(points, fill="yellow")
        for f in range(0, max_disp_freq+1, 1000):
            x = f / max_disp_freq * canvas_width
            self.spec_canvas.create_line(x, canvas_height, x, canvas_height-10, fill="white")
            self.spec_canvas.create_text(x, canvas_height-15, text=f"{f}Hz", fill="white", font=("Arial",9))
        self.after(100, self.update_spectrum)

    def create_keyboard(self):
        kb_frame = tk.Frame(self)
        kb_frame.pack(pady=10)
        self.piano_canvas = tk.Canvas(kb_frame, width=800, height=150, bg="gray")
        self.piano_canvas.pack()
        # 白鍵
        white_keys = [60, 62, 64, 65, 67, 69, 71]
        white_labels = ["C", "D", "E", "F", "G", "A", "B"]
        white_width = 100
        white_height = 150
        self.white_key_ids = {}
        for i, note in enumerate(white_keys):
            x = i * white_width
            rect = self.piano_canvas.create_rectangle(x, 0, x+white_width, white_height, fill="white", outline="black")
            self.piano_canvas.create_text(x+white_width/2, white_height-20, text=white_labels[i], font=("Arial", 12))
            self.white_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note: self.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note: self.on_note_release(n))
        # 黒鍵
        black_keys = [61, 63, 66, 68, 70]
        black_labels = ["C#", "D#", "F#", "G#", "A#"]
        black_width = 60
        black_height = 90
        black_positions = [75, 175, 375, 475, 575]
        self.black_key_ids = {}
        for pos, note, label in zip(black_positions, black_keys, black_labels):
            rect = self.piano_canvas.create_rectangle(pos, 0, pos+black_width, black_height, fill="black", outline="black")
            self.piano_canvas.create_text(pos+black_width/2, black_height-20, text=label, fill="white", font=("Arial",10))
            self.black_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note: self.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note: self.on_note_release(n))

    def save_preset(self):
        pass

    def load_preset_slot(self, slot: int):
        if slot in presets:
            preset = presets[slot]
            poly_synth.set_attack(preset["attack"])
            poly_synth.set_decay(preset["decay"])
            poly_synth.set_sustain(preset["sustain"])
            poly_synth.set_release(preset["release"])
            poly_synth.set_cutoff(preset["cutoff"])
            poly_synth.set_osc_type(preset["osc_type"])
            poly_synth.set_resonance(preset["resonance"])
            poly_synth.set_lfo_rate(preset["lfo_rate"])
            poly_synth.set_lfo_depth(preset["lfo_depth"])
            poly_synth.set_noise_mix(preset["noise_mix"])
            poly_synth.set_duty_cycle(preset["duty_cycle"])
            poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            self.osc_var.set(preset["osc_type"])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            self.noise_type_var.set(preset["noise_type"])
            self.preset_label.config(text=f"プリセット {slot} を読み込みました")
        else:
            self.preset_label.config(text=f"プリセット {slot} は存在しません")

    def save_temp(self, slot: str):
        preset = {
            "attack": poly_synth.attack,
            "decay": poly_synth.decay,
            "sustain": poly_synth.sustain,
            "release": poly_synth.release,
            "cutoff": poly_synth.cutoff,
            "osc_type": poly_synth.osc_type,
            "resonance": poly_synth.resonance,
            "lfo_rate": poly_synth.lfo_rate,
            "lfo_depth": poly_synth.lfo_depth,
            "noise_mix": poly_synth.noise_mix,
            "duty_cycle": poly_synth.duty_cycle,
            "noise_type": poly_synth.noise_type
        }
        temp_tones[slot] = preset
        self.update_temp_label()

    def load_temp(self, slot: str):
        if temp_tones[slot]:
            preset = temp_tones[slot]
            poly_synth.set_attack(preset["attack"])
            poly_synth.set_decay(preset["decay"])
            poly_synth.set_sustain(preset["sustain"])
            poly_synth.set_release(preset["release"])
            poly_synth.set_cutoff(preset["cutoff"])
            poly_synth.set_osc_type(preset["osc_type"])
            poly_synth.set_resonance(preset["resonance"])
            poly_synth.set_lfo_rate(preset["lfo_rate"])
            poly_synth.set_lfo_depth(preset["lfo_depth"])
            poly_synth.set_noise_mix(preset["noise_mix"])
            poly_synth.set_duty_cycle(preset["duty_cycle"])
            poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            self.osc_var.set(preset["osc_type"])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            self.noise_type_var.set(preset["noise_type"])
            self.preset_label.config(text=f"一時保存 {slot} を読み込みました")
        else:
            self.preset_label.config(text=f"一時保存 {slot} に保存された音色はありません")

    def create_waveform_display(self, parent):
        title = ttk.Label(parent, text="波形表示", font=("Arial", 10, "bold"))
        title.pack()
        self.wave_canvas = tk.Canvas(parent, width=450, height=150, bg="black")
        self.wave_canvas.pack(pady=5)

    def create_spectrum_display(self, parent):
        title = ttk.Label(parent, text="スペクトル表示", font=("Arial", 10, "bold"))
        title.pack()
        self.spec_canvas = tk.Canvas(parent, width=450, height=200, bg="black")
        self.spec_canvas.pack(pady=5)

    def create_keyboard(self):
        kb_frame = tk.Frame(self)
        kb_frame.pack(pady=10)
        self.piano_canvas = tk.Canvas(kb_frame, width=800, height=150, bg="gray")
        self.piano_canvas.pack()
        # 白鍵
        white_keys = [60, 62, 64, 65, 67, 69, 71]
        white_labels = ["C", "D", "E", "F", "G", "A", "B"]
        white_width = 100
        white_height = 150
        self.white_key_ids = {}
        for i, note in enumerate(white_keys):
            x = i * white_width
            rect = self.piano_canvas.create_rectangle(x, 0, x+white_width, white_height, fill="white", outline="black")
            self.piano_canvas.create_text(x+white_width/2, white_height-20, text=white_labels[i], font=("Arial", 12))
            self.white_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note: self.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note: self.on_note_release(n))
        # 黒鍵
        black_keys = [61, 63, 66, 68, 70]
        black_labels = ["C#", "D#", "F#", "G#", "A#"]
        black_width = 60
        black_height = 90
        black_positions = [75, 175, 375, 475, 575]
        self.black_key_ids = {}
        for pos, note, label in zip(black_positions, black_keys, black_labels):
            rect = self.piano_canvas.create_rectangle(pos, 0, pos+black_width, black_height, fill="black", outline="black")
            self.piano_canvas.create_text(pos+black_width/2, black_height-20, text=label, fill="white", font=("Arial",10))
            self.black_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note: self.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note: self.on_note_release(n))

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

    def update_waveform(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        xdata = np.linspace(0, 450, WAVEFORM_BUFFER_SIZE)
        ydata = 75 - (buf * 75)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.wave_canvas.delete("all")
        self.wave_canvas.create_line(points, fill="green")
        self.after(50, self.update_waveform)

    def update_spectrum(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        window = np.hamming(len(buf))
        fft_result = fft.rfft(buf * window)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-6)
        freqs = fft.rfftfreq(len(buf), 1.0 / SAMPLE_RATE)
        max_disp_freq = 5000
        idx_max = np.searchsorted(freqs, max_disp_freq)
        freqs = freqs[:idx_max]
        magnitude_db = magnitude_db[:idx_max]
        canvas_width = 450
        canvas_height = 200
        xdata = np.linspace(0, canvas_width, len(freqs))
        ydata = canvas_height - ((magnitude_db + 120) / 120 * canvas_height)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.spec_canvas.delete("all")
        self.spec_canvas.create_line(points, fill="yellow")
        for f in range(0, max_disp_freq+1, 1000):
            x = f / max_disp_freq * canvas_width
            self.spec_canvas.create_line(x, canvas_height, x, canvas_height-10, fill="white")
            self.spec_canvas.create_text(x, canvas_height-15, text=f"{f}Hz", fill="white", font=("Arial",9))
        self.after(100, self.update_spectrum)

    def create_preset_controls(self, parent):
        preset_frame = ttk.LabelFrame(parent, text="プリセット（上書き不可）")
        preset_frame.pack(fill="x", padx=5, pady=5)
        button_frame = ttk.Frame(preset_frame)
        button_frame.pack(side="top", padx=5, pady=5)
        self.preset_buttons = {}
        for i in range(1, 6):
            btn = ttk.Button(button_frame, text=f"プリセット {i}", command=lambda i=i: self.load_preset_slot(i))
            btn.grid(row=0, column=i-1, padx=3)
            self.preset_buttons[i] = btn
        self.preset_label = ttk.Label(preset_frame, text="読み込み中のプリセット: なし")
        self.preset_label.pack(side="top", padx=5, pady=5)

    def create_temp_save_controls(self, parent):
        temp_frame = ttk.LabelFrame(parent, text="一時保存（上書き可能）")
        temp_frame.pack(fill="x", padx=5, pady=5)
        btn_frame = ttk.Frame(temp_frame)
        btn_frame.pack(side="top", padx=5, pady=5)
        self.save_tempA_btn = ttk.Button(btn_frame, text="一時保存 A 保存", command=lambda: self.save_temp("A"))
        self.save_tempA_btn.grid(row=0, column=0, padx=3)
        self.load_tempA_btn = ttk.Button(btn_frame, text="一時保存 A 読み込み", command=lambda: self.load_temp("A"))
        self.load_tempA_btn.grid(row=0, column=1, padx=3)
        self.save_tempB_btn = ttk.Button(btn_frame, text="一時保存 B 保存", command=lambda: self.save_temp("B"))
        self.save_tempB_btn.grid(row=0, column=2, padx=3)
        self.load_tempB_btn = ttk.Button(btn_frame, text="一時保存 B 読み込み", command=lambda: self.load_temp("B"))
        self.load_tempB_btn.grid(row=0, column=3, padx=3)
        self.temp_label = ttk.Label(temp_frame, text="一時保存状態: A: なし, B: なし")
        self.temp_label.pack(side="top", padx=5, pady=5)

    def create_waveform_display(self, parent):
        title = ttk.Label(parent, text="波形表示", font=("Arial", 10, "bold"))
        title.pack()
        self.wave_canvas = tk.Canvas(parent, width=450, height=150, bg="black")
        self.wave_canvas.pack(pady=5)

    def create_spectrum_display(self, parent):
        title = ttk.Label(parent, text="スペクトル表示", font=("Arial", 10, "bold"))
        title.pack()
        self.spec_canvas = tk.Canvas(parent, width=450, height=200, bg="black")
        self.spec_canvas.pack(pady=5)

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

    def update_waveform(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        xdata = np.linspace(0, 450, WAVEFORM_BUFFER_SIZE)
        ydata = 75 - (buf * 75)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.wave_canvas.delete("all")
        self.wave_canvas.create_line(points, fill="green")
        self.after(50, self.update_waveform)

    def update_spectrum(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        window = np.hamming(len(buf))
        fft_result = fft.rfft(buf * window)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-6)
        freqs = fft.rfftfreq(len(buf), 1.0 / SAMPLE_RATE)
        max_disp_freq = 5000
        idx_max = np.searchsorted(freqs, max_disp_freq)
        freqs = freqs[:idx_max]
        magnitude_db = magnitude_db[:idx_max]
        canvas_width = 450
        canvas_height = 200
        xdata = np.linspace(0, canvas_width, len(freqs))
        ydata = canvas_height - ((magnitude_db + 120) / 120 * canvas_height)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.spec_canvas.delete("all")
        self.spec_canvas.create_line(points, fill="yellow")
        for f in range(0, max_disp_freq+1, 1000):
            x = f / max_disp_freq * canvas_width
            self.spec_canvas.create_line(x, canvas_height, x, canvas_height-10, fill="white")
            self.spec_canvas.create_text(x, canvas_height-15, text=f"{f}Hz", fill="white", font=("Arial",9))
        self.after(100, self.update_spectrum)

    def save_preset(self):
        # プリセットは固定なので操作しません
        pass

    def load_preset_slot(self, slot: int):
        if slot in presets:
            preset = presets[slot]
            poly_synth.set_attack(preset["attack"])
            poly_synth.set_decay(preset["decay"])
            poly_synth.set_sustain(preset["sustain"])
            poly_synth.set_release(preset["release"])
            poly_synth.set_cutoff(preset["cutoff"])
            poly_synth.set_osc_type(preset["osc_type"])
            poly_synth.set_resonance(preset["resonance"])
            poly_synth.set_lfo_rate(preset["lfo_rate"])
            poly_synth.set_lfo_depth(preset["lfo_depth"])
            poly_synth.set_noise_mix(preset["noise_mix"])
            poly_synth.set_duty_cycle(preset["duty_cycle"])
            poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            self.osc_var.set(preset["osc_type"])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            self.noise_type_var.set(preset["noise_type"])
            self.preset_label.config(text=f"プリセット {slot} を読み込みました")
        else:
            self.preset_label.config(text=f"プリセット {slot} は存在しません")

    def save_temp(self, slot: str):
        preset = {
            "attack": poly_synth.attack,
            "decay": poly_synth.decay,
            "sustain": poly_synth.sustain,
            "release": poly_synth.release,
            "cutoff": poly_synth.cutoff,
            "osc_type": poly_synth.osc_type,
            "resonance": poly_synth.resonance,
            "lfo_rate": poly_synth.lfo_rate,
            "lfo_depth": poly_synth.lfo_depth,
            "noise_mix": poly_synth.noise_mix,
            "duty_cycle": poly_synth.duty_cycle,
            "noise_type": poly_synth.noise_type
        }
        temp_tones[slot] = preset
        self.update_temp_label()

    def load_temp(self, slot: str):
        if temp_tones[slot]:
            preset = temp_tones[slot]
            poly_synth.set_attack(preset["attack"])
            poly_synth.set_decay(preset["decay"])
            poly_synth.set_sustain(preset["sustain"])
            poly_synth.set_release(preset["release"])
            poly_synth.set_cutoff(preset["cutoff"])
            poly_synth.set_osc_type(preset["osc_type"])
            poly_synth.set_resonance(preset["resonance"])
            poly_synth.set_lfo_rate(preset["lfo_rate"])
            poly_synth.set_lfo_depth(preset["lfo_depth"])
            poly_synth.set_noise_mix(preset["noise_mix"])
            poly_synth.set_duty_cycle(preset["duty_cycle"])
            poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            self.osc_var.set(preset["osc_type"])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            self.noise_type_var.set(preset["noise_type"])
            self.preset_label.config(text=f"一時保存 {slot} を読み込みました")
        else:
            self.preset_label.config(text=f"一時保存 {slot} に保存された音色はありません")

# ---------------------------
# MIDI入力用スレッド
# ---------------------------
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

# ---------------------------
# オーディオコールバック（PolySynth版）
# ---------------------------
def audio_callback(outdata: np.ndarray, frames: int, time_info, status) -> None:
    if status:
        print("Audio status:", status)
    outdata.fill(0.0)
    for i in range(frames):
        sample = poly_synth.process()
        outdata[i, 0] = sample
        global buffer_index
        with buffer_lock:
            waveform_buffer[buffer_index] = sample
            buffer_index = (buffer_index + 1) % WAVEFORM_BUFFER_SIZE

# ---------------------------
# メイン関数
# ---------------------------
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
