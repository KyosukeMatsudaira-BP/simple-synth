# gui.py
import numpy as np
import numpy.fft as fft
import tkinter as tk
from tkinter import ttk
import threading
from synth_core import PolySynth, SAMPLE_RATE
from audio_engine import waveform_buffer, buffer_lock, WAVEFORM_BUFFER_SIZE

# 既定のプリセット（上書き不可）
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
presets = default_presets.copy()

class SynthGUI(tk.Tk):
    def __init__(self, poly_synth: PolySynth):
        super().__init__()
        self.poly_synth = poly_synth
        self.title("ポリフォニックシンセ")
        self.geometry("900x900")
        # 文字サイズ
        self.font_size = 16
        # スタイル設定（ttkウィジェット用）
        self.style = ttk.Style(self)
        self.style.configure("My.TLabelframe", font=("Arial", self.font_size))
        self.style.configure("My.TLabel", font=("Arial", self.font_size))
        self.style.configure("My.TButton", font=("Arial", self.font_size))
        # LabelFrame のタイトル用スタイルを設定
        self.style.configure("My.TLabelframe.Label", font=("Arial", self.font_size, "bold"))
        # キャンバスサイズ（調整可能）
        self.waveform_width = 600
        self.waveform_height = 300
        self.spectrum_width = 600
        self.spectrum_height = 300
        self.create_main_layout()
        self.bind("<KeyPress>", self.on_key_press)
        self.bind("<KeyRelease>", self.on_key_release)
        self.pc_key_map = {
            'z': 60, 's': 61, 'x': 62, 'd': 63, 'c': 64,
            'v': 65, 'g': 66, 'b': 67, 'h': 68, 'n': 69, 'j': 70, 'm': 71
        }
        self.active_keys = set()
        self.after(50, self.update_waveform)
        self.after(100, self.update_spectrum)

    def create_main_layout(self):
        # 左側：コントロール＆プリセット、右側：波形表示、スペクトル表示、ピアノ鍵盤
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        self.create_controls(left_frame)
        self.create_preset_controls(left_frame)
        self.create_reset_controls(left_frame)
        top_right = ttk.Frame(right_frame)
        top_right.pack(fill="both", expand=True, padx=5, pady=5)
        self.create_waveform_display(top_right)
        self.create_spectrum_display(top_right)
        self.create_keyboard(right_frame)

    def create_reset_controls(self, parent):
        reset_frame = ttk.LabelFrame(parent, text="リセット", style="My.TLabelframe")
        reset_frame.pack(fill="x", padx=5, pady=5)
        reset_btn = ttk.Button(reset_frame, text="リセット (サイン波に戻す)", command=self.reset_to_sine, style="My.TButton")
        reset_btn.pack(padx=5, pady=5)

    def reset_to_sine(self):
        self.poly_synth.set_osc_type("sine")
        self.poly_synth.set_noise_mix(0.0)
        self.poly_synth.set_lfo_depth(0.0)
        self.poly_synth.set_resonance(0.0)
        self.poly_synth.set_attack(0.1)
        self.poly_synth.set_decay(0.2)
        self.poly_synth.set_sustain(1.0)
        self.poly_synth.set_release(0.5)
        self.poly_synth.set_cutoff(5000)
        # UI側の変数もリセット
        self.osc_var.set("サイン波")
        self.noise_mix_var.set(0.0)
        self.lfo_depth_var.set(0.0)
        self.resonance_var.set(0.0)
        self.attack_var.set(0.01)
        self.decay_var.set(0.2)
        self.sustain_var.set(0.5)
        self.release_var.set(0.5)
        self.cutoff_var.set(1000)
        self.lfo_rate_var.set(5.0)
        self.duty_cycle_var.set(0.5)

    def load_preset_slot(self, slot: int):
        if slot in presets:
            preset = presets[slot]
            self.poly_synth.set_attack(preset["attack"])
            self.poly_synth.set_decay(preset["decay"])
            self.poly_synth.set_sustain(preset["sustain"])
            self.poly_synth.set_release(preset["release"])
            self.poly_synth.set_cutoff(preset["cutoff"])
            self.poly_synth.set_osc_type(preset["osc_type"])
            self.poly_synth.set_resonance(preset["resonance"])
            self.poly_synth.set_lfo_rate(preset["lfo_rate"])
            self.poly_synth.set_lfo_depth(preset["lfo_depth"])
            self.poly_synth.set_noise_mix(preset["noise_mix"])
            self.poly_synth.set_duty_cycle(preset["duty_cycle"])
            self.poly_synth.set_noise_type(preset["noise_type"])
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            inv_osc_map = {v: k for k, v in {"サイン波": "sine", "三角波": "triangle", "矩形波": "square", "ノコギリ波": "sawtooth"}.items()}
            self.osc_var.set(inv_osc_map[preset["osc_type"]])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            inv_noise_map = {v: k for k, v in {"ホワイトノイズ": "white", "ピンクノイズ": "pink"}.items()}
            self.noise_type_var.set(inv_noise_map[preset["noise_type"]])
            self.preset_label.config(text=f"プリセット {slot} を読み込みました")
        else:
            self.preset_label.config(text=f"プリセット {slot} は存在しません")

    def create_controls(self, parent):
        container = ttk.LabelFrame(parent, text="コントロール", style="My.TLabelframe")
        container.pack(fill="both", expand=True, padx=5, pady=5)

        # オシレーター
        osc_frame = ttk.LabelFrame(container, text="オシレーター", style="My.TLabelframe")
        osc_frame.pack(fill="both", expand=True, padx=5, pady=5)
        ttk.Label(osc_frame, text="波形種別", style="My.TLabel").grid(row=0, column=0, sticky="w")
        self.osc_types = ["サイン波", "三角波", "矩形波", "ノコギリ波"]
        self.osc_type_map = {"サイン波": "sine", "三角波": "triangle", "矩形波": "square", "ノコギリ波": "sawtooth"}
        self.osc_var = tk.StringVar(value="サイン波")
        osc_menu = tk.OptionMenu(osc_frame, self.osc_var, *self.osc_types,
                                 command=lambda val: self.poly_synth.set_osc_type(self.osc_type_map[val]))
        osc_menu.config(width=12, font=("Arial", self.font_size))
        osc_menu.grid(row=0, column=1)
        ttk.Label(osc_frame, text="ノイズミックス", style="My.TLabel").grid(row=1, column=0, sticky="w")
        self.noise_mix_var = tk.DoubleVar(value=self.poly_synth.noise_mix)
        noise_mix_slider = tk.Scale(osc_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                    length=350, variable=self.noise_mix_var,
                                    font=("Arial", self.font_size),
                                    command=lambda val: self.poly_synth.set_noise_mix(float(val)))
        noise_mix_slider.grid(row=1, column=1)
        ttk.Label(osc_frame, text="デューティー比", style="My.TLabel").grid(row=2, column=0, sticky="w")
        self.duty_cycle_var = tk.DoubleVar(value=self.poly_synth.duty_cycle)
        duty_cycle_slider = tk.Scale(osc_frame, from_=0.1, to=0.9, resolution=0.01, orient=tk.HORIZONTAL,
                                     length=350, variable=self.duty_cycle_var,
                                     font=("Arial", self.font_size),
                                     command=lambda val: self.poly_synth.set_duty_cycle(float(val)))
        duty_cycle_slider.grid(row=2, column=1)
        ttk.Label(osc_frame, text="ノイズタイプ", style="My.TLabel").grid(row=3, column=0, sticky="w")
        self.noise_types = ["ホワイトノイズ", "ピンクノイズ"]
        self.noise_type_map = {"ホワイトノイズ": "white", "ピンクノイズ": "pink"}
        self.noise_type_var = tk.StringVar(value="ホワイトノイズ")
        noise_menu = tk.OptionMenu(osc_frame, self.noise_type_var, *self.noise_types,
                                   command=lambda val: self.poly_synth.set_noise_type(self.noise_type_map[val]))
        noise_menu.config(width=12, font=("Arial", self.font_size))
        noise_menu.grid(row=3, column=1)

        # フィルター
        filter_frame = ttk.LabelFrame(container, text="フィルター", style="My.TLabelframe")
        filter_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.cutoff_var = tk.DoubleVar(value=self.poly_synth.cutoff)
        ttk.Label(filter_frame, text="カットオフ（Hz）", style="My.TLabel").grid(row=0, column=0, sticky="w")
        cutoff_slider = tk.Scale(filter_frame, from_=20.0, to=5000.0, resolution=1, orient=tk.HORIZONTAL,
                                 length=350, variable=self.cutoff_var,
                                 font=("Arial", self.font_size),
                                 command=lambda val: self.poly_synth.set_cutoff(float(val)))
        cutoff_slider.grid(row=0, column=1)
        ttk.Label(filter_frame, text="レゾナンス", style="My.TLabel").grid(row=1, column=0, sticky="w")
        self.resonance_var = tk.DoubleVar(value=self.poly_synth.resonance)
        resonance_slider = tk.Scale(filter_frame, from_=0.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
                                    length=350, variable=self.resonance_var,
                                    font=("Arial", self.font_size),
                                    command=lambda val: self.poly_synth.set_resonance(float(val)))
        resonance_slider.grid(row=1, column=1)

        # ADSR（アンプエンベロープ）
        adsr_frame = ttk.LabelFrame(container, text="ADSR（アンプエンベロープ）", style="My.TLabelframe")
        adsr_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.attack_var = tk.DoubleVar(value=self.poly_synth.attack)
        ttk.Label(adsr_frame, text="アタック", style="My.TLabel").grid(row=0, column=0, sticky="w")
        attack_slider = tk.Scale(adsr_frame, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL,
                                 length=350, variable=self.attack_var,
                                 font=("Arial", self.font_size),
                                 command=lambda val: self.poly_synth.set_attack(float(val)))
        attack_slider.grid(row=0, column=1)
        self.decay_var = tk.DoubleVar(value=self.poly_synth.decay)
        ttk.Label(adsr_frame, text="ディケイ", style="My.TLabel").grid(row=1, column=0, sticky="w")
        decay_slider = tk.Scale(adsr_frame, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL,
                                length=350, variable=self.decay_var,
                                font=("Arial", self.font_size),
                                command=lambda val: self.poly_synth.set_decay(float(val)))
        decay_slider.grid(row=1, column=1)
        self.sustain_var = tk.DoubleVar(value=self.poly_synth.sustain)
        ttk.Label(adsr_frame, text="サステイン", style="My.TLabel").grid(row=2, column=0, sticky="w")
        sustain_slider = tk.Scale(adsr_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                  length=350, variable=self.sustain_var,
                                  font=("Arial", self.font_size),
                                  command=lambda val: self.poly_synth.set_sustain(float(val)))
        sustain_slider.grid(row=2, column=1)
        self.release_var = tk.DoubleVar(value=self.poly_synth.release)
        ttk.Label(adsr_frame, text="リリース", style="My.TLabel").grid(row=3, column=0, sticky="w")
        release_slider = tk.Scale(adsr_frame, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL,
                                  length=350, variable=self.release_var,
                                  font=("Arial", self.font_size),
                                  command=lambda val: self.poly_synth.set_release(float(val)))
        release_slider.grid(row=3, column=1)
        
        # モジュレータ（ビブラート）
        lfo_frame = ttk.LabelFrame(container, text="モジュレータ（ビブラート）", style="My.TLabelframe")
        lfo_frame.pack(fill="both", expand=True, padx=5, pady=5)
        ttk.Label(lfo_frame, text="レート（Hz）", style="My.TLabel").grid(row=0, column=0, sticky="w")
        self.lfo_rate_var = tk.DoubleVar(value=self.poly_synth.lfo_rate)
        lfo_rate_slider = tk.Scale(lfo_frame, from_=0.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
                                   length=350, variable=self.lfo_rate_var,
                                   font=("Arial", self.font_size),
                                   command=lambda val: self.poly_synth.set_lfo_rate(float(val)))
        lfo_rate_slider.grid(row=0, column=1)
        ttk.Label(lfo_frame, text="デプス", style="My.TLabel").grid(row=1, column=0, sticky="w")
        self.lfo_depth_var = tk.DoubleVar(value=self.poly_synth.lfo_depth)
        lfo_depth_slider = tk.Scale(lfo_frame, from_=0.0, to=0.05, resolution=0.001, orient=tk.HORIZONTAL,
                                    length=350, variable=self.lfo_depth_var,
                                    font=("Arial", self.font_size),
                                    command=lambda val: self.poly_synth.set_lfo_depth(float(val)))
        lfo_depth_slider.grid(row=1, column=1)

    def create_preset_controls(self, parent):
        preset_frame = ttk.LabelFrame(parent, text="プリセット（上書き不可）", style="My.TLabelframe")
        preset_frame.pack(fill="both", expand=True, padx=5, pady=5)
        button_frame = ttk.Frame(preset_frame)
        button_frame.pack(side="top", padx=5, pady=5)
        self.preset_buttons = {}
        for i in range(1, 6):
            btn = ttk.Button(button_frame, text=f"プリセット {i}",
                             command=lambda slot=i: self.load_preset_slot(slot), style="My.TButton")
            btn.grid(row=0, column=i-1, padx=3)
            self.preset_buttons[i] = btn
        self.preset_label = ttk.Label(preset_frame, text="読み込み中のプリセット: なし", style="My.TLabel")
        self.preset_label.pack(side="top", padx=5, pady=5)

    def create_waveform_display(self, parent):
        title = ttk.Label(parent, text="波形", style="My.TLabel")
        title.config(font=("Arial", self.font_size, "bold"))
        title.pack()
        self.wave_canvas = tk.Canvas(parent, width=self.waveform_width, height=self.waveform_height, bg="black")
        self.wave_canvas.pack(pady=5)

    def create_spectrum_display(self, parent):
        title = ttk.Label(parent, text="スペクトル", style="My.TLabel")
        title.config(font=("Arial", self.font_size, "bold"))
        title.pack()
        self.spec_canvas = tk.Canvas(parent, width=self.spectrum_width, height=self.spectrum_height, bg="black")
        self.spec_canvas.pack(pady=5)

    def create_keyboard(self, parent):
        kb_frame = ttk.Frame(parent)
        kb_frame.pack(pady=5)
        self.piano_canvas = tk.Canvas(kb_frame, width=600, height=120, bg="gray")
        self.piano_canvas.pack()
        # 白鍵：C～C（8鍵）として配置
        white_keys = [60, 62, 64, 65, 67, 69, 71, 72]
        white_labels = ["z", "x", "c", "v", "b", "n", "m", ","]  # 対応するラベル
        white_width = 80
        white_height = 120
        self.white_key_ids = {}
        for i, note in enumerate(white_keys):
            x = i * white_width
            rect = self.piano_canvas.create_rectangle(x, 0, x+white_width, white_height, fill="white", outline="black")
            self.piano_canvas.create_text(x+white_width/2, white_height-15, text=white_labels[i], font=("Arial", self.font_size), fill="black")
            self.white_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note, s=self: s.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note, s=self: s.on_note_release(n))
        # 黒鍵：C#、D#、F#、G#、A#
        black_keys = [61, 63, 66, 68, 70]
        black_labels = ["s", "d", "g", "h", "j"]
        black_width = 50
        black_height = 70
        # 黒鍵は白鍵間に配置
        black_positions = [int(0.7*white_width), int(1.7*white_width), int(3.7*white_width), int(4.7*white_width), int(5.7*white_width)]
        self.black_key_ids = {}
        for pos, note, label in zip(black_positions, black_keys, black_labels):
            rect = self.piano_canvas.create_rectangle(pos, 0, pos+black_width, black_height, fill="black", outline="black")
            self.piano_canvas.create_text(pos+black_width/2, black_height-15, text=label, font=("Arial", self.font_size), fill="gold")
            self.black_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note, s=self: s.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note, s=self: s.on_note_release(n))

    def on_note_press(self, note):
        self.poly_synth.note_on(note)

    def on_note_release(self, note):
        self.poly_synth.note_off(note)

    def on_key_press(self, event):
        key = event.char
        if key in self.pc_key_map and key not in self.active_keys:
            self.active_keys.add(key)
            note = self.pc_key_map[key]
            self.poly_synth.note_on(note)

    def on_key_release(self, event):
        key = event.char
        if key in self.pc_key_map and key in self.active_keys:
            self.active_keys.remove(key)
            note = self.pc_key_map[key]
            self.poly_synth.note_off(note)

    def update_waveform(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        xdata = np.linspace(0, self.waveform_width, WAVEFORM_BUFFER_SIZE)
        ydata = self.waveform_height/2 - (buf * self.waveform_height/2)
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
        raw_db = 20 * np.log10(magnitude + 1e-6)
        max_db = np.max(raw_db)
        margin = 5.0
        top = max_db + margin
        bottom = top - 120
        canvas_width = self.spectrum_width
        canvas_height = self.spectrum_height
        margin_x = 25  # 全体の左右余白
        xdata = np.linspace(margin_x, canvas_width - margin_x, len(fft_result))
        ydata = canvas_height - ((raw_db - bottom) / (top - bottom) * canvas_height)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.spec_canvas.delete("all")
        self.spec_canvas.create_line(points, fill="yellow")
        freqs = fft.rfftfreq(len(buf), 1.0 / SAMPLE_RATE)
        max_disp_freq = 10000
        for f in range(0, max_disp_freq+1, 2000):
            x = margin_x + (f / max_disp_freq) * (canvas_width - 2 * margin_x)
            self.spec_canvas.create_line(x, canvas_height, x, canvas_height-10, fill="white")
            self.spec_canvas.create_text(x, canvas_height-15, text=f"{f}Hz", fill="white", font=("Arial", self.font_size), anchor="n")
        self.after(100, self.update_spectrum)

if __name__ == "__main__":
    from synth_core import PolySynth
    poly_synth = PolySynth(max_voices=6, sample_rate=SAMPLE_RATE)
    app = SynthGUI(poly_synth)
    app.mainloop()
