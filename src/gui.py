# gui.py
import numpy as np
import numpy.fft as fft
import tkinter as tk
from tkinter import ttk
import threading
from synth_core import PolySynth, SAMPLE_RATE, SynthMode
from audio_engine import waveform_buffer, buffer_lock, WAVEFORM_BUFFER_SIZE
from presets import default_presets

# プリセットデータをインポート
presets = default_presets.copy()

class SynthGUI(tk.Tk):
    def __init__(self, poly_synth: PolySynth):
        super().__init__()
        self.poly_synth = poly_synth
        self.title("ポリフォニックシンセ")
        
        # MacBook Air画面サイズギリギリまで使用
        self.geometry("1360x750")
        self.resizable(True, True)  # 全画面表示可能
        
        # 文字サイズを読みやすく
        self.font_size = 12
        
        # スタイル設定
        self.style = ttk.Style(self)
        self.style.configure("My.TLabelframe", font=("Arial", self.font_size))
        self.style.configure("My.TLabel", font=("Arial", self.font_size))
        self.style.configure("My.TButton", font=("Arial", self.font_size))
        
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
    
    def toggle_reverb(self):
        """リバーブのオンオフを切り替え"""
        self.poly_synth.set_reverb_enabled(self.reverb_enabled.get())
    
    def detune_decrease(self):
        """デチューン値を減少"""
        try:
            step = self.detune_step_var.get()
            current = self.osc2_detune_var.get()
            new_value = max(-1200, current - step)  # 下限チェック
            self.osc2_detune_var.set(new_value)
            self.poly_synth.set_osc2_detune(new_value)
        except tk.TclError:
            # 無効な値の場合はデフォルトステップを使用
            current = self.osc2_detune_var.get()
            new_value = max(-1200, current - 1.0)
            self.osc2_detune_var.set(new_value)
            self.poly_synth.set_osc2_detune(new_value)
    
    def detune_increase(self):
        """デチューン値を増加"""
        try:
            step = self.detune_step_var.get()
            current = self.osc2_detune_var.get()
            new_value = min(1200, current + step)  # 上限チェック
            self.osc2_detune_var.set(new_value)
            self.poly_synth.set_osc2_detune(new_value)
        except tk.TclError:
            # 無効な値の場合はデフォルトステップを使用
            current = self.osc2_detune_var.get()
            new_value = min(1200, current + 1.0)
            self.osc2_detune_var.set(new_value)
            self.poly_synth.set_osc2_detune(new_value)

    def toggle_ring_modulation(self):
        """リング変調のオンオフを切り替え"""
        if self.ring_mod_enabled.get():
            # リング変調をオン
            self.poly_synth.ring_mod_enabled = True
            self.poly_synth.set_ring_mod_enabled(True)
            # シンクを無効化
            self.sync_enabled.set(False)
            self.poly_synth.sync_enabled = False
            self.poly_synth.set_sync_enabled(False)
        else:
            # リング変調をオフ
            self.poly_synth.ring_mod_enabled = False
            self.poly_synth.set_ring_mod_enabled(False)

    def toggle_oscillator_sync(self):
        """オシレーターシンクのオンオフを切り替え"""
        if self.sync_enabled.get():
            # シンクをオン
            self.poly_synth.sync_enabled = True
            self.poly_synth.set_sync_enabled(True)
            # リング変調を無効化
            self.ring_mod_enabled.set(False)
            self.poly_synth.ring_mod_enabled = False
            self.poly_synth.set_ring_mod_enabled(False)
        else:
            # シンクをオフ
            self.poly_synth.sync_enabled = False
            self.poly_synth.set_sync_enabled(False)

    def create_main_layout(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 上部：コントロールパネル（4列レイアウト）
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=(0, 5))
        
        # 列1: OSC1
        col1 = ttk.Frame(control_frame)
        col1.pack(side="left", fill="both", expand=True, padx=3)
        self.create_osc1_controls(col1)
        
        # 列2: フィルター & ADSR
        col2 = ttk.Frame(control_frame)
        col2.pack(side="left", fill="both", expand=True, padx=3)
        self.create_filter_controls(col2)
        self.create_adsr_controls(col2)
        
        # 列3: LFO & ノイズ
        col3 = ttk.Frame(control_frame)
        col3.pack(side="left", fill="both", expand=True, padx=3)
        self.create_lfo_controls(col3)
        self.create_noise_controls(col3)
        
        # 列4: OSC2
        col4 = ttk.Frame(control_frame)
        col4.pack(side="left", fill="both", expand=True, padx=3)
        self.create_osc2_controls(col4)
        
        # 中部：表示エリア
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill="x", pady=5)
        
        # 左：波形とスペクトル
        left_display = ttk.Frame(display_frame)
        left_display.pack(side="left", padx=(0, 10))
        self.create_displays(left_display)
        
        # 右：プリセット
        right_display = ttk.Frame(display_frame)
        right_display.pack(side="right", fill="x", expand=True)
        self.create_preset_controls(right_display)
        
        # 下部：ピアノ鍵盤
        self.create_keyboard(main_frame)

    def create_osc1_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="オシレーター1", style="My.TLabelframe")
        frame.pack(fill="x", pady=2)
        
        # 波形選択
        ttk.Label(frame, text="波形", style="My.TLabel").grid(row=0, column=0, sticky="w", padx=2, pady=1)
        self.osc_types = ["サイン波", "三角波", "矩形波", "ノコギリ波"]
        self.osc_type_map = {"サイン波": "sine", "三角波": "triangle", "矩形波": "square", "ノコギリ波": "sawtooth"}
        self.osc_var = tk.StringVar(value="サイン波")
        osc_combo = ttk.Combobox(frame, textvariable=self.osc_var, values=self.osc_types, 
                                width=12, font=("Arial", self.font_size), state="readonly")
        osc_combo.grid(row=0, column=1, padx=2, pady=1)
        osc_combo.bind("<<ComboboxSelected>>", lambda e: self.poly_synth.set_osc_type(self.osc_type_map[self.osc_var.get()]))
        

        
        # デューティーサイクル
        ttk.Label(frame, text="デューティー", style="My.TLabel").grid(row=2, column=0, sticky="w", padx=2, pady=1)
        self.duty_cycle_var = tk.DoubleVar(value=0.5)
        duty_slider = tk.Scale(frame, from_=0.1, to=0.9, resolution=0.01, orient=tk.HORIZONTAL,
                              length=180, variable=self.duty_cycle_var, font=("Arial", self.font_size),
                              command=lambda val: self.poly_synth.set_duty_cycle(float(val)))
        duty_slider.grid(row=2, column=1, padx=2, pady=1)

    def create_filter_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="フィルター", style="My.TLabelframe")
        frame.pack(fill="x", pady=2)
        
        # カットオフ
        ttk.Label(frame, text="カットオフ（Hz）", style="My.TLabel").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.cutoff_var = tk.DoubleVar(value=1000)
        cutoff_slider = tk.Scale(frame, from_=20, to=5000, resolution=1, orient=tk.HORIZONTAL,
                                length=180, variable=self.cutoff_var, font=("Arial", self.font_size),
                                command=lambda val: self.poly_synth.set_cutoff(float(val)))
        cutoff_slider.grid(row=0, column=1, padx=2, pady=2)
        
        # レゾナンス
        ttk.Label(frame, text="レゾナンス", style="My.TLabel").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self.resonance_var = tk.DoubleVar(value=0.0)
        res_slider = tk.Scale(frame, from_=0.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
                             length=180, variable=self.resonance_var, font=("Arial", self.font_size),
                             command=lambda val: self.poly_synth.set_resonance(float(val)))
        res_slider.grid(row=1, column=1, padx=2, pady=2)

    def create_adsr_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="ADSR", style="My.TLabelframe")
        frame.pack(fill="x", pady=2)
        
        # アタック
        ttk.Label(frame, text="アタック", style="My.TLabel").grid(row=0, column=0, sticky="w", padx=2, pady=1)
        self.attack_var = tk.DoubleVar(value=0.1)
        attack_slider = tk.Scale(frame, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL,
                                length=180, variable=self.attack_var, font=("Arial", self.font_size),
                                command=lambda val: self.poly_synth.set_attack(float(val)))
        attack_slider.grid(row=0, column=1, padx=2, pady=1)
        
        # ディケイ
        ttk.Label(frame, text="ディケイ", style="My.TLabel").grid(row=1, column=0, sticky="w", padx=2, pady=1)
        self.decay_var = tk.DoubleVar(value=0.2)
        decay_slider = tk.Scale(frame, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL,
                               length=180, variable=self.decay_var, font=("Arial", self.font_size),
                               command=lambda val: self.poly_synth.set_decay(float(val)))
        decay_slider.grid(row=1, column=1, padx=2, pady=1)
        
        # サステイン
        ttk.Label(frame, text="サステイン", style="My.TLabel").grid(row=2, column=0, sticky="w", padx=2, pady=1)
        self.sustain_var = tk.DoubleVar(value=0.7)
        sustain_slider = tk.Scale(frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                 length=180, variable=self.sustain_var, font=("Arial", self.font_size),
                                 command=lambda val: self.poly_synth.set_sustain(float(val)))
        sustain_slider.grid(row=2, column=1, padx=2, pady=1)
        
        # リリース
        ttk.Label(frame, text="リリース", style="My.TLabel").grid(row=3, column=0, sticky="w", padx=2, pady=1)
        self.release_var = tk.DoubleVar(value=0.5)
        release_slider = tk.Scale(frame, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL,
                                 length=180, variable=self.release_var, font=("Arial", self.font_size),
                                 command=lambda val: self.poly_synth.set_release(float(val)))
        release_slider.grid(row=3, column=1, padx=2, pady=1)

    def create_lfo_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="LFO", style="My.TLabelframe")
        frame.pack(fill="x", pady=2)
        
        # レート
        ttk.Label(frame, text="レート（Hz）", style="My.TLabel").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.lfo_rate_var = tk.DoubleVar(value=5.0)
        rate_slider = tk.Scale(frame, from_=0.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
                              length=180, variable=self.lfo_rate_var, font=("Arial", self.font_size),
                              command=lambda val: self.poly_synth.set_lfo_rate(float(val)))
        rate_slider.grid(row=0, column=1, padx=2, pady=2)
        
        # デプス
        ttk.Label(frame, text="デプス", style="My.TLabel").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self.lfo_depth_var = tk.DoubleVar(value=0.0)
        depth_slider = tk.Scale(frame, from_=0.0, to=0.05, resolution=0.001, orient=tk.HORIZONTAL,
                               length=180, variable=self.lfo_depth_var, font=("Arial", self.font_size),
                               command=lambda val: self.poly_synth.set_lfo_depth(float(val)))
        depth_slider.grid(row=1, column=1, padx=2, pady=2)

    def create_noise_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="ノイズ", style="My.TLabelframe")
        frame.pack(fill="x", pady=2)
        
        # ノイズミックス
        ttk.Label(frame, text="ミックス", style="My.TLabel").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.noise_mix_var = tk.DoubleVar(value=0.0)
        noise_slider = tk.Scale(frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                               length=180, variable=self.noise_mix_var, font=("Arial", self.font_size),
                               command=lambda val: self.poly_synth.set_noise_mix(float(val)))
        noise_slider.grid(row=0, column=1, padx=2, pady=2)
        
        # ノイズタイプ
        ttk.Label(frame, text="タイプ", style="My.TLabel").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self.noise_types = ["ホワイト", "ピンク"]
        self.noise_type_map = {"ホワイト": "white", "ピンク": "pink"}
        self.noise_type_var = tk.StringVar(value="ホワイト")
        noise_combo = ttk.Combobox(frame, textvariable=self.noise_type_var, values=self.noise_types,
                                  width=12, font=("Arial", self.font_size), state="readonly")
        noise_combo.grid(row=1, column=1, padx=2, pady=2)
        noise_combo.bind("<<ComboboxSelected>>", lambda e: self.poly_synth.set_noise_type(self.noise_type_map[self.noise_type_var.get()]))

    def create_osc2_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="OSC2（セカンドオシレーター）", style="My.TLabelframe")
        frame.pack(fill="x", pady=2)
        
        # 波形
        ttk.Label(frame, text="波形", style="My.TLabel").grid(row=0, column=0, sticky="w", padx=2, pady=1)
        self.osc2_var = tk.StringVar(value="サイン波")
        osc2_combo = ttk.Combobox(frame, textvariable=self.osc2_var, values=self.osc_types,
                                 width=12, font=("Arial", self.font_size), state="readonly")
        osc2_combo.grid(row=0, column=1, padx=2, pady=1)
        osc2_combo.bind("<<ComboboxSelected>>", lambda e: self.poly_synth.set_osc2_type(self.osc_type_map[self.osc2_var.get()]))
        

        
        # デチューン（範囲を大幅に拡張）
        ttk.Label(frame, text="デチューン（セント）", style="My.TLabel").grid(row=2, column=0, sticky="w", padx=2, pady=1)
        
        # デチューンコントロールフレーム
        detune_control_frame = ttk.Frame(frame)
        detune_control_frame.grid(row=2, column=1, padx=2, pady=1, sticky="w")
        
        # デチューンスライダー
        self.osc2_detune_var = tk.DoubleVar(value=0.0)
        detune_slider = tk.Scale(detune_control_frame, from_=-1200, to=1200, resolution=1, orient=tk.HORIZONTAL,
                                length=180, variable=self.osc2_detune_var, font=("Arial", self.font_size),
                                command=lambda val: self.poly_synth.set_osc2_detune(float(val)))
        detune_slider.grid(row=0, column=0, columnspan=4)
        
        # 微調整ステップ入力
        ttk.Label(detune_control_frame, text="ステップ:", font=("Arial", self.font_size-2)).grid(row=1, column=0, sticky="w")
        self.detune_step_var = tk.DoubleVar(value=1.0)
        step_entry = tk.Entry(detune_control_frame, textvariable=self.detune_step_var, width=4, font=("Arial", self.font_size-1))
        step_entry.grid(row=1, column=1, padx=2)
        
        # 微調整ボタン
        minus_btn = tk.Button(detune_control_frame, text="-", width=2, font=("Arial", self.font_size-1),
                             command=self.detune_decrease)
        minus_btn.grid(row=1, column=2, padx=1)
        
        plus_btn = tk.Button(detune_control_frame, text="+", width=2, font=("Arial", self.font_size-1),
                            command=self.detune_increase)
        plus_btn.grid(row=1, column=3, padx=1)
        
        # ミックス
        ttk.Label(frame, text="ミックス（0=OSC1, 1=OSC2）", style="My.TLabel").grid(row=3, column=0, sticky="w", padx=2, pady=1)
        self.osc_mix_var = tk.DoubleVar(value=0.5)
        mix_slider = tk.Scale(frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                             length=180, variable=self.osc_mix_var, font=("Arial", self.font_size),
                             command=lambda val: self.poly_synth.set_osc_mix(float(val)))
        mix_slider.grid(row=3, column=1, padx=2, pady=1)
        


    def create_displays(self, parent):
        # 波形表示（見やすいサイズ）
        wave_frame = ttk.LabelFrame(parent, text="波形表示", style="My.TLabelframe")
        wave_frame.pack(pady=3)
        self.wave_canvas = tk.Canvas(wave_frame, width=400, height=120, bg="black")
        self.wave_canvas.pack(padx=5, pady=5)
        
        # スペクトル表示（見やすいサイズ）
        spec_frame = ttk.LabelFrame(parent, text="周波数スペクトル", style="My.TLabelframe")
        spec_frame.pack(pady=3)
        self.spec_canvas = tk.Canvas(spec_frame, width=400, height=120, bg="black")
        self.spec_canvas.pack(padx=5, pady=5)

    def create_preset_controls(self, parent):
        # 合成モード選択（コンパクト化）
        mode_frame = ttk.LabelFrame(parent, text="合成モード", style="My.TLabelframe")
        mode_frame.pack(fill="x", pady=2)
        
        # 簡潔な説明
        ttk.Label(mode_frame, text="優先順位: FM > シンク > リング変調 > 通常ミックス", 
                 style="My.TLabel", font=("Arial", 9)).pack(padx=3, pady=1)
        
        # モード設定ボタン
        mode_buttons_frame = ttk.Frame(mode_frame)
        mode_buttons_frame.pack(padx=3, pady=3)
        
        # リング変調オンオフボタン
        self.ring_mod_enabled = tk.BooleanVar(value=False)
        self.ring_check = tk.Checkbutton(mode_buttons_frame, text="リング変調", 
                                        variable=self.ring_mod_enabled,
                                        command=self.toggle_ring_modulation,
                                        font=("Arial", self.font_size, "bold"),
                                        fg="red", selectcolor="lightcoral")
        self.ring_check.grid(row=0, column=0, padx=10, pady=2)
        
        # シンクオンオフボタン
        self.sync_enabled = tk.BooleanVar(value=False)
        self.sync_check = tk.Checkbutton(mode_buttons_frame, text="オシレーターシンク", 
                                        variable=self.sync_enabled,
                                        command=self.toggle_oscillator_sync,
                                        font=("Arial", self.font_size, "bold"),
                                        fg="blue", selectcolor="lightblue")
        self.sync_check.grid(row=0, column=1, padx=10, pady=2)
        
        # FM合成パラメーター
        fm_frame = ttk.LabelFrame(parent, text="FM合成", style="My.TLabelframe")
        fm_frame.pack(fill="x", pady=3)
        
        # モジュレーター比（キャリアを1とした時の比率）
        ttk.Label(fm_frame, text="モジュレーター比", style="My.TLabel").grid(row=0, column=0, sticky="w", padx=2, pady=1)
        self.fm_modulator_ratio_var = tk.DoubleVar(value=1.0)
        modulator_slider = tk.Scale(fm_frame, from_=0.1, to=8.0, resolution=0.1, orient=tk.HORIZONTAL,
                                   length=180, variable=self.fm_modulator_ratio_var, font=("Arial", self.font_size),
                                   command=lambda val: self.poly_synth.set_fm_modulator_ratio(float(val)))
        modulator_slider.grid(row=0, column=1, padx=2, pady=1)
        
        # 変調インデックス
        ttk.Label(fm_frame, text="変調インデックス", style="My.TLabel").grid(row=1, column=0, sticky="w", padx=2, pady=1)
        self.fm_modulation_index_var = tk.DoubleVar(value=0.0)
        index_slider = tk.Scale(fm_frame, from_=0.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
                               length=180, variable=self.fm_modulation_index_var, font=("Arial", self.font_size),
                               command=lambda val: self.poly_synth.set_fm_modulation_index(float(val)))
        index_slider.grid(row=1, column=1, padx=2, pady=1)
        
        # リバーブエフェクト
        reverb_frame = ttk.LabelFrame(parent, text="リバーブ（シュレーダーアルゴリズム）", style="My.TLabelframe")
        reverb_frame.pack(fill="x", pady=3)
        
        # リバーブオンオフ
        self.reverb_enabled = tk.BooleanVar(value=False)
        reverb_check = tk.Checkbutton(reverb_frame, text="リバーブ有効", 
                                     variable=self.reverb_enabled,
                                     command=self.toggle_reverb,
                                     font=("Arial", self.font_size, "bold"),
                                     fg="green", selectcolor="lightgreen")
        reverb_check.pack(padx=5, pady=2)
        
        # リバーブパラメーター
        reverb_params_frame = ttk.Frame(reverb_frame)
        reverb_params_frame.pack(padx=5, pady=5)
        
        # 部屋サイズ
        ttk.Label(reverb_params_frame, text="部屋サイズ", style="My.TLabel").grid(row=0, column=0, sticky="w", padx=2, pady=1)
        self.reverb_room_size_var = tk.DoubleVar(value=0.8)
        room_slider = tk.Scale(reverb_params_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                              length=120, variable=self.reverb_room_size_var, font=("Arial", self.font_size-1),
                              command=lambda val: self.poly_synth.set_reverb_room_size(float(val)))
        room_slider.grid(row=0, column=1, padx=2, pady=1)
        # 初期値を設定
        self.poly_synth.set_reverb_room_size(0.8)
        
        # ウェットレベル
        ttk.Label(reverb_params_frame, text="ウェット", style="My.TLabel").grid(row=0, column=2, sticky="w", padx=2, pady=1)
        self.reverb_wet_var = tk.DoubleVar(value=0.3)
        wet_slider = tk.Scale(reverb_params_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                             length=120, variable=self.reverb_wet_var, font=("Arial", self.font_size-1),
                             command=lambda val: self.poly_synth.set_reverb_wet_level(float(val)))
        wet_slider.grid(row=0, column=3, padx=2, pady=1)
        # 初期値を設定
        self.poly_synth.set_reverb_wet_level(0.3)
        
        # ドライレベル
        ttk.Label(reverb_params_frame, text="ドライ", style="My.TLabel").grid(row=1, column=0, sticky="w", padx=2, pady=1)
        self.reverb_dry_var = tk.DoubleVar(value=0.7)
        dry_slider = tk.Scale(reverb_params_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                             length=120, variable=self.reverb_dry_var, font=("Arial", self.font_size-1),
                             command=lambda val: self.poly_synth.set_reverb_dry_level(float(val)))
        dry_slider.grid(row=1, column=1, padx=2, pady=1)
        # 初期値を設定
        self.poly_synth.set_reverb_dry_level(0.7)
        
        # ディレイタイム調整（コムフィルター1）
        ttk.Label(reverb_params_frame, text="ディレイ1(ms)", style="My.TLabel").grid(row=1, column=2, sticky="w", padx=2, pady=1)
        self.reverb_delay1_var = tk.DoubleVar(value=29.7)
        delay1_slider = tk.Scale(reverb_params_frame, from_=10.0, to=100.0, resolution=0.1, orient=tk.HORIZONTAL,
                                length=120, variable=self.reverb_delay1_var, font=("Arial", self.font_size-1),
                                command=lambda val: self.poly_synth.set_reverb_delay_time(0, float(val)))
        delay1_slider.grid(row=1, column=3, padx=2, pady=1)
        
        # プリセット
        preset_frame = ttk.LabelFrame(parent, text="プリセット", style="My.TLabelframe")
        preset_frame.pack(fill="x", pady=3)
        
        # プリセットボタン（10個、2行に配置）
        button_frame = ttk.Frame(preset_frame)
        button_frame.pack(padx=5, pady=5)
        
        # 1行目：プリセット1-5
        for i in range(1, 6):
            btn = ttk.Button(button_frame, text=f"プリセット{i}", width=12,
                           command=lambda slot=i: self.load_preset_slot(slot), style="My.TButton")
            btn.grid(row=0, column=i-1, padx=3, pady=2)
        
        # 2行目：プリセット6-10
        for i in range(6, 11):
            btn = ttk.Button(button_frame, text=f"プリセット{i}", width=12,
                           command=lambda slot=i: self.load_preset_slot(slot), style="My.TButton")
            btn.grid(row=1, column=i-6, padx=3, pady=2)
        
        # リセットボタン
        reset_btn = ttk.Button(preset_frame, text="リセット", command=self.reset_to_sine, style="My.TButton")
        reset_btn.pack(pady=5)

    def create_keyboard(self, parent):
        kb_frame = ttk.LabelFrame(parent, text="ピアノ鍵盤（キーボード：zxcvbnm）", style="My.TLabelframe")
        kb_frame.pack(fill="x", pady=5)
        
        # 大きめのピアノ鍵盤
        self.piano_canvas = tk.Canvas(kb_frame, width=800, height=80, bg="gray")
        self.piano_canvas.pack(padx=5, pady=5)
        
        # 白鍵
        white_keys = [60, 62, 64, 65, 67, 69, 71, 72]
        white_labels = ["z", "x", "c", "v", "b", "n", "m", ","]
        white_width = 100
        white_height = 80
        self.white_key_ids = {}
        
        for i, (note, label) in enumerate(zip(white_keys, white_labels)):
            x = i * white_width
            rect = self.piano_canvas.create_rectangle(x, 0, x+white_width, white_height, 
                                                    fill="white", outline="black", width=2)
            self.piano_canvas.create_text(x+white_width/2, white_height-15, text=label, 
                                        font=("Arial", 12, "bold"), fill="black")
            self.white_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note: self.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note: self.on_note_release(n))
        
        # 黒鍵
        black_keys = [61, 63, 66, 68, 70]
        black_labels = ["s", "d", "g", "h", "j"]
        black_width = 60
        black_height = 50
        black_positions = [70, 170, 370, 470, 570]
        self.black_key_ids = {}
        
        for pos, note, label in zip(black_positions, black_keys, black_labels):
            rect = self.piano_canvas.create_rectangle(pos, 0, pos+black_width, black_height,
                                                    fill="black", outline="gray", width=2)
            self.piano_canvas.create_text(pos+black_width/2, black_height-10, text=label,
                                        font=("Arial", 10, "bold"), fill="white")
            self.black_key_ids[note] = rect
            self.piano_canvas.tag_bind(rect, "<ButtonPress-1>", lambda e, n=note: self.on_note_press(n))
            self.piano_canvas.tag_bind(rect, "<ButtonRelease-1>", lambda e, n=note: self.on_note_release(n))

    def reset_to_sine(self):
        self.poly_synth.set_osc_type("sine")
        self.poly_synth.set_osc2_type("sine")
        self.poly_synth.set_osc2_detune(0.0)
        self.poly_synth.set_osc_mix(0.5)

        self.poly_synth.set_noise_mix(0.0)
        self.poly_synth.set_lfo_depth(0.0)
        self.poly_synth.set_resonance(0.0)
        self.poly_synth.set_attack(0.1)
        self.poly_synth.set_decay(0.2)
        self.poly_synth.set_sustain(0.7)
        self.poly_synth.set_release(0.5)
        self.poly_synth.set_cutoff(1000)
        # FM合成リセット
        self.poly_synth.set_fm_carrier_ratio(1.0)
        self.poly_synth.set_fm_modulator_ratio(1.0)
        self.poly_synth.set_fm_modulation_index(0.0)
        
        # UI変数リセット
        self.osc_var.set("サイン波")
        self.osc2_var.set("サイン波")
        self.osc2_detune_var.set(0.0)
        self.osc_mix_var.set(0.5)

        self.noise_mix_var.set(0.0)
        self.lfo_depth_var.set(0.0)
        self.resonance_var.set(0.0)
        self.attack_var.set(0.1)
        self.decay_var.set(0.2)
        self.sustain_var.set(0.7)
        self.release_var.set(0.5)
        self.cutoff_var.set(1000)
        self.lfo_rate_var.set(5.0)
        self.duty_cycle_var.set(0.5)
        # FM合成UIリセット
        self.fm_modulator_ratio_var.set(1.0)
        self.fm_modulation_index_var.set(0.0)
        # モードリセット
        self.ring_mod_enabled.set(False)
        self.sync_enabled.set(False)

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
            
            # OSC2パラメーター
            if "osc2_type" in preset:
                self.poly_synth.set_osc2_type(preset["osc2_type"])
            if "osc2_detune" in preset:
                self.poly_synth.set_osc2_detune(preset["osc2_detune"])
            if "osc_mix" in preset:
                self.poly_synth.set_osc_mix(preset["osc_mix"])
            
            # FM合成パラメーター
            if "fm_modulator_ratio" in preset:
                self.poly_synth.set_fm_modulator_ratio(preset["fm_modulator_ratio"])
            if "fm_modulation_index" in preset:
                self.poly_synth.set_fm_modulation_index(preset["fm_modulation_index"])
            
            # リング変調とシンク
            if "ring_mod_enabled" in preset:
                self.poly_synth.set_ring_mod_enabled(preset["ring_mod_enabled"])
            if "sync_enabled" in preset:
                self.poly_synth.set_sync_enabled(preset["sync_enabled"])
            
            # UI更新
            self.attack_var.set(preset["attack"])
            self.decay_var.set(preset["decay"])
            self.sustain_var.set(preset["sustain"])
            self.release_var.set(preset["release"])
            self.cutoff_var.set(preset["cutoff"])
            inv_osc_map = {v: k for k, v in self.osc_type_map.items()}
            self.osc_var.set(inv_osc_map[preset["osc_type"]])
            self.resonance_var.set(preset["resonance"])
            self.lfo_rate_var.set(preset["lfo_rate"])
            self.lfo_depth_var.set(preset["lfo_depth"])
            self.noise_mix_var.set(preset["noise_mix"])
            self.duty_cycle_var.set(preset["duty_cycle"])
            inv_noise_map = {v: k for k, v in self.noise_type_map.items()}
            self.noise_type_var.set(inv_noise_map[preset["noise_type"]])
            
            # OSC2 UI更新
            if "osc2_type" in preset:
                self.osc2_var.set(inv_osc_map[preset["osc2_type"]])
            if "osc2_detune" in preset:
                self.osc2_detune_var.set(preset["osc2_detune"])
            if "osc_mix" in preset:
                self.osc_mix_var.set(preset["osc_mix"])
            
            # FM合成 UI更新
            if "fm_modulator_ratio" in preset:
                self.fm_modulator_ratio_var.set(preset["fm_modulator_ratio"])
            if "fm_modulation_index" in preset:
                self.fm_modulation_index_var.set(preset["fm_modulation_index"])
            
            # モード UI更新
            if "ring_mod_enabled" in preset:
                self.ring_mod_enabled.set(preset["ring_mod_enabled"])
            if "sync_enabled" in preset:
                self.sync_enabled.set(preset["sync_enabled"])

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
        xdata = np.linspace(0, 400, WAVEFORM_BUFFER_SIZE)
        ydata = 60 - (buf * 60)
        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.wave_canvas.delete("all")
        if len(points) > 2:
            self.wave_canvas.create_line(points, fill="green", width=2)
        self.after(50, self.update_waveform)

    def update_spectrum(self):
        with buffer_lock:
            buf = waveform_buffer.copy()
        window = np.hamming(len(buf))
        fft_result = fft.rfft(buf * window)
        magnitude = np.abs(fft_result)
        raw_db = 20 * np.log10(magnitude + 1e-6)
        
        fixed_top = 60.0
        fixed_bottom = -60.0
        canvas_width = 400
        canvas_height = 120
        
        f_min = 20
        f_max = 20000
        freqs = fft.rfftfreq(len(buf), 1.0 / SAMPLE_RATE)
        valid_indices = np.where((freqs >= f_min) & (freqs <= f_max))[0]
        freqs_valid = freqs[valid_indices]
        
        margin_x = 15
        xdata = margin_x + (np.log10(freqs_valid) - np.log10(f_min)) / (np.log10(f_max) - np.log10(f_min)) * (canvas_width - 2 * margin_x)
        
        db_valid = raw_db[valid_indices]
        ydata = canvas_height - ((db_valid - fixed_bottom) / (fixed_top - fixed_bottom)) * canvas_height

        points = []
        for x, y in zip(xdata, ydata):
            points.extend([x, y])
        self.spec_canvas.delete("all")
        if len(points) > 2:
            self.spec_canvas.create_line(points, fill="yellow", width=2)
        self.after(100, self.update_spectrum)

if __name__ == "__main__":
    from synth_core import PolySynth
    poly_synth = PolySynth(max_voices=6, sample_rate=SAMPLE_RATE)
    app = SynthGUI(poly_synth)
    app.mainloop()
