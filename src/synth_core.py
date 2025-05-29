# synth_core.py
import math, random, time
from enum import Enum
from lfo import LFO, SAMPLE_RATE as LFO_SR
from filters import ResonantLPF, SAMPLE_RATE as FILTER_SR
from reverb import SchroederReverb

SAMPLE_RATE = 44100  # 共通のサンプルレート

class SynthMode(Enum):
    """シンセサイザーの合成モード"""
    NORMAL = "normal"        # 通常のミックス
    RING_MOD = "ring_mod"    # リング変調
    SYNC = "sync"            # オシレーターシンク
    FM = "fm"                # FM合成

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
        # OSC1 (拡張)
        self.phase = 0.0
        self.frequency = 0.0
        self.osc_type = "sine"
        self.duty_cycle = 0.5
        
        # OSC2 (拡張)
        self.osc2_type = "sine"
        self.osc2_phase = 0.0
        self.osc2_detune = 0.0  # セント単位 (-50 ~ +50)
        self.osc_mix = 0.5  # 0.0=OSC1のみ, 1.0=OSC2のみ
        
        # FM合成パラメーター
        self.fm_modulator_ratio = 1.0  # モジュレーター周波数比（キャリアを1とした時の比率）
        self.fm_modulation_index = 0.0 # 変調インデックス（初期値0.0）
        
        # 合成モード
        self.synth_mode = SynthMode.NORMAL
        # モードフラグ
        self.ring_mod_enabled = False
        self.sync_enabled = False
        
        # リバーブエフェクト
        self.reverb = SchroederReverb(SAMPLE_RATE)
        self.reverb_enabled = False
        
        

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
        self.attack = a; self.decay = d; self.sustain = s; self.release = r

    def set_cutoff(self, cutoff: float):
        self.cutoff = cutoff

    def set_osc_type(self, osc_type: str):
        self.osc_type = osc_type

    def set_osc2_type(self, osc2_type: str):
        self.osc2_type = osc2_type

    def set_osc2_detune(self, detune: float):
        self.osc2_detune = detune

    def set_osc_mix(self, mix: float):
        self.osc_mix = mix



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

    def set_synth_mode(self, mode: SynthMode):
        """合成モードを設定"""
        self.synth_mode = mode

    def set_fm_carrier_ratio(self, ratio: float):
        """互換性のため残す（実際は使用されない）"""
        pass

    def set_fm_modulator_ratio(self, ratio: float):
        """FMモジュレーター周波数比を設定"""
        self.fm_modulator_ratio = ratio

    def set_fm_modulation_index(self, index: float):
        """FM変調インデックスを設定"""
        self.fm_modulation_index = index

    def process_fm_synthesis(self, carrier_freq: float, modulator_freq: float) -> float:
        """FM合成処理（排他モード用）
        
        正しいFM合成の数式:
        キャリア周波数: fc = carrier_freq（基音周波数）
        モジュレーター周波数: fm = carrier_freq × fm_modulator_ratio
        モジュレーター信号: M(t) = sin(2π × fm × t)
        変調された位相: φ(t) = 2π × fc × t + fm_modulation_index × M(t)
        最終出力: sin(φ(t))
        
        モジュレーター比: 基音に対するモジュレーター周波数の比率（例: 1.0 = 基音と同じ、2.0 = 1オクターブ上）
        変調インデックス: 変調の深さ（0 = 変調なし、大きいほど複雑な倍音）
        """
        # 実際の周波数計算（キャリアは基音周波数そのまま）
        actual_carrier_freq = carrier_freq
        actual_modulator_freq = carrier_freq * self.fm_modulator_ratio
        
        # モジュレーター位相の初期化
        if not hasattr(self, 'fm_modulator_phase'):
            self.fm_modulator_phase = 0.0
        if not hasattr(self, 'fm_carrier_phase'):
            self.fm_carrier_phase = 0.0
        
        # モジュレーター信号生成
        modulator_signal = math.sin(self.fm_modulator_phase)
        
        # FM変調: キャリアの位相を変調
        # 位相変調 = 変調インデックス × モジュレーター信号
        phase_modulation = self.fm_modulation_index * modulator_signal
        
        # 変調されたキャリア信号生成
        modulated_phase = self.fm_carrier_phase + phase_modulation
        fm_output = math.sin(modulated_phase)
        
        # 位相更新
        carrier_phase_inc = (2.0 * math.pi * actual_carrier_freq) / SAMPLE_RATE
        modulator_phase_inc = (2.0 * math.pi * actual_modulator_freq) / SAMPLE_RATE
        
        self.fm_carrier_phase += carrier_phase_inc
        self.fm_modulator_phase += modulator_phase_inc
        
        # 位相を2πで正規化
        if self.fm_carrier_phase >= 2.0 * math.pi:
            self.fm_carrier_phase -= 2.0 * math.pi
        if self.fm_modulator_phase >= 2.0 * math.pi:
            self.fm_modulator_phase -= 2.0 * math.pi
        
        return fm_output

    def process_oscillator_sync(self, osc1_freq: float, osc2_freq: float) -> float:
        """オシレーターシンク処理（排他モード用）"""
        # OSC1波形生成
        osc1_waveform = self.generate_waveform(self.osc_type, self.phase)
        
        # OSC1の位相リセット検出（ゼロクロッシング）
        if not hasattr(self, 'prev_osc1_phase'):
            self.prev_osc1_phase = 0.0
        
        # OSC1が一周した時にOSC2の位相をリセット
        if self.prev_osc1_phase > self.phase:  # 位相が巻き戻った = 一周完了
            self.osc2_phase = 0.0
        
        self.prev_osc1_phase = self.phase
        
        # OSC2波形生成（シンクされた）
        osc2_waveform = self.generate_waveform(self.osc2_type, self.osc2_phase)
        
        # シンク効果: OSC1とOSC2のミックス
        return (1.0 - self.osc_mix) * osc1_waveform + self.osc_mix * osc2_waveform

    def process_ring_modulation(self, osc1_freq: float, osc2_freq: float) -> float:
        """リング変調処理（排他モード用）"""
        # OSC1とOSC2の波形生成
        osc1_waveform = self.generate_waveform(self.osc_type, self.phase)
        osc2_waveform = self.generate_waveform(self.osc2_type, self.osc2_phase)
        
        # リング変調: OSC1 × OSC2
        ring_mod_waveform = osc1_waveform * osc2_waveform
        
        return ring_mod_waveform

    def process_normal_mix(self, osc1_freq: float, osc2_freq: float) -> float:
        """通常ミックス処理（排他モード用）"""
        # OSC1とOSC2の波形生成
        osc1_waveform = self.generate_waveform(self.osc_type, self.phase)
        osc2_waveform = self.generate_waveform(self.osc2_type, self.osc2_phase)
        
        # 通常のミックス（フィルターなし）
        return (1.0 - self.osc_mix) * osc1_waveform + self.osc_mix * osc2_waveform

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

    def generate_waveform(self, osc_type: str, phase: float) -> float:
        """指定された波形タイプと位相から波形を生成"""
        if osc_type == "sine":
            return math.sin(phase)
        elif osc_type == "triangle":
            return 2 * abs(2 * ((phase / (2 * math.pi)) - math.floor(phase / (2 * math.pi) + 0.5))) - 1
        elif osc_type == "square":
            return 1.0 if (phase / (2 * math.pi)) < self.duty_cycle else -1.0
        elif osc_type == "sawtooth":
            return 2 * (phase / (2 * math.pi)) - 1
        else:
            return math.sin(phase)

    def oscillator(self) -> float:
        # LFO処理
        lfo_val = self.lfo.process()
        modulated_freq = self.frequency * (1 + self.lfo.depth * lfo_val)
        
        # OSC2の周波数計算（デチューン適用）
        detune_ratio = 2 ** (self.osc2_detune / 1200.0)
        osc2_freq = modulated_freq * detune_ratio
        
        # 位相更新を波形生成の前に行う
        phase_inc = (2.0 * math.pi * modulated_freq) / SAMPLE_RATE
        self.phase += phase_inc
        if self.phase >= 2.0 * math.pi:
            self.phase -= 2.0 * math.pi
            
        # OSC2の位相更新（デチューン適用）
        osc2_phase_inc = (2.0 * math.pi * osc2_freq) / SAMPLE_RATE
        self.osc2_phase += osc2_phase_inc
        if self.osc2_phase >= 2.0 * math.pi:
            self.osc2_phase -= 2.0 * math.pi
        
        # 排他的優先順位モード（優先順位: FM > シンク > リング変調 > 通常ミックス）
        mixed_waveform = 0.0
        
        # 排他的優先順位モード判定
        # 1. FM合成 (最優先) - 変調インデックス > 0
        if self.fm_modulation_index > 0.0:
            mixed_waveform = self.process_fm_synthesis(modulated_freq, osc2_freq)
        # 2. リング変調 - 明示的に有効化された場合
        elif self.ring_mod_enabled:
            mixed_waveform = self.process_ring_modulation(modulated_freq, osc2_freq)
        # 3. オシレーターシンク - 明示的に有効化された場合
        elif self.sync_enabled:
            mixed_waveform = self.process_oscillator_sync(modulated_freq, osc2_freq)
        # 4. 通常ミックス (デフォルト)
        else:
            mixed_waveform = self.process_normal_mix(modulated_freq, osc2_freq)
            
        # ノイズ生成
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
            
        # 最終ミックス（オシレーター + ノイズ）
        final_wave = (1 - self.noise_mix) * mixed_waveform + self.noise_mix * noise
        return final_wave

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
        self.synth.set_osc2_type(master_params.get("osc2_type", "sine"))
        self.synth.set_osc2_detune(master_params.get("osc2_detune", 0.0))
        self.synth.set_osc_mix(master_params.get("osc_mix", 0.5))
        self.synth.set_resonance(master_params.get("resonance", 0.0))
        self.synth.set_lfo_rate(master_params.get("lfo_rate", 5.0))
        self.synth.set_lfo_depth(master_params.get("lfo_depth", 0.0))
        self.synth.set_noise_mix(master_params.get("noise_mix", 0.0))
        self.synth.set_duty_cycle(master_params.get("duty_cycle", 0.5))
        self.synth.noise_type = master_params.get("noise_type", "white")
        # FM合成パラメーター設定
        self.synth.set_fm_carrier_ratio(master_params.get("fm_carrier_ratio", 1.0))
        self.synth.set_fm_modulator_ratio(master_params.get("fm_modulator_ratio", 1.0))
        self.synth.set_fm_modulation_index(master_params.get("fm_modulation_index", 0.0))
        # モードフラグ設定
        self.synth.ring_mod_enabled = master_params.get("ring_mod_enabled", False)
        self.synth.sync_enabled = master_params.get("sync_enabled", False)
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

class PolySynth:
    def __init__(self, max_voices=10, sample_rate=SAMPLE_RATE):
        self.max_voices = max_voices
        self.voices = []
        self.sample_rate = sample_rate
        self.attack = 0.1
        self.decay = 0.2
        self.sustain = 0.7
        self.release = 0.5
        self.cutoff = 1000.0
        self.osc_type = "sine"
        # OSC2関連パラメーター
        self.osc2_type = "sine"
        self.osc2_detune = 0.0
        self.osc_mix = 0.5
        self.resonance = 0.0
        self.lfo_rate = 5.0
        self.lfo_depth = 0.0
        self.noise_mix = 0.0
        self.duty_cycle = 0.5
        self.noise_type = "white"
        # FM合成パラメーター
        self.fm_modulator_ratio = 1.0
        self.fm_modulation_index = 0.0
        # 合成モード
        self.synth_mode = SynthMode.NORMAL
        # モードフラグ
        self.ring_mod_enabled = False
        self.sync_enabled = False
        # リバーブエフェクト
        self.reverb = SchroederReverb(SAMPLE_RATE)
        self.reverb_enabled = False

    def update_parameters(self):
        for voice in self.voices:
            voice.synth.set_adsr(self.attack, self.decay, self.sustain, self.release)
            voice.synth.set_cutoff(self.cutoff)
            voice.synth.set_osc_type(self.osc_type)
            voice.synth.set_osc2_type(self.osc2_type)
            voice.synth.set_osc2_detune(self.osc2_detune)
            voice.synth.set_osc_mix(self.osc_mix)
            voice.synth.set_resonance(self.resonance)
            voice.synth.set_lfo_rate(self.lfo_rate)
            voice.synth.set_lfo_depth(self.lfo_depth)
            voice.synth.set_noise_mix(self.noise_mix)
            voice.synth.set_duty_cycle(self.duty_cycle)
            voice.synth.noise_type = self.noise_type

    def set_attack(self, a: float):
        self.attack = a; self.update_parameters()
    def set_decay(self, d: float):
        self.decay = d; self.update_parameters()
    def set_sustain(self, s: float):
        self.sustain = s; self.update_parameters()
    def set_release(self, r: float):
        self.release = r; self.update_parameters()
    def set_cutoff(self, c: float):
        self.cutoff = c; self.update_parameters()
    def set_osc_type(self, osc: str):
        self.osc_type = osc; self.update_parameters()
    def set_osc2_type(self, osc2: str):
        self.osc2_type = osc2; self.update_parameters()
    def set_osc2_detune(self, detune: float):
        self.osc2_detune = detune; self.update_parameters()
    def set_osc_mix(self, mix: float):
        self.osc_mix = mix; self.update_parameters()
    def set_resonance(self, res: float):
        self.resonance = res; self.update_parameters()
    def set_lfo_rate(self, rate: float):
        self.lfo_rate = rate; self.update_parameters()
    def set_lfo_depth(self, depth: float):
        self.lfo_depth = depth; self.update_parameters()
    def set_noise_mix(self, mix: float):
        self.noise_mix = mix; self.update_parameters()
    def set_duty_cycle(self, duty: float):
        self.duty_cycle = duty; self.update_parameters()
    def set_noise_type(self, ntype: str):
        self.noise_type = ntype; self.update_parameters()

    def set_synth_mode(self, mode: SynthMode):
        """合成モードを設定"""
        self.synth_mode = mode
        for voice in self.voices:
            voice.synth.set_synth_mode(mode)
    
    def set_fm_carrier_ratio(self, ratio: float):
        """互換性のため残す（実際は使用されない）"""
        pass
    
    def set_fm_modulator_ratio(self, ratio: float):
        """FMモジュレーター周波数比を設定"""
        self.fm_modulator_ratio = ratio
        for voice in self.voices:
            voice.synth.set_fm_modulator_ratio(ratio)
    
    def set_fm_modulation_index(self, index: float):
        """FM変調インデックスを設定"""
        self.fm_modulation_index = index
        for voice in self.voices:
            voice.synth.set_fm_modulation_index(index)
    
    def set_ring_mod_enabled(self, enabled: bool):
        """リング変調の有効/無効を設定"""
        for voice in self.voices:
            voice.synth.ring_mod_enabled = enabled
    
    def set_sync_enabled(self, enabled: bool):
        """オシレーターシンクの有効/無効を設定"""
        for voice in self.voices:
            voice.synth.sync_enabled = enabled

    # 互換性のため残しておく（GUIで使用される可能性）
    def set_osc2_level(self, level: float):
        """互換性のため残す（実際は使用されない）"""
        pass

    def note_on(self, note_number: int):
        # 既存の同じノートを探す（リトリガー対応）
        for voice in self.voices:
            if voice.note_number == note_number and voice.active:
                # 同じノートが既に鳴っている場合は、ソフトリトリガー
                voice.synth.note_on(note_number)
                voice.start_time = time.time()  # タイムスタンプ更新
                return
        
        master_params = {
            "attack": self.attack,
            "decay": self.decay,
            "sustain": self.sustain,
            "release": self.release,
            "cutoff": self.cutoff,
            "osc_type": self.osc_type,
            "osc2_type": self.osc2_type,
            "osc2_detune": self.osc2_detune,
            "osc_mix": self.osc_mix,
            "resonance": self.resonance,
            "lfo_rate": self.lfo_rate,
            "lfo_depth": self.lfo_depth,
            "noise_mix": self.noise_mix,
            "duty_cycle": self.duty_cycle,
            "noise_type": self.noise_type,
            "fm_modulator_ratio": self.fm_modulator_ratio,
            "fm_modulation_index": self.fm_modulation_index,
            "ring_mod_enabled": self.ring_mod_enabled,
            "sync_enabled": self.sync_enabled
        }
        
        if len(self.voices) < self.max_voices:
            # 新しいボイスを作成
            new_voice = Voice(note_number, master_params, self.sample_rate)
            self.voices.append(new_voice)
        else:
            # ボイス数が上限に達している場合、最も古いボイスを再利用
            # ただし、リリース中のボイスを優先的に選択
            release_voices = [v for v in self.voices if v.synth.env_state == 'release']
            if release_voices:
                oldest_voice = min(release_voices, key=lambda v: v.start_time)
            else:
                oldest_voice = min(self.voices, key=lambda v: v.start_time)
            
            # ソフトな移行のため、急激な音量変化を避ける
            oldest_voice.note_number = note_number
            oldest_voice.synth.note_on(note_number)
            oldest_voice.start_time = time.time()

    def note_off(self, note_number: int):
        for voice in self.voices:
            if voice.note_number == note_number and voice.active:
                voice.note_off()

    # リバーブ関連メソッド
    def set_reverb_enabled(self, enabled: bool):
        """リバーブの有効/無効を設定"""
        self.reverb_enabled = enabled
    
    def set_reverb_room_size(self, room_size: float):
        """リバーブの部屋サイズを設定"""
        self.reverb.set_room_size(room_size)
    
    def set_reverb_damping(self, damping: float):
        """リバーブのダンピングを設定"""
        self.reverb.set_damping(damping)
    
    def set_reverb_wet_level(self, wet_level: float):
        """リバーブのウェットレベルを設定"""
        self.reverb.set_wet_level(wet_level)
    
    def set_reverb_dry_level(self, dry_level: float):
        """リバーブのドライレベルを設定"""
        self.reverb.set_dry_level(dry_level)
    
    def set_reverb_delay_time(self, comb_index: int, delay_ms: float):
        """特定のコムフィルターのディレイタイムを設定"""
        self.reverb.set_delay_time(comb_index, delay_ms)

    def process(self) -> float:
        sample_sum = 0.0
        active_voices = []
        for voice in self.voices:
            sample_sum += voice.process()
            if voice.active:
                active_voices.append(voice)
        self.voices = active_voices
        
        # 基本的な音量正規化
        dry_signal = sample_sum / max(1, self.max_voices)
        
        # リバーブ処理
        if self.reverb_enabled:
            return self.reverb.process(dry_signal)
        else:
            return dry_signal
