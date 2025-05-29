# reverb.py
import numpy as np
from typing import List

class CombFilter:
    """コムフィルター（IIRディレイライン）"""
    def __init__(self, delay_samples: int, feedback: float, sample_rate: int = 44100):
        self.delay_samples = delay_samples
        self.feedback = feedback
        self.sample_rate = sample_rate
        
        # ディレイバッファ
        self.buffer = np.zeros(delay_samples, dtype=np.float32)
        self.write_index = 0
        
    def process(self, input_sample: float) -> float:
        """コムフィルター処理"""
        # ディレイバッファから読み取り
        delayed_sample = self.buffer[self.write_index]
        
        # フィードバック処理
        self.buffer[self.write_index] = input_sample + (delayed_sample * self.feedback)
        
        # 書き込みインデックス更新
        self.write_index = (self.write_index + 1) % self.delay_samples
        
        return delayed_sample
    
    def set_feedback(self, feedback: float):
        """フィードバック量を設定"""
        self.feedback = max(0.0, min(0.99, feedback))  # 安定性のため0.99以下に制限

class AllPassFilter:
    """オールパスフィルター"""
    def __init__(self, delay_samples: int, gain: float, sample_rate: int = 44100):
        self.delay_samples = delay_samples
        self.gain = gain
        self.sample_rate = sample_rate
        
        # ディレイバッファ
        self.buffer = np.zeros(delay_samples, dtype=np.float32)
        self.write_index = 0
        
    def process(self, input_sample: float) -> float:
        """オールパスフィルター処理"""
        # ディレイバッファから読み取り
        delayed_sample = self.buffer[self.write_index]
        
        # オールパス処理
        # y[n] = -g*x[n] + x[n-d] + g*y[n-d]
        output = -self.gain * input_sample + delayed_sample
        self.buffer[self.write_index] = input_sample + self.gain * output
        
        # 書き込みインデックス更新
        self.write_index = (self.write_index + 1) % self.delay_samples
        
        return output
    
    def set_gain(self, gain: float):
        """ゲインを設定"""
        self.gain = max(-0.99, min(0.99, gain))  # 安定性のため±0.99以下に制限

class SchroederReverb:
    """シュレーダーのリバーブアルゴリズム
    
    4つのコムフィルターを並列接続し、
    その後に2つのオールパスフィルターを直列接続
    """
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
        # シュレーダーの推奨ディレイタイム（ミリ秒）
        # より長いディレイタイムでリバーブ効果を強化
        self.comb_delays_ms = [50.0, 56.0, 61.0, 68.0]  # ミリ秒
        self.allpass_delays_ms = [6.0, 10.0]  # ミリ秒
        
        # パラメーター
        self.room_size = 0.5      # 部屋のサイズ（0.0-1.0）
        self.damping = 0.5        # ダンピング（0.0-1.0）
        self.wet_level = 0.3      # ウェットレベル（0.0-1.0）
        self.dry_level = 0.7      # ドライレベル（0.0-1.0）
        self.width = 1.0          # ステレオ幅（0.0-1.0）
        
        # フィルター初期化
        self.init_filters()
        
    def init_filters(self):
        """フィルターを初期化"""
        # コムフィルター（4つ並列）
        self.comb_filters = []
        for delay_ms in self.comb_delays_ms:
            delay_samples = int(delay_ms * self.sample_rate / 1000.0)
            feedback = 0.5 + (self.room_size * 0.4)  # フィードバック量を強化（0.5-0.9）
            self.comb_filters.append(CombFilter(delay_samples, feedback, self.sample_rate))
        
        # オールパスフィルター（2つ直列）
        self.allpass_filters = []
        for delay_ms in self.allpass_delays_ms:
            delay_samples = int(delay_ms * self.sample_rate / 1000.0)
            gain = 0.5  # オールパスゲイン
            self.allpass_filters.append(AllPassFilter(delay_samples, gain, self.sample_rate))
    
    def process(self, input_sample: float) -> float:
        """リバーブ処理（モノラル）"""
        # コムフィルター並列処理
        comb_output = 0.0
        for comb_filter in self.comb_filters:
            comb_output += comb_filter.process(input_sample)
        
        # オールパスフィルター直列処理
        allpass_output = comb_output
        for allpass_filter in self.allpass_filters:
            allpass_output = allpass_filter.process(allpass_output)
        
        # ドライ/ウェットミックス
        wet_signal = allpass_output * self.wet_level
        dry_signal = input_sample * self.dry_level
        
        return dry_signal + wet_signal
    
    def set_room_size(self, room_size: float):
        """部屋のサイズを設定（0.0-1.0）"""
        self.room_size = max(0.0, min(1.0, room_size))
        # コムフィルターのフィードバック量を更新
        for comb_filter in self.comb_filters:
            feedback = 0.5 + (self.room_size * 0.4)  # 強化されたフィードバック量
            comb_filter.set_feedback(feedback)
    
    def set_damping(self, damping: float):
        """ダンピングを設定（0.0-1.0）"""
        self.damping = max(0.0, min(1.0, damping))
        # 実装では高周波減衰フィルターを追加可能
        # 現在はシンプル実装のため省略
    
    def set_wet_level(self, wet_level: float):
        """ウェットレベルを設定（0.0-1.0）"""
        self.wet_level = max(0.0, min(1.0, wet_level))
    
    def set_dry_level(self, dry_level: float):
        """ドライレベルを設定（0.0-1.0）"""
        self.dry_level = max(0.0, min(1.0, dry_level))
    
    def set_width(self, width: float):
        """ステレオ幅を設定（0.0-1.0）"""
        self.width = max(0.0, min(1.0, width))
    
    def set_delay_time(self, comb_index: int, delay_ms: float):
        """特定のコムフィルターのディレイタイムを設定"""
        if 0 <= comb_index < len(self.comb_delays_ms):
            self.comb_delays_ms[comb_index] = max(1.0, min(100.0, delay_ms))
            # フィルターを再初期化
            delay_samples = int(self.comb_delays_ms[comb_index] * self.sample_rate / 1000.0)
            feedback = 0.5 + (self.room_size * 0.4)  # 強化されたフィードバック量
            self.comb_filters[comb_index] = CombFilter(delay_samples, feedback, self.sample_rate)
    
    def get_delay_times(self) -> List[float]:
        """現在のディレイタイムを取得"""
        return self.comb_delays_ms.copy()

class StereoSchroederReverb:
    """ステレオ版シュレーダーリバーブ"""
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
        # 左右チャンネル用のリバーブ
        self.left_reverb = SchroederReverb(sample_rate)
        self.right_reverb = SchroederReverb(sample_rate)
        
        # 右チャンネルのディレイタイムを少しずらしてステレオ効果を作る
        right_delays = [31.3, 39.7, 43.3, 46.1]  # 左とは異なる値
        for i, delay_ms in enumerate(right_delays):
            self.right_reverb.set_delay_time(i, delay_ms)
    
    def process_stereo(self, left_input: float, right_input: float) -> tuple:
        """ステレオ処理"""
        left_output = self.left_reverb.process(left_input)
        right_output = self.right_reverb.process(right_input)
        return left_output, right_output
    
    def process_mono_to_stereo(self, input_sample: float) -> tuple:
        """モノラル入力をステレオリバーブで処理"""
        left_output = self.left_reverb.process(input_sample)
        right_output = self.right_reverb.process(input_sample)
        return left_output, right_output
    
    def set_room_size(self, room_size: float):
        """両チャンネルの部屋サイズを設定"""
        self.left_reverb.set_room_size(room_size)
        self.right_reverb.set_room_size(room_size)
    
    def set_damping(self, damping: float):
        """両チャンネルのダンピングを設定"""
        self.left_reverb.set_damping(damping)
        self.right_reverb.set_damping(damping)
    
    def set_wet_level(self, wet_level: float):
        """両チャンネルのウェットレベルを設定"""
        self.left_reverb.set_wet_level(wet_level)
        self.right_reverb.set_wet_level(wet_level)
    
    def set_dry_level(self, dry_level: float):
        """両チャンネルのドライレベルを設定"""
        self.left_reverb.set_dry_level(dry_level)
        self.right_reverb.set_dry_level(dry_level)
    
    def get_delay_times(self) -> tuple:
        """左右のディレイタイムを取得"""
        return self.left_reverb.get_delay_times(), self.right_reverb.get_delay_times()
