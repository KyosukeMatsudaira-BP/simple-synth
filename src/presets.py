# presets.py - プリセットデータ定義

# 既定のプリセット（上書き不可）- 10個の特徴的なサウンド
default_presets = {
    # 1. クラシックパッド - 温かみのあるストリングス風（OSC2使用）
    1: {"attack": 1.2, "decay": 0.8, "sustain": 0.9, "release": 2.5,
        "cutoff": 1200, "osc_type": "sine", "resonance": 0.2,
        "lfo_rate": 2.5, "lfo_depth": 0.005, "noise_mix": 0.05,
        "duty_cycle": 0.5, "noise_type": "pink",
        "osc2_type": "triangle", "osc2_detune": 7.0, "osc_mix": 0.3,
        "fm_modulator_ratio": 1.0, "fm_modulation_index": 0.0,
        "ring_mod_enabled": False, "sync_enabled": False},
    
    # 2. アシッドベース - TB-303風のスクエア波ベース（OSC2使用）
    2: {"attack": 0.01, "decay": 0.15, "sustain": 0.3, "release": 0.2,
        "cutoff": 800, "osc_type": "square", "resonance": 8.5,
        "lfo_rate": 0.0, "lfo_depth": 0.0, "noise_mix": 0.0,
        "duty_cycle": 0.3, "noise_type": "white",
        "osc2_type": "square", "osc2_detune": -12.0, "osc_mix": 0.2,
        "fm_modulator_ratio": 1.0, "fm_modulation_index": 0.0,
        "ring_mod_enabled": False, "sync_enabled": False},
    
    # 3. FM エレピ - 電子ピアノ風のFM合成
    3: {"attack": 0.02, "decay": 0.8, "sustain": 0.4, "release": 1.2,
        "cutoff": 3000, "osc_type": "sine", "resonance": 0.1,
        "lfo_rate": 0.0, "lfo_depth": 0.0, "noise_mix": 0.02,
        "duty_cycle": 0.5, "noise_type": "white",
        "osc2_type": "sine", "osc2_detune": 0.0, "osc_mix": 0.5,
        "fm_modulator_ratio": 4.0, "fm_modulation_index": 2.5,
        "ring_mod_enabled": False, "sync_enabled": False},
    
    # 4. リードシンセ - 鋭いリード音（OSC2使用）
    4: {"attack": 0.05, "decay": 0.3, "sustain": 0.8, "release": 0.4,
        "cutoff": 2200, "osc_type": "sawtooth", "resonance": 4.0,
        "lfo_rate": 6.0, "lfo_depth": 0.015, "noise_mix": 0.0,
        "duty_cycle": 0.6, "noise_type": "white",
        "osc2_type": "square", "osc2_detune": 5.0, "osc_mix": 0.4,
        "fm_modulator_ratio": 1.0, "fm_modulation_index": 0.0,
        "ring_mod_enabled": False, "sync_enabled": False},
    
    # 5. ブラス - 金管楽器風（OSC2使用）
    5: {"attack": 0.1, "decay": 0.2, "sustain": 0.9, "release": 0.8,
        "cutoff": 1800, "osc_type": "sawtooth", "resonance": 1.5,
        "lfo_rate": 4.5, "lfo_depth": 0.008, "noise_mix": 0.08,
        "duty_cycle": 0.5, "noise_type": "pink",
        "osc2_type": "triangle", "osc2_detune": -7.0, "osc_mix": 0.6,
        "fm_modulator_ratio": 1.0, "fm_modulation_index": 0.0,
        "ring_mod_enabled": False, "sync_enabled": False},
    
    # 6. アルペジエーター - 短いプラック音（OSC2使用）
    6: {"attack": 0.01, "decay": 0.4, "sustain": 0.0, "release": 0.6,
        "cutoff": 3500, "osc_type": "triangle", "resonance": 2.0,
        "lfo_rate": 0.0, "lfo_depth": 0.0, "noise_mix": 0.05,
        "duty_cycle": 0.5, "noise_type": "white",
        "osc2_type": "sine", "osc2_detune": 1200.0, "osc_mix": 0.25,
        "fm_modulator_ratio": 1.0, "fm_modulation_index": 0.0,
        "ring_mod_enabled": False, "sync_enabled": False},
    
    # 7. シンクリード - オシレーターシンク使用
    7: {"attack": 0.02, "decay": 0.1, "sustain": 0.7, "release": 0.3,
        "cutoff": 2800, "osc_type": "sawtooth", "resonance": 3.5,
        "lfo_rate": 7.0, "lfo_depth": 0.02, "noise_mix": 0.0,
        "duty_cycle": 0.4, "noise_type": "white",
        "osc2_type": "square", "osc2_detune": 0.0, "osc_mix": 0.8,
        "fm_modulator_ratio": 1.0, "fm_modulation_index": 0.0,
        "ring_mod_enabled": False, "sync_enabled": True},
    
    # 8. リングモッドベル - リング変調でベル音
    8: {"attack": 0.01, "decay": 1.5, "sustain": 0.2, "release": 2.0,
        "cutoff": 4000, "osc_type": "sine", "resonance": 0.5,
        "lfo_rate": 0.0, "lfo_depth": 0.0, "noise_mix": 0.0,
        "duty_cycle": 0.5, "noise_type": "white",
        "osc2_type": "sine", "osc2_detune": 700.0, "osc_mix": 0.5,
        "fm_modulator_ratio": 1.0, "fm_modulation_index": 0.0,
        "ring_mod_enabled": True, "sync_enabled": False},
    
    # 9. ノイズパーカッション - ドラム風（OSC2使用）
    9: {"attack": 0.001, "decay": 0.08, "sustain": 0.0, "release": 0.15,
        "cutoff": 1500, "osc_type": "square", "resonance": 6.0,
        "lfo_rate": 0.0, "lfo_depth": 0.0, "noise_mix": 0.8,
        "duty_cycle": 0.2, "noise_type": "white",
        "osc2_type": "triangle", "osc2_detune": -600.0, "osc_mix": 0.3,
        "fm_modulator_ratio": 1.0, "fm_modulation_index": 0.0,
        "ring_mod_enabled": False, "sync_enabled": False},
    
    # 10. FM ベル - 複雑なFM合成
    10: {"attack": 0.01, "decay": 2.0, "sustain": 0.3, "release": 3.0,
         "cutoff": 5000, "osc_type": "sine", "resonance": 0.2,
         "lfo_rate": 1.5, "lfo_depth": 0.003, "noise_mix": 0.0,
         "duty_cycle": 0.5, "noise_type": "white",
         "osc2_type": "sine", "osc2_detune": 0.0, "osc_mix": 0.5,
         "fm_modulator_ratio": 3.5, "fm_modulation_index": 4.0,
         "ring_mod_enabled": False, "sync_enabled": False}
}
