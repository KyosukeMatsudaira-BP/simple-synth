# synth_core.py
import math, random, time
from lfo import LFO, SAMPLE_RATE as LFO_SR
from filters import ResonantLPF, SAMPLE_RATE as FILTER_SR

SAMPLE_RATE = 44100  # 共通のサンプルレート

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
        self.attack = a; self.decay = d; self.sustain = s; self.release = r

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
