# lfo.py
import math

SAMPLE_RATE = 44100

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
