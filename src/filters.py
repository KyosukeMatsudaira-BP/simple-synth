# filters.py
import math

SAMPLE_RATE = 44100

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
