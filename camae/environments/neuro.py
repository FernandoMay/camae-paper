import numpy as np
from typing import Dict
from .base_channel import BaseChannel

class NeuroChannel(BaseChannel):
    def __init__(self, num_subcarriers: int = 64, n0: float = 1e-3,
                 cauchy_gamma: float = 0.04, sync_ratio: float = 0.3):
        super().__init__(num_subcarriers, n0)
        self.gamma_c = float(cauchy_gamma)

        self.sync_mask = np.random.rand(self.N) < sync_ratio
        self.sync_phase = np.zeros(self.N, dtype=np.float64)
        self.drift_phases = np.random.uniform(0, 2 * np.pi, self.N)

        self.base_freqs = np.random.uniform(0.5, 2.0, self.N)

    def step(self) -> Dict[str, np.ndarray]:
        self.t_step += 1

        phase_slips = np.random.standard_cauchy(self.N) * self.gamma_c

        self.sync_phase = (self.sync_phase + phase_slips * 0.1) % (2 * np.pi)
        self.drift_phases = (self.drift_phases + phase_slips) % (2 * np.pi)

        phases = np.where(self.sync_mask, self.sync_phase, self.drift_phases)

        H = np.exp(1j * phases) + self.generate_awgn()
        H_sq = np.abs(H) ** 2

        mean_phase = np.angle(np.mean(np.exp(1j * phases)))
        plv_vector = np.abs(np.exp(1j * (phases - mean_phase)))

        C = np.where(self.sync_mask, 0.8, 0.2).astype(np.float64)

        I = np.abs(np.random.normal(0, 5e-4, self.N)) ** 2

        R = plv_vector * np.where(self.sync_mask, 0.9, 0.3)
        self.update_memory_state(R)

        return {"H_sq": H_sq, "I": I, "R": self.M, "C": C}
