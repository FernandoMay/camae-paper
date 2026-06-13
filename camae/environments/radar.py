import numpy as np
from typing import Dict
from .base_channel import BaseChannel

class RadarChannel(BaseChannel):
    def __init__(self, num_subcarriers: int = 64, n0: float = 1e-3,
                 pareto_alpha: float = 1.2, pareto_scale: float = 2.0,
                 jam_prob: float = 0.10):
        super().__init__(num_subcarriers, n0)
        self.alpha_p = float(pareto_alpha)
        self.x_m = float(pareto_scale)
        self.jam_prob = float(jam_prob)

    def step(self) -> Dict[str, np.ndarray]:
        self.t_step += 1

        H_sq = np.ones(self.N, dtype=np.float64) * 0.9

        I = np.zeros(self.N, dtype=np.float64)
        if np.random.rand() < self.jam_prob:
            block_start = np.random.randint(0, max(1, self.N - 8))
            jamming_intensity = (np.random.pareto(self.alpha_p) + 1.0) * self.x_m
            block_end = min(block_start + 8, self.N)
            I[block_start:block_end] = jamming_intensity

        C = np.ones(self.N, dtype=np.float64) * 0.2
        if np.any(I > 0.0):
            C[I > 0.0] = 2.5

        R = np.ones(self.N, dtype=np.float64) * 0.5
        self.update_memory_state(R)

        return {"H_sq": H_sq, "I": I, "R": self.M, "C": C}
