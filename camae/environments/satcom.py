import numpy as np
from typing import Dict
from .base_channel import BaseChannel

class SatcomChannel(BaseChannel):
    def __init__(self, num_subcarriers: int = 64, n0: float = 1e-3,
                 rho: float = 0.96, sigma_d: float = 12.0):
        super().__init__(num_subcarriers, n0)
        self.rho = float(rho)
        self.sigma_d = float(sigma_d)
        self.f_d = np.zeros(self.N, dtype=np.float64)
        self.phi = np.random.uniform(0, 2 * np.pi, self.N)

    def step(self) -> Dict[str, np.ndarray]:
        self.t_step += 1

        epsilon = np.random.normal(0, self.sigma_d, self.N)
        self.f_d = self.rho * self.f_d + np.sqrt(1.0 - self.rho ** 2) * epsilon

        self.phi = (self.phi + 2.0 * np.pi * self.f_d * 1e-3) % (2 * np.pi)

        amplitude = np.sqrt(np.random.exponential(1.0, self.N))
        H = amplitude * np.exp(1j * self.phi) + self.generate_awgn()
        H_sq = np.abs(H) ** 2

        I = np.abs(np.random.normal(0, 1e-4, self.N)) ** 2
        C = np.ones(self.N, dtype=np.float64)

        R = 1.0 / (1.0 + np.abs(self.f_d) + 1e-15)
        self.update_memory_state(R)

        return {"H_sq": H_sq, "I": I, "R": self.M, "C": C}
