import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

class BaseChannel(ABC):
    def __init__(self, num_subcarriers: int = 64, n0: float = 1e-3, lambda_m: float = 0.8):
        self.N = num_subcarriers
        self.N0 = float(n0)
        self.lambda_m = float(lambda_m)
        self.M = np.zeros(self.N, dtype=np.float64)
        self.t_step = 0

    def generate_awgn(self) -> np.ndarray:
        real_noise = np.random.normal(0, np.sqrt(self.N0 / 2.0), self.N)
        imag_noise = np.random.normal(0, np.sqrt(self.N0 / 2.0), self.N)
        return real_noise + 1j * imag_noise

    def update_memory_state(self, R: np.ndarray) -> np.ndarray:
        self.M = self.lambda_m * self.M + (1.0 - self.lambda_m) * R
        return self.M

    @abstractmethod
    def step(self) -> Dict[str, np.ndarray]:
        pass
