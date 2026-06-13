import numpy as np
from typing import Dict, Optional
from .optimizer import CognitiveOptimizer

class CAMAE:
    def __init__(self, optimizer: Optional[CognitiveOptimizer] = None, **kwargs):
        self.optimizer = optimizer if optimizer else CognitiveOptimizer(**kwargs)
        self.state_history = []

    def step(self, C: np.ndarray, H_sq: np.ndarray,
             R: np.ndarray, I: np.ndarray) -> Dict[str, np.ndarray]:
        result = self.optimizer.allocate_resources(C, H_sq, R, I)
        self.state_history.append(result)
        return result

    def reset(self):
        self.state_history = []
