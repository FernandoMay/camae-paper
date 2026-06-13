import numpy as np
from typing import Dict, Tuple

class CognitiveOptimizer:
    def __init__(self,
                 num_subcarriers: int = 64,
                 p_total: float = 1.0,
                 n0: float = 1e-3,
                 alpha: float = 1.0,
                 gamma: float = 0.5,
                 delta: float = 1.0,
                 tau: float = 0.8,
                 max_iter_nr: int = 100,
                 tol_nr: float = 1e-6):

        self.N = num_subcarriers
        self.P_total = float(p_total)
        self.N0 = float(n0)

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.tau = float(tau)

        self.max_iter_nr = max_iter_nr
        self.tol_nr = tol_nr

        self.LOG2_E = 1.4426950408889634

    def compute_utilities(self, C: np.ndarray, R: np.ndarray,
                          I: np.ndarray) -> np.ndarray:
        return self.alpha * C + self.gamma * R - self.delta * I

    def compute_softmax_weights(self, U: np.ndarray) -> np.ndarray:
        max_u = np.max(U)
        tau = max(self.tau, 1e-15)
        exp_u = np.exp((U - max_u) / tau)
        return exp_u / (np.sum(exp_u) + 1e-15)

    def compute_cognitive_entropy(self, w: np.ndarray) -> float:
        eps = 1e-15
        return -float(np.sum(w * np.log2(w + eps)))

    def _water_filling_constraint_eval(self, lambda_val: float,
                                       w: np.ndarray,
                                       Q_eff: np.ndarray) -> Tuple[float, float]:
        lam = max(lambda_val, 1e-15)
        thresholds = w * self.LOG2_E / lam
        inverse_q = 1.0 / (Q_eff + 1e-15)
        active_mask = thresholds > inverse_q

        if not np.any(active_mask):
            return -self.P_total, -1.0

        p_allocated = thresholds[active_mask] - inverse_q[active_mask]
        f_val = np.sum(p_allocated) - self.P_total

        f_prime_val = -np.sum(w[active_mask] * self.LOG2_E) / (lam ** 2)

        return f_val, f_prime_val

    def solve_lagrangian_multiplier(self, w: np.ndarray,
                                    Q_eff: np.ndarray) -> float:
        nonzero_w = w[w > 1e-10]
        lambda_curr = 1.0
        if len(nonzero_w) > 0:
            lambda_curr = (np.min(nonzero_w) * self.LOG2_E) / (self.P_total / self.N + 1e-3)
        lambda_curr = max(lambda_curr, 1e-10)

        for _ in range(self.max_iter_nr):
            f_val, f_prime = self._water_filling_constraint_eval(lambda_curr, w, Q_eff)
            if abs(f_prime) < 1e-12:
                break
            lambda_next = lambda_curr - (f_val / f_prime)
            if lambda_next <= 1e-15:
                lambda_next = lambda_curr * 0.5
            if abs(lambda_next - lambda_curr) < self.tol_nr:
                return lambda_next
            lambda_curr = lambda_next

        return lambda_curr

    def allocate_resources(self, C: np.ndarray, H_sq: np.ndarray,
                           R: np.ndarray, I: np.ndarray) -> Dict[str, np.ndarray]:
        Q_eff = H_sq / (self.N0 + I + 1e-15)

        U = self.compute_utilities(C, R, I)
        w = self.compute_softmax_weights(U)

        entropy = self.compute_cognitive_entropy(w)

        lambda_opt = self.solve_lagrangian_multiplier(w, Q_eff)

        lam = max(lambda_opt, 1e-15)
        p_opt = np.maximum(0.0, (w * self.LOG2_E / lam) - (1.0 / (Q_eff + 1e-15)))

        capacity = np.log2(1.0 + p_opt * Q_eff + 1e-15)

        return {
            "power_vector": p_opt,
            "weights": w,
            "entropy": np.array([entropy], dtype=np.float64),
            "lambda": np.array([lambda_opt], dtype=np.float64),
            "capacity": capacity
        }
