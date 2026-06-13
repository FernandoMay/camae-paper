import numpy as np
import pandas as pd
from typing import Dict

class MetricsCollector:
    def __init__(self, n0: float = 1e-3):
        self.N0 = float(n0)
        self.records = []

    def record(self, timestep: int, env_name: str, allocation: Dict[str, np.ndarray],
               env_state: Dict[str, np.ndarray], previous_entropy: float):
        w = allocation["weights"]
        entropy = allocation["entropy"][0]
        capacity = allocation["capacity"]
        p_vector = allocation["power_vector"]

        total_capacity = float(np.sum(capacity))
        Q_eff = env_state["H_sq"] / (self.N0 + env_state["I"] + 1e-15)
        psi_t = float(np.sum(w * Q_eff) / (1.0 + entropy + 1e-15))

        esi_instantaneous = float(abs(entropy - previous_entropy))

        active_channels = int(np.sum(p_vector > 1e-4))

        self.records.append({
            "Timestep": timestep,
            "Environment": env_name,
            "Total_Capacity": total_capacity,
            "Cognitive_Entropy": entropy,
            "Spectral_Resilience_Psi": psi_t,
            "ESI_Delta": esi_instantaneous,
            "Mean_Interference": float(np.mean(env_state["I"])),
            "Active_Channels": active_channels
        })

        return entropy

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def compute_esi(self, df: pd.DataFrame) -> float:
        return float(df["ESI_Delta"].mean())

    def compute_aggregate_efficiency(self, df: pd.DataFrame, p_total: float = 1.0) -> float:
        return float(df["Total_Capacity"].mean()) / (p_total + 1e-15)

    def reset(self):
        self.records = []
