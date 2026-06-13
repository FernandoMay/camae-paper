import numpy as np
import pandas as pd
import os, sys, time
from typing import Dict, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camae.core.optimizer import CognitiveOptimizer
from camae.core.engine import CAMAE
from camae.environments.satcom import SatcomChannel
from camae.environments.radar import RadarChannel
from camae.environments.neuro import NeuroChannel
from camae.utils.metrics import MetricsCollector
from camae.utils.visualization import CognitiveVisualizer


def run_camae(env_fn: Callable, N: int, P_total: float, N0: float,
              T: int) -> pd.DataFrame:
    opt = CognitiveOptimizer(
        num_subcarriers=N, p_total=P_total, n0=N0,
        alpha=1.0, gamma=0.5, delta=1.0, tau=1.0
    )
    camae = CAMAE(optimizer=opt)
    collector = MetricsCollector(n0=N0)
    env = env_fn()
    w_init = np.ones(N) / N
    prev_entropy = opt.compute_cognitive_entropy(w_init)
    for t in range(T):
        s = env.step()
        a = camae.step(C=s["C"], H_sq=s["H_sq"], R=s["R"], I=s["I"])
        prev_entropy = collector.record(t, "CAMAE", a, s, prev_entropy)
    return collector.to_dataframe()


def run_sua(env_fn: Callable, N: int, P_total: float, N0: float,
            T: int) -> pd.DataFrame:
    records = []
    env = env_fn()
    p_k = P_total / max(N, 1)
    for t in range(T):
        s = env.step()
        cap = np.sum(np.log2(1.0 + p_k * s["H_sq"] / (N0 + s["I"] + 1e-15)))
        w = np.ones(N) / N
        entropy = -np.sum(w * np.log2(w + 1e-15))
        Q_eff = s["H_sq"] / (N0 + s["I"] + 1e-15)
        psi = np.sum(w * Q_eff) / (1.0 + entropy + 1e-15)
        records.append({
            "Timestep": t, "Environment": "SUA",
            "Total_Capacity": float(cap),
            "Cognitive_Entropy": float(entropy),
            "Spectral_Resilience_Psi": float(psi),
            "ESI_Delta": 0.0,
            "Mean_Interference": float(np.mean(s["I"])),
            "Active_Channels": N
        })
    return pd.DataFrame(records)


def run_cwf(env_fn: Callable, N: int, P_total: float, N0: float,
            T: int) -> pd.DataFrame:
    records = []
    env = env_fn()
    LOG2_E = 1.4426950408889634

    for t in range(T):
        s = env.step()
        Q = s["H_sq"] / (N0 + s["I"] + 1e-15)

        lam_low, lam_high = 1e-10, 1e10
        for _ in range(80):
            lam = (lam_low + lam_high) / 2
            p = np.maximum(0, LOG2_E / lam - 1.0 / (Q + 1e-15))
            total_p = np.sum(p)
            if abs(total_p - P_total) < 1e-10:
                break
            if total_p > P_total:
                lam_low = lam
            else:
                lam_high = lam

        cap = np.sum(np.log2(1.0 + p * Q + 1e-15))
        w = np.ones(N) / N
        entropy = -np.sum(w * np.log2(w + 1e-15))
        psi = np.sum(w * Q) / (1.0 + entropy + 1e-15)
        active = int(np.sum(p > 1e-4))
        records.append({
            "Timestep": t, "Environment": "CWF",
            "Total_Capacity": float(cap),
            "Cognitive_Entropy": float(entropy),
            "Spectral_Resilience_Psi": float(psi),
            "ESI_Delta": 0.0,
            "Mean_Interference": float(np.mean(s["I"])),
            "Active_Channels": active
        })
    return pd.DataFrame(records)


class MonteCarloEngine:
    def __init__(self, num_subcarriers: int = 64, timesteps: int = 1000,
                 p_total: float = 1.0, seed: int = 42):
        self.N = num_subcarriers
        self.T = timesteps
        self.P_total = p_total
        self.N0 = 1e-3

    def run_all_methods(self, env_name: str, factory: Callable) -> Dict[str, pd.DataFrame]:
        runners = [
            ("CAMAE", lambda: run_camae(factory, self.N, self.P_total, self.N0, self.T)),
            ("SUA", lambda: run_sua(factory, self.N, self.P_total, self.N0, self.T)),
            ("CWF", lambda: run_cwf(factory, self.N, self.P_total, self.N0, self.T)),
        ]
        results = {}
        for mname, runner in runners:
            np.random.seed(42)
            t0 = time.time()
            df = runner()
            elapsed = time.time() - t0
            esi = df["ESI_Delta"].mean()
            cap = df["Total_Capacity"].mean()
            psi = df["Spectral_Resilience_Psi"].mean()
            ent = df["Cognitive_Entropy"].mean()
            act = df["Active_Channels"].mean()
            results[f"{env_name}_{mname}"] = df
            print(f"  [{mname:5s}] H={ent:.4f} ESI={esi:.5f} "
                  f"Cap={cap:.2f} Psi={psi:.2f} Active={act:.1f} [{elapsed:.1f}s]")
        return results

    def execute_all(self) -> Dict[str, pd.DataFrame]:
        print(f"[*] CAMAE Monte Carlo (N={self.N}, T={self.T}, P_total={self.P_total})")
        print()

        scenarios = [
            ("Satcom_LEO_Doppler",
             lambda: SatcomChannel(self.N, self.N0, rho=0.96, sigma_d=12.0)),
            ("Cognitive_Radar_Jamming",
             lambda: RadarChannel(self.N, self.N0, pareto_alpha=1.2,
                                  pareto_scale=2.0, jam_prob=0.10)),
            ("Neuro_Oscillatory_Drift",
             lambda: NeuroChannel(self.N, self.N0, cauchy_gamma=0.04)),
        ]

        results = {}
        for env_name, factory in scenarios:
            print(f"[->] {env_name}:")
            results.update(self.run_all_methods(env_name, factory))
            print()
        return results


def build_paper_table(results: Dict[str, pd.DataFrame],
                      scenarios: list, methods: list) -> pd.DataFrame:
    rows = []
    for senv in scenarios:
        for method in methods:
            key = f"{senv}_{method}"
            if key not in results:
                continue
            df = results[key]
            rows.append({
                "Scenario": senv.replace('_', ' '),
                "Method": method,
                "H(t)": f"{df['Cognitive_Entropy'].mean():.4f}",
                "ESI": f"{df['ESI_Delta'].mean():.5f}",
                "Capacity": f"{df['Total_Capacity'].mean():.2f}",
                "Psi(t)": f"{df['Spectral_Resilience_Psi'].mean():.2f}",
                "Active Ch.": f"{df['Active_Channels'].mean():.1f}"
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    engine = MonteCarloEngine(num_subcarriers=64, timesteps=1000, seed=42)
    data_suite = engine.execute_all()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    rd = os.path.join(base_dir, "results")
    fd = os.path.join(base_dir, "output_figures")
    os.makedirs(rd, exist_ok=True)
    os.makedirs(fd, exist_ok=True)

    for name, df in data_suite.items():
        df.to_csv(os.path.join(rd, f"{name}.csv"), index=False)

    vis = CognitiveVisualizer(output_dir=fd)
    vis.generate_all_plots(data_suite)

    scenarios = ["Satcom_LEO_Doppler", "Cognitive_Radar_Jamming", "Neuro_Oscillatory_Drift"]
    methods = ["CAMAE", "SUA", "CWF"]
    pt = build_paper_table(data_suite, scenarios, methods)

    print("=" * 95)
    print("  TABLE I: Comparative Performance Metrics (CAMAE vs Baselines)")
    print("=" * 95)
    print(pt.to_string(index=False))
    pt.to_csv(os.path.join(rd, "paper_table_results.csv"), index=False)
