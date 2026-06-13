import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import Dict

class CognitiveVisualizer:
    def __init__(self, output_dir: str = "output_figures"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.titlesize'] = 11
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 8
        plt.rcParams['figure.dpi'] = 300

    def _filter_camae(self, data_suite):
        return {k: v for k, v in data_suite.items() if "CAMAE" in k}

    def _scenario_name(self, key):
        mapping = {
            "Satcom_LEO_Doppler": "LEO Satellite Doppler",
            "Cognitive_Radar_Jamming": "Cognitive Radar Jamming",
            "Neuro_Oscillatory_Drift": "Neuro-Inspired Oscillatory Drift"
        }
        for sk, label in mapping.items():
            if sk in key:
                return label
        return key.replace('_', ' ')

    # --- FIGURE 1: Cognitive Entropy Evolution ---
    def plot_cognitive_entropy_evolution(self, data_suite):
        camae = self._filter_camae(data_suite)
        fig, axes = plt.subplots(len(camae), 1, figsize=(6.5, 7.5), sharex=True)
        colors = ['#1f77b4', '#d62728', '#2ca02c']
        if len(camae) == 1:
            axes = [axes]
        for idx, (name, df) in enumerate(camae.items()):
            ax = axes[idx]
            t = df["Timestep"].values
            h = df["Cognitive_Entropy"].values
            ax.plot(t, h, color=colors[idx % len(colors)], linewidth=0.7,
                    label=self._scenario_name(name))
            ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.6)
            ax.set_ylabel(r"$\mathcal{H}(t)$ [bits]")
            ax.legend(loc="upper right", frameon=True, facecolor='white',
                      edgecolor='none', fontsize=8)
            if "Radar" in name:
                ji = np.where(df["Mean_Interference"].values > 0.1)[0]
                if len(ji) > 0:
                    for j in ji[::20]:
                        ax.axvline(x=j, color='red', linestyle=':', alpha=0.15)
            if "Neuro" in name:
                hd = np.where(df["ESI_Delta"].values > df["ESI_Delta"].mean() * 2)[0]
                for step in hd[::10]:
                    ax.axvline(x=step, color='gray', linestyle=':', alpha=0.15)
        axes[-1].set_xlabel("Discrete Timestep ($t$)")
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "fig1_cognitive_entropy.pdf"),
                    bbox_inches='tight')
        fig.savefig(os.path.join(self.output_dir, "fig1_cognitive_entropy.png"),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)

    # --- FIGURE 2: Spectral Resilience Psi(t) ---
    def plot_spectral_resilience(self, data_suite):
        camae = self._filter_camae(data_suite)
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        styles = [('-', '#1f77b4'), ('--', '#d62728'), ('-.', '#2ca02c')]
        for idx, (name, df) in enumerate(camae.items()):
            t = df["Timestep"].values
            psi = df["Spectral_Resilience_Psi"].values
            ax.plot(t, psi, styles[idx][0], color=styles[idx][1], linewidth=0.7,
                    label=self._scenario_name(name))
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.6)
        ax.set_xlabel("Discrete Timestep ($t$)")
        ax.set_ylabel(r"Spectral Resilience $\Psi(t)$")
        ax.legend(loc="best", frameon=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "fig2_spectral_resilience.pdf"),
                    bbox_inches='tight')
        fig.savefig(os.path.join(self.output_dir, "fig2_spectral_resilience.png"),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)

    # --- FIGURE 3: Phase Space ---
    def plot_entropy_capacity_phase_space(self, data_suite):
        camae = self._filter_camae(data_suite)
        fig, axes = plt.subplots(1, len(camae), figsize=(6.5, 2.5), sharey=True)
        if len(camae) == 1:
            axes = [axes]
        labels_map = {
            "Satcom_LEO_Doppler": "Satcom",
            "Cognitive_Radar_Jamming": "Radar",
            "Neuro_Oscillatory_Drift": "Neuro"
        }
        max_t = 0
        for _, df in camae.items():
            max_t = max(max_t, df["Timestep"].max())
        for idx, (name, df) in enumerate(camae.items()):
            ax = axes[idx]
            h = df["Cognitive_Entropy"].values
            c = df["Total_Capacity"].values
            ax.plot(h, c, color='gray', alpha=0.2, linewidth=0.4)
            ax.scatter(h, c, c=df["Timestep"].values, cmap="viridis",
                       s=2, alpha=0.7, edgecolors='none')
            ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.6)
            ax.set_xlabel(r"$\mathcal{H}(t)$ [bits]")
            ax.set_title(labels_map.get(name, name.split('_')[0]), fontsize=9)
            if idx == 0:
                ax.set_ylabel("Capacity [bps/Hz]")
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        norm = plt.Normalize(vmin=0, vmax=max_t)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                     cax=cbar_ax, label="Time ($t$)")
        fig.subplots_adjust(wspace=0.15)
        fig.savefig(os.path.join(self.output_dir, "fig3_phase_space.pdf"),
                    bbox_inches='tight')
        fig.savefig(os.path.join(self.output_dir, "fig3_phase_space.png"),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)

    # --- FIGURE 4: Radar ESI Detail ---
    def plot_esi_radar_detail(self, data_suite):
        radar_key = None
        for k in data_suite:
            if "Radar" in k and "CAMAE" in k:
                radar_key = k
                break
        if not radar_key:
            return
        df = data_suite[radar_key]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True)

        ax1.plot(df["Timestep"], df["Total_Capacity"], color='#1f77b4', linewidth=0.8)
        ax1.set_ylabel("Capacity [bps/Hz]")
        ax1.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

        ax2.plot(df["Timestep"], df["ESI_Delta"], color='#d62728', linewidth=0.5, alpha=0.7)
        ax2.fill_between(df["Timestep"], 0, df["ESI_Delta"], color='#d62728', alpha=0.15)
        ax2.set_xlabel("Discrete Timestep ($t$)")
        ax2.set_ylabel("|ESI| Instantaneous")
        ax2.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "fig4_esi_radar_detail.pdf"),
                    bbox_inches='tight')
        fig.savefig(os.path.join(self.output_dir, "fig4_esi_radar_detail.png"),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)

    # --- FIGURE 5: Capacity Comparison ---
    def plot_capacity_comparison(self, data_suite):
        scenarios = ["Satcom_LEO_Doppler", "Cognitive_Radar_Jamming", "Neuro_Oscillatory_Drift"]
        titles = ["LEO Satellite Doppler", "Cognitive Radar Jamming", "Neuro-Inspired Drift"]
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)
        cmap = {'CAMAE': '#1f77b4', 'SUA': '#d62728', 'CWF': '#2ca02c'}
        for idx, (senv, stitle) in enumerate(zip(scenarios, titles)):
            ax = axes[idx]
            for method in ['CAMAE', 'SUA', 'CWF']:
                key = f"{senv}_{method}"
                if key not in data_suite:
                    continue
                df = data_suite[key]
                cap = df["Total_Capacity"].values
                bins = np.linspace(min(cap), max(cap) + 1e-6, 25)
                ax.hist(cap, bins=bins, alpha=0.5, label=method,
                        color=cmap[method], density=True)
            ax.set_xlabel("Capacity [bps/Hz]")
            ax.set_title(stitle, fontsize=9)
            ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)
            if idx == 0:
                ax.set_ylabel("Density")
            if idx == 2:
                ax.legend(fontsize=7, frameon=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "fig5_capacity_comparison.pdf"),
                    bbox_inches='tight')
        fig.savefig(os.path.join(self.output_dir, "fig5_capacity_comparison.png"),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)

    def generate_all_plots(self, data_suite):
        print("[*] Generating publication-grade figures...")
        self.plot_cognitive_entropy_evolution(data_suite)
        self.plot_spectral_resilience(data_suite)
        self.plot_entropy_capacity_phase_space(data_suite)
        self.plot_esi_radar_detail(data_suite)
        self.plot_capacity_comparison(data_suite)
        print(f"[+] Figures exported to '{self.output_dir}/'")
