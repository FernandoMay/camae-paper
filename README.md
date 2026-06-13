# CAMAE: Cognitive Adaptive Multicarrier Allocation Engine

## Overview

CAMAE is a unified spectral-cognitive coordination framework that extends classical multicarrier modulation into a context-aware adaptive orchestration architecture for heterogeneous distributed systems operating under Partially Observable Dynamic Environments (PODEs).

## Repository Structure

```
camae_paper/
├── camae/                          # Python package
│   ├── core/
│   │   ├── optimizer.py            # CognitiveOptimizer (Softmax + Newton-Raphson)
│   │   └── engine.py               # CAMAE main loop
│   ├── environments/
│   │   ├── base_channel.py         # Abstract stochastic channel
│   │   ├── satcom.py               # LEO Satellite Doppler channel
│   │   ├── radar.py                # Cognitive Radar jamming channel
│   │   └── neuro.py                # Neuro-inspired oscillatory drift channel
│   └── utils/
│       ├── metrics.py              # Metrics collection (ESI, Psi, etc.)
│       └── visualization.py        # IEEE-grade figure generation
├── run_montecarlo.py               # Monte Carlo simulation engine
├── requirements.txt                # Python dependencies
├── results/                        # CSV output data
├── output_figures/                 # Generated publication figures
├── manuscript/                     # LaTeX paper source
│   ├── camae_ieee_manuscript.tex   # Complete IEEEtran paper
│   └── references.bib             # BibTeX references
└── README.md
```

## Requirements

- Python 3.8+
- numpy, pandas, matplotlib, scipy

Install: `pip install -r requirements.txt`

## Running the Simulation

```bash
python run_montecarlo.py
```

This executes Monte Carlo simulations across 3 scenarios (Satcom, Radar, Neuro) with 3 methods (CAMAE, SUA, CWF), outputs CSVs, and generates publication-quality figures.

## Paper

The complete IEEEtran LaTeX manuscript is in `manuscript/camae_ieee_manuscript.tex`.

## Key Metrics

- **H(t)**: Cognitive Allocation Entropy
- **ESI**: Entropy Stability Index
- **Psi(t)**: Dynamic Spectral Resilience Function

## Citation

```bibtex
@article{fernandez2026camae,
  title={Cognitive Adaptive Multicarrier Allocation Engine (CAMAE):
         Entropy-Aware Spectral Coordination Under Partially Observable Dynamic Environments},
  author={Fern{\'a}ndez, F. M. Y.},
  journal={IEEE Systems Journal},
  year={2026}
}
```
