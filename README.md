# Risk-Aware O-RAN based Integrated TN-NTN Implementation

A Python-based simulation platform modeling integrated Terrestrial and
Non-Terrestrial Networks (TN-NTN) for reliable communications.
This implements risk-disjoint backup path scheduling alongside a multi-connectivity (MC) benchmark under heterogeneous network failure conditions, including catastrophic GBS failures.

This project is based on and extension of paper "A. Manzoor, M. Ozger, and C. Cavdar, “Risk-Aware Backup Path Allocation in O-RAN Based Integrated Terrestrial and Non-Terrestrial Networks,” in Proc. 21st IEEE International Conference on the Design of Reliable Communication Networks (DRCN), 2025. doi:10.1109/DRCN65040.2025.11046130"

## How to Run

### 1. Dependencies

Ensure Python 3.9+ and the following packages (listed in `requirements.txt`) are installed:
* `numpy`
* `scipy`
* `pandas`
* `matplotlib`

### 2. Run Simulation

``` bash
python main_simulation.py
```

## Key System Parameters

Configured in `src/constants.py`:
| Parameter | Value | Description |
|---|---|---|
| Carrier Frequency | 2.0 GHz | S-band carrier frequency |
| System Bandwidth | 10.0 MHz | Total transmission bandwidth (50 RBs) |
| RB Bandwidth | 180 kHz | Single RB bandwidth (12 subcarriers × 15 kHz) |
| Terrestrial Topology | 7 GBS | Localized cell deployment |
| Non-Terrestrial Nodes | HAP and LEO  | For global network coverage
| Frequency Reuse | FR-1 | Universal frequency reuse with co-channel interference |
| Latency Threshold | 30.0 ms | End-to-end latency deadline |

## Implementation Enhancements (Beyond Paper Baseline)

*   **Catastrophic Multi-GBS Failures:** Extends beyond single-node outage assumptions by modeling simultaneous, multi-GBS failure  (3 adjacent GBS outages) to evaluate large-scale disaster resilience.
*   **Spatially Correlated Fading:** Models spatial correlation matrices via Bessel functions with Rician fading for terrestrial and NTN links.
*   **3D Antenna Patterns:** Simulates 3-sector horizontal/vertical radiation patterns with 8-element array factor beamforming.
*   **Universal FRF-1 Co-Channel Interference:** Dynamically aggregates cross-sector and neighboring cell interference.
*   **Latency & ARQ Verification:** Enforces a 30 ms delay budget factoring in propagation, transmission, queueing, and retransmission scaling.

## Repository Structure

``` text
├── main_simulation.py       # Main simulation entry point
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── src/
    ├── constants.py         # System parameters
    ├── system_model.py      # Path loss, 3D antenna patterns, fading, and SINR
    ├── topology.py          # Node coordinate generation and hexagonal cell plotting
    ├── link_evaluator.py    # Channel calculations and candidate link budgets
    ├── primary_path.py      # Primary link association 
    ├── mc_scheme.py         # Dual-connectivity baseline scheme
    ├── risk_profiles.py     # Disjoint failure injection 
    ├── scheduler_final.py   # Risk-disjoint grouping and dynamic backup allocation
    ├── parameter_sweep.py   # Contour dimensioning and MC vs. BC benchmarks
    └── visualization.py     # 3D topology, SINR, and evaluation plots
```
