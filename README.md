# DD Gate Fidelity

Simulation and hardware code for studying dynamical decoupling (DD) in superconducting transmon qubits. Part of PHY763 final project at UW-Madison.

## What this does

We compare different DD sequences (Hahn Echo, CPMG, XY-4, XY-8) to see which ones actually improve gate fidelity when inserted between Clifford gates in randomized benchmarking.

The key finding: DD helps when dephasing dominates, but hurts when pulse errors add up. There's a crossover point.

## Results

| Condition | EPG (%) | vs Baseline |
|-----------|---------|-------------|
| No DD | 0.388 | -- |
| XY-4 | 0.212 | 1.83x better |
| XY-8 | 0.155 | 2.50x better |

Crossover points (where DD stops helping):
- XY-4: gate error ~0.20%
- XY-8: gate error ~0.13%

## Repo layout

```
simulation/          Density matrix simulator + experiments
  qubit_simulator.py     Core simulator with noise channels
  run_experiments.py     T2 and RB experiments
  sweep_gate_error.py    Crossover analysis

hardware/            QUA code for OPX
  DD_two_scenarios.py    Main experiment script
  README.md              Setup notes

data/                JSON output from runs
paper/               LaTeX source + figures
```

## Running the simulation

```bash
pip install numpy scipy matplotlib

cd simulation
python run_experiments.py      # T2 + RB 
python sweep_gate_error.py     # crossover sweep
```

Takes about 25 min on an M3 Pro.

## Hardware (OPX)

Needs a calibrated qubit with x90/x180/y90/y180 pulses. See `hardware/README.md` for details.

```bash
cd hardware
python DD_two_scenarios.py
```

## Parameters used

| Param | Value |
|-------|-------|
| T1 | 80 us |
| T2* | 10 us |
| T2_echo | 60 us |
| Gate error | 0.02% |
| DD idle time | 400 ns |

