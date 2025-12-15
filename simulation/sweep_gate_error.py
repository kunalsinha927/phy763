"""
Sweep gate error to find DD crossover boundary.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import sys

from qubit_simulator import QubitSimulator, QubitParameters, fit_rb_decay


def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def progress_bar(current, total, prefix='', suffix='', length=40):
    percent = 100 * current / total
    filled = int(length * current // total)
    bar = '#' * filled + '-' * (length - filled)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}')
    sys.stdout.flush()
    if current == total:
        print()


def run_sweep():
    """Sweep gate error to find crossover point."""
    
    output_dir = Path('../figures')
    data_dir = Path('../data')
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Gate errors to sweep (log-spaced)
    gate_errors = np.logspace(-5, -2, 20)  # 1e-5 to 1e-2
    
    # Fixed parameters
    dd_idle_time = 400e-9  # 400 ns
    rb_depths = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    n_sequences = 100  # Increased for better statistics
    n_shots = 250      # Increased for better statistics
    
    results = {
        'gate_errors': gate_errors.tolist(),
        'XY4': {'EPG': [], 'EPG_err': []},
        'XY8': {'EPG': [], 'EPG_err': []},
        'baseline': {'EPG': [], 'EPG_err': []},
    }
    
    log("Starting gate error sweep")
    log(f"  Gate errors: {gate_errors[0]:.1e} to {gate_errors[-1]:.1e}")
    log(f"  n_sequences: {n_sequences}, n_shots: {n_shots}")
    
    total_start = time.time()
    
    for i, err in enumerate(gate_errors):
        progress_bar(i + 1, len(gate_errors), prefix='Sweep', suffix=f'err={err:.1e}')
        
        # Create simulator with this gate error
        params = QubitParameters(
            T1=80e-6,
            T2_star=10e-6,
            T2_echo=60e-6,
            gate_time_90=20e-9,
            gate_time_180=40e-9,
            gate_error_single=err,
            noise_exponent=1.0
        )
        sim = QubitSimulator(params)
        
        # Baseline (Ramsey during idle - fair comparison)
        base_res = sim.simulate_rb(rb_depths, n_sequences, n_shots,
                                    dd_mode='Ramsey', dd_idle_time=dd_idle_time, dd_n_pulses=0)
        base_fit = fit_rb_decay(rb_depths, base_res['survival_prob'])
        results['baseline']['EPG'].append(base_fit['EPG'])
        results['baseline']['EPG_err'].append(base_fit.get('EPG_err', 0))
        
        # XY-4
        xy4_res = sim.simulate_rb(rb_depths, n_sequences, n_shots,
                                   dd_mode='XY4', dd_idle_time=dd_idle_time, dd_n_pulses=4)
        xy4_fit = fit_rb_decay(rb_depths, xy4_res['survival_prob'])
        results['XY4']['EPG'].append(xy4_fit['EPG'])
        results['XY4']['EPG_err'].append(xy4_fit.get('EPG_err', 0))
        
        # XY-8
        xy8_res = sim.simulate_rb(rb_depths, n_sequences, n_shots,
                                   dd_mode='XY8', dd_idle_time=dd_idle_time, dd_n_pulses=8)
        xy8_fit = fit_rb_decay(rb_depths, xy8_res['survival_prob'])
        results['XY8']['EPG'].append(xy8_fit['EPG'])
        results['XY8']['EPG_err'].append(xy8_fit.get('EPG_err', 0))
    
    elapsed = time.time() - total_start
    log(f"Sweep complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    # Save data
    with open(data_dir / 'gate_error_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)
    log(f"Saved: {data_dir / 'gate_error_sweep.json'}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: EPG vs gate error
    ax1 = axes[0]
    ax1.loglog(gate_errors, results['baseline']['EPG'], 'o-', label='No DD (baseline)', color='gray')
    ax1.loglog(gate_errors, results['XY4']['EPG'], 's--', label='XY-4', color='purple')
    ax1.loglog(gate_errors, results['XY8']['EPG'], '^--', label='XY-8', color='brown')
    ax1.set_xlabel('Single-Qubit Gate Error', fontsize=11)
    ax1.set_ylabel('Error Per Gate (EPG)', fontsize=11)
    ax1.set_title('EPG vs Gate Error', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Gain (baseline / DD)
    ax2 = axes[1]
    gain_xy4 = np.array(results['baseline']['EPG']) / np.array(results['XY4']['EPG'])
    gain_xy8 = np.array(results['baseline']['EPG']) / np.array(results['XY8']['EPG'])
    
    ax2.semilogx(gate_errors, gain_xy4, 's-', label='XY-4', color='purple')
    ax2.semilogx(gate_errors, gain_xy8, '^-', label='XY-8', color='brown')
    ax2.axhline(1, color='k', linestyle='--', alpha=0.5, label='Break-even')
    ax2.fill_between(gate_errors, 1, max(max(gain_xy4), max(gain_xy8)) * 1.1, 
                     alpha=0.1, color='green', label='DD beneficial')
    ax2.fill_between(gate_errors, 0, 1, alpha=0.1, color='red', label='DD harmful')
    
    ax2.set_xlabel('Single-Qubit Gate Error', fontsize=11)
    ax2.set_ylabel('Gain (EPG_baseline / EPG_DD)', fontsize=11)
    ax2.set_title('DD Crossover Analysis', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim(0, max(max(gain_xy4), max(gain_xy8)) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dd_crossover_analysis.png', dpi=150)
    plt.savefig(output_dir / 'dd_crossover_analysis.pdf')
    log(f"Saved: {output_dir / 'dd_crossover_analysis.png'}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CROSSOVER ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Find crossover points
    for name, gains in [('XY-4', gain_xy4), ('XY-8', gain_xy8)]:
        crossover_idx = np.where(gains < 1)[0]
        if len(crossover_idx) > 0:
            cross_err = gate_errors[crossover_idx[0]]
            print(f"{name}: Crossover at gate error ~ {cross_err:.2e} ({cross_err*100:.3f}%)")
        else:
            print(f"{name}: DD beneficial across entire range")
    
    print("=" * 60)


if __name__ == "__main__":
    run_sweep()

