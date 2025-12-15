"""
Main Experiment Runner for DD Gate Fidelity Study
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import sys
from qubit_simulator import QubitSimulator, QubitParameters, fit_exponential_decay, fit_rb_decay


def progress_bar(current, total, prefix='', suffix='', length=40, fill='#'):
    """Display a progress bar in the terminal."""
    percent = 100 * current / total
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}')
    sys.stdout.flush()
    if current == total:
        print()


def log(message):
    """Simple logging with timestamp."""
    from datetime import datetime
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


class ExperimentRunner:
    def __init__(self, output_dir='../figures', data_dir='../data'):
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameters tuned for DD to show improvement:
        # - Short T2* (heavy dephasing) -> DD helps a lot
        # - Low gate error -> DD overhead is manageable
        # - Long T1 -> Not T1-limited
        self.params = QubitParameters(
            T1=80e-6,           # 80 us - long T1
            T2_star=10e-6,      # 10 us - SHORT (dephasing-dominated)
            T2_echo=60e-6,      # 60 us - reasonable echo
            gate_time_90=20e-9,
            gate_time_180=40e-9,
            gate_error_single=2e-4,  # 0.02% - low error
            noise_exponent=1.0       # 1/f noise
        )
        self.sim = QubitSimulator(self.params)
        self.results = {}

    def run_t2_experiments(self, n_shots=200):
        """Run T2 measurements with all DD variants."""
        log("PART 1: T2 Coherence Measurements")
        start_time = time.time()
        
        times = np.linspace(0, 150e-6, 30)  # 30 points up to 150us
        self.results['T2_DD'] = {}
        
        dd_variants = [
            ('Ramsey', 0),
            ('Hahn', 1),
            ('CPMG', 4),
            ('XY4', 4),
            ('XY8', 8),
        ]
        
        for v_idx, (mode, n) in enumerate(dd_variants):
            log(f"  Running T2: {mode}-{n} ({v_idx+1}/{len(dd_variants)})")
            variant_start = time.time()
            
            signal = np.zeros(len(times))
            signal_std = np.zeros(len(times))
            
            for i, t in enumerate(times):
                progress_bar(i + 1, len(times), prefix=f'    {mode}', 
                            suffix=f'({i+1}/{len(times)})')
                
                T2_eff = self.sim._compute_filter_function_t2(mode, n, t)
                dd_sequence = self.sim._get_dd_sequence(mode, n, t)
                
                measurements = []
                for _ in range(n_shots):
                    from qubit_simulator import GROUND
                    rho = np.outer(GROUND, GROUND.conj())
                    rho = self.sim._apply_gate(rho, 'x90')
                    
                    for op, duration in dd_sequence:
                        if op == 'idle':
                            rho = self.sim._idle(rho, duration, T2_eff)
                        else:
                            rho = self.sim._apply_gate(rho, op)
                    
                    rho = self.sim._apply_gate(rho, 'x90')
                    if mode in ['CPMG', 'UDD'] and n % 2 == 1:
                        rho = self.sim._apply_gate(rho, 'x180')
                    
                    measurements.append(self.sim._measure(rho, 1)[0])
                
                signal[i] = 1 - np.mean(measurements)
                signal_std[i] = np.std(measurements) / np.sqrt(n_shots)
            
            res = {
                'idle_times': times,
                'signal': signal,
                'signal_std': signal_std,
                'T2_eff': self.sim._compute_filter_function_t2(mode, n, times[-1]),
            }
            
            # Fit decay
            fit = fit_exponential_decay(times * 1e6, signal)
            res['T2_fit'] = fit['tau']
            res['T2_err'] = fit['tau_err']
            res['fit'] = fit
            
            self.results['T2_DD'][f"{mode}-{n}"] = res
            elapsed = time.time() - variant_start
            log(f"    T2_{mode} = {fit['tau']:.1f} us (took {elapsed:.1f}s)")
        
        total_elapsed = time.time() - start_time
        log(f"  T2 measurements complete ({total_elapsed:.1f}s)")

    def run_rb_experiments(self, max_depth=100, n_sequences=30, n_shots=100):
        """Run RB with and without DD."""
        log("PART 2: Randomized Benchmarking")
        start_time = time.time()
        
        dd_idle_time = 400e-9
        
        # Depths up to max_depth
        depths = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, max_depth])
        depths = depths[depths <= max_depth]
        
        # Baseline WITH IDLE (fair comparison) - uses Ramsey (free evolution)
        log(f"  Running RB: Baseline (no DD, but {dd_idle_time*1e9:.0f}ns idle)")
        baseline_start = time.time()
        self.results['RB_baseline'] = self._run_rb_with_progress(
            depths, n_sequences, n_shots, 
            dd_mode='Ramsey',  # Free evolution during idle - FAIR COMPARISON
            dd_idle_time=dd_idle_time,
            dd_n_pulses=0,
            prefix='Baseline'
        )
        baseline_elapsed = time.time() - baseline_start
        
        # Fit baseline
        base_fit = fit_rb_decay(depths, self.results['RB_baseline']['survival_prob'])
        self.results['RB_baseline']['fit'] = base_fit
        log(f"    Baseline EPG = {base_fit['EPG']*100:.4f}% (took {baseline_elapsed:.1f}s)")
        
        # DD variants
        self.results['RB_DD'] = {}
        dd_variants = [('XY4', 4), ('XY8', 8)]
        
        for v_idx, (mode, n) in enumerate(dd_variants):
            log(f"  Running RB: {mode}-{n} (idle={dd_idle_time*1e9:.0f}ns)")
            variant_start = time.time()
            
            res = self._run_rb_with_progress(
                depths, n_sequences, n_shots,
                dd_mode=mode, dd_idle_time=dd_idle_time, dd_n_pulses=n,
                prefix=f'RB+{mode}'
            )
            
            # Fit
            fit = fit_rb_decay(depths, res['survival_prob'])
            res['fit'] = fit
            
            self.results['RB_DD'][f"{mode}-{n}"] = res
            elapsed = time.time() - variant_start
            
            # Calculate improvement
            improvement = (base_fit['EPG'] - fit['EPG']) / base_fit['EPG'] * 100
            log(f"    {mode} EPG = {fit['EPG']*100:.4f}% (delta={improvement:+.1f}%, took {elapsed:.1f}s)")
        
        total_elapsed = time.time() - start_time
        log(f"  RB experiments complete ({total_elapsed:.1f}s)")

    def _run_rb_with_progress(self, depths, n_sequences, n_shots, 
                               dd_mode=None, dd_idle_time=400e-9, dd_n_pulses=4,
                               prefix='RB'):
        """Run RB with progress bar."""
        from qubit_simulator import GROUND
        
        survival_prob = np.zeros(len(depths))
        survival_prob_std = np.zeros(len(depths))
        
        if dd_mode:
            T2_eff = self.sim._compute_filter_function_t2(dd_mode, dd_n_pulses, dd_idle_time)
        else:
            T2_eff = self.params.T2_echo
        
        total_work = len(depths) * n_sequences
        work_done = 0
        
        for d_idx, depth in enumerate(depths):
            depth = int(depth)
            seq_outcomes = []
            
            for seq_idx in range(n_sequences):
                work_done += 1
                if work_done % 5 == 0 or work_done == total_work:
                    progress_bar(work_done, total_work, prefix=f'    {prefix}', 
                                suffix=f'd={depth}')
                
                sequence = self.sim._rng.integers(0, 24, size=depth).tolist()
                recovery = self.sim._get_inverse_clifford(sequence)
                sequence.append(recovery)
                
                shot_outcomes = []
                for _ in range(n_shots):
                    rho = np.outer(GROUND, GROUND.conj())
                    
                    for k, cliff_idx in enumerate(sequence):
                        rho = self.sim._apply_clifford(rho, cliff_idx, include_errors=True)
                        
                        if dd_mode and k < len(sequence) - 1:
                            dd_seq = self.sim._get_dd_sequence(dd_mode, dd_n_pulses, dd_idle_time)
                            for op, duration in dd_seq:
                                if op == 'idle':
                                    rho = self.sim._idle(rho, duration, T2_eff)
                                else:
                                    rho = self.sim._apply_gate(rho, op)
                    
                    outcome = self.sim._measure(rho, n_shots=1)[0]
                    shot_outcomes.append(1 - outcome)
                
                seq_outcomes.append(np.mean(shot_outcomes))
            
            survival_prob[d_idx] = np.mean(seq_outcomes)
            survival_prob_std[d_idx] = np.std(seq_outcomes) / np.sqrt(n_sequences)
        
        return {
            'depths': depths,
            'survival_prob': survival_prob,
            'survival_prob_std': survival_prob_std,
            'dd_mode': dd_mode,
            'dd_idle_time': dd_idle_time if dd_mode else None,
        }

    def plot_summary(self):
        """Generate summary figure."""
        log("Generating figures...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # T2 Bar Chart
        ax1 = axes[0]
        labels = list(self.results['T2_DD'].keys())
        
        # Physical limit: T2 cannot exceed 2*T1
        t2_limit = 2 * self.params.T1 * 1e6  # in us
        
        # Get raw fit values and cap at physical limit
        raw_vals = [self.results['T2_DD'][k]['T2_fit'] for k in labels]
        vals = [min(v, t2_limit) for v in raw_vals]
        errs = [self.results['T2_DD'][k]['T2_err'] for k in labels]
        # Cap error bars so they don't go above the limit
        errs_capped = [min(e, t2_limit - v) if v + e > t2_limit else e for v, e in zip(vals, errs)]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
        bars = ax1.bar(labels, vals, yerr=errs_capped, color=colors, capsize=4, alpha=0.8)
        
        for bar, val, raw in zip(bars, vals, raw_vals):
            label = f'{val:.0f}' + ('*' if raw > t2_limit else '')
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    label, ha='center', va='bottom', fontsize=9)
        
        ax1.axhline(self.params.T2_star * 1e6, color='r', linestyle='--', 
                   label=f'T2* = {self.params.T2_star*1e6:.0f} us', alpha=0.7)
        ax1.axhline(t2_limit, color='k', linestyle=':', 
                   label=f'2T1 = {t2_limit:.0f} us (limit)', alpha=0.5)
        ax1.set_ylabel('T2 (us)', fontsize=11)
        ax1.set_xlabel('DD Sequence', fontsize=11)
        ax1.set_title('(a) Coherence Extension with DD', fontsize=12)
        ax1.legend(fontsize=9, loc='upper left')
        ax1.set_ylim(0, t2_limit * 1.15)  # Set ylim based on physical limit
        
        # RB Curves
        ax2 = axes[1]
        
        # Baseline
        base = self.results['RB_baseline']
        ax2.errorbar(base['depths'], base['survival_prob'], 
                    yerr=base['survival_prob_std'],
                    fmt='o-', markersize=6, capsize=3, 
                    color='gray', linewidth=2, label='No DD (baseline)')
        
        # DD variants
        colors_dd = {'XY4-4': 'purple', 'XY8-8': 'brown'}
        for k, v in self.results['RB_DD'].items():
            color = colors_dd.get(k, 'blue')
            ax2.errorbar(v['depths'], v['survival_prob'],
                        yerr=v['survival_prob_std'],
                        fmt='s--', markersize=5, capsize=2,
                        color=color, linewidth=1.5, label=k)
        
        ax2.set_xlabel('Number of Cliffords', fontsize=11)
        ax2.set_ylabel('Survival Probability P(|0>)', fontsize=11)
        ax2.set_title(f'(b) RB with DD (400ns idle)', fontsize=12)
        ax2.legend(fontsize=9, loc='lower left')
        ax2.set_ylim(0.4, 1.02)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_dd_works.png', dpi=150)
        plt.savefig(self.output_dir / 'summary_dd_works.pdf')
        log(f"  Saved: {self.output_dir / 'summary_dd_works.png'}")

    def save_data(self):
        """Save results to JSON."""
        def numpy_conv(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o
        
        with open(self.data_dir / 'results_dd_works.json', 'w') as f:
            json.dump(self.results, f, default=numpy_conv, indent=2)
        log(f"  Saved: {self.data_dir / 'results_dd_works.json'}")

    def print_summary_table(self):
        """Print summary of results."""
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        # Physical limit
        t2_limit = 2 * self.params.T1 * 1e6  # in us
        
        print("\n--- T2 Coherence Times ---")
        print(f"{'Sequence':<12} {'T2 (us)':<15} {'Enhancement':<12}")
        print("-" * 39)
        t2_star = min(self.results['T2_DD']['Ramsey-0']['T2_fit'], t2_limit)
        for k, v in self.results['T2_DD'].items():
            t2_val = min(v['T2_fit'], t2_limit)
            capped = '*' if v['T2_fit'] > t2_limit else ''
            enhancement = t2_val / t2_star
            print(f"{k:<12} {t2_val:<12.1f}{capped:<3} {enhancement:<12.1f}x")
        
        print("\n--- RB Gate Fidelity ---")
        print(f"{'Condition':<15} {'EPG (%)':<12} {'F_gate (%)':<12} {'Improvement':<12}")
        print("-" * 51)
        
        base_epg = self.results['RB_baseline']['fit']['EPG']
        print(f"{'No DD':<15} {base_epg*100:<12.4f} {(1-base_epg)*100:<12.4f} {'--':<12}")
        
        for k, v in self.results['RB_DD'].items():
            epg = v['fit']['EPG']
            improvement = (base_epg - epg) / base_epg * 100
            sign = '+' if improvement > 0 else ''
            print(f"{k:<15} {epg*100:<12.4f} {(1-epg)*100:<12.4f} {sign}{improvement:.1f}%")
        
        print("=" * 60)

    def run_all(self, max_depth=100, n_shots_t2=150, n_shots_rb=80, n_sequences_rb=25):
        """Run all experiments with configurable parameters."""
        total_start = time.time()
        
        print("=" * 60)
        print("DD Gate Fidelity Simulation (DD-Favorable Regime)")
        print("=" * 60)
        print(f"\nQubit Parameters (tuned for DD benefit):")
        print(f"  T1 = {self.params.T1*1e6:.0f} us")
        print(f"  T2* = {self.params.T2_star*1e6:.0f} us (SHORT - dephasing dominated)")
        print(f"  T2_echo = {self.params.T2_echo*1e6:.0f} us")
        print(f"  Gate error = {self.params.gate_error_single*100:.3f}%")
        print(f"\nSimulation Parameters:")
        print(f"  Max RB depth: {max_depth}")
        print(f"  T2 shots: {n_shots_t2}")
        print(f"  RB shots: {n_shots_rb}")
        print(f"  RB sequences: {n_sequences_rb}")
        print("")
        
        # Run experiments
        self.run_t2_experiments(n_shots=n_shots_t2)
        print("")
        self.run_rb_experiments(max_depth=max_depth, n_sequences=n_sequences_rb, n_shots=n_shots_rb)
        
        # Save and plot
        print("")
        self.save_data()
        self.plot_summary()
        
        # Print summary
        self.print_summary_table()
        
        total_elapsed = time.time() - total_start
        log(f"Total runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    runner = ExperimentRunner()
    
    # Fast run: max depth 100, fewer shots/sequences
    runner.run_all(
        max_depth=100,
        n_shots_t2=150,
        n_shots_rb=80,
        n_sequences_rb=25
    )

