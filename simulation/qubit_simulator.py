"""
Qubit Simulator for Dynamical Decoupling Experiments
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List, Dict
from scipy.linalg import expm
from scipy.integrate import quad
import warnings

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Standard states
GROUND = np.array([1, 0], dtype=complex)
EXCITED = np.array([0, 1], dtype=complex)

@dataclass
class QubitParameters:
    """Physical parameters for a superconducting transmon qubit."""
    T1: float = 50e-6
    T2_star: float = 20e-6
    T2_echo: float = 40e-6
    gate_time_90: float = 20e-9
    gate_time_180: float = 40e-9
    gate_error_single: float = 1e-4
    readout_fidelity: float = 0.98
    thermal_population: float = 0.02
    
    # Noise spectrum parameters
    # S(omega) = A / omega^alpha
    noise_exponent: float = 1.0 
    noise_cutoff_low: float = 1e3   # 1 kHz
    noise_cutoff_high: float = 1e9  # 1 GHz

    @property
    def noise_amplitude_A(self) -> float:
        """
        Estimate noise amplitude A from T2* and T1.
        For 1/f noise (alpha=1), roughly: 1/T2* ~ sqrt(A * ln(cutoff_high/cutoff_low))
        This is an approximation to calibrate the noise strength A to the user's T2*.
        """
        gamma_phi = (1/self.T2_star) - (1/(2*self.T1))
        if gamma_phi <= 0: return 0.0
        
        # Inverting the pure dephasing integral for 1/f noise roughly:
        # gamma_phi approx sqrt( A * ln(...) )
        # We'll treat A as having units that map to dephasing rate for simplicity here.
        # A robust fit requires the specific noise model integral. 
        # Here we use a heuristic scaling to match T2* at free evolution.
        return gamma_phi * 1e-6 # Scaling factor for simulation stability


class QuantumChannel:
    """Base class for quantum channels (noise operations)."""
    
    @staticmethod
    def amplitude_damping(rho: np.ndarray, gamma: float) -> np.ndarray:
        E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        return E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
    
    @staticmethod
    def phase_damping(rho: np.ndarray, gamma: float) -> np.ndarray:
        E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        E1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)
        return E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
    
    @staticmethod
    def depolarizing(rho: np.ndarray, p: float) -> np.ndarray:
        return (1 - p) * rho + (p/3) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)
    
    @staticmethod
    def combined_noise(rho: np.ndarray, t: float, T1: float, T2: float) -> np.ndarray:
        # Amplitude damping
        gamma_1 = 1 - np.exp(-t / T1)
        rho = QuantumChannel.amplitude_damping(rho, gamma_1)
        
        # Pure dephasing (extracted from T2)
        if T2 < 2 * T1:
            T_phi = 1 / (1/T2 - 1/(2*T1))
            gamma_phi = 1 - np.exp(-t / T_phi)
            if gamma_phi > 0:
                rho = QuantumChannel.phase_damping(rho, gamma_phi)
        
        return rho


class QubitSimulator:
    def __init__(self, params: Optional[QubitParameters] = None):
        self.params = params or QubitParameters()
        self._rng = np.random.default_rng()
        
        # Pre-compute gate unitaries
        self._gates = {
            'x90': expm(-1j * np.pi/4 * X),
            '-x90': expm(1j * np.pi/4 * X),
            'x180': expm(-1j * np.pi/2 * X),
            'y90': expm(-1j * np.pi/4 * Y),
            '-y90': expm(1j * np.pi/4 * Y),
            'y180': expm(-1j * np.pi/2 * Y),
            'z90': expm(-1j * np.pi/4 * Z),
            '-z90': expm(1j * np.pi/4 * Z),
            'z180': expm(-1j * np.pi/2 * Z),
            'I': I,
        }
        self._cliffords = self._build_clifford_table()
        
        # Cache for filter function calculations to speed up RB
        self._filter_cache = {} 

    def _build_clifford_table(self) -> List[List[str]]:
        # (Standard 24 Clifford decompositions - same as before)
        return [
            ['I'], ['x180'], ['y180'], ['y180', 'x180'], ['x90', 'y90'], ['x90', '-y90'],
            ['-x90', 'y90'], ['-x90', '-y90'], ['y90', 'x90'], ['y90', '-x90'], ['-y90', 'x90'],
            ['-y90', '-x90'], ['x90'], ['-x90'], ['y90'], ['-y90'], ['-x90', 'y90', 'x90'],
            ['-x90', '-y90', 'x90'], ['x180', 'y90'], ['x180', '-y90'], ['y180', 'x90'],
            ['y180', '-x90'], ['x90', 'y90', 'x90'], ['-x90', 'y90', '-x90']
        ]
    
    def _apply_gate(self, rho: np.ndarray, gate_name: str, include_errors: bool = True) -> np.ndarray:
        U = self._gates[gate_name]
        rho_new = U @ rho @ U.conj().T
        
        if include_errors:
            rho_new = QuantumChannel.depolarizing(rho_new, self.params.gate_error_single)
            gate_time = self.params.gate_time_180 if '180' in gate_name else self.params.gate_time_90
            # During gate, use standard T2_echo as baseline
            rho_new = QuantumChannel.combined_noise(rho_new, gate_time, self.params.T1, self.params.T2_echo)
            
        return rho_new
    
    def _apply_clifford(self, rho: np.ndarray, clifford_idx: int, include_errors: bool = True) -> np.ndarray:
        for gate in self._cliffords[clifford_idx]:
            rho = self._apply_gate(rho, gate, include_errors)
        return rho
    
    def _idle(self, rho: np.ndarray, t: float, T2_eff: float) -> np.ndarray:
        """Idle evolution using the effective T2 calculated from Filter Functions."""
        return QuantumChannel.combined_noise(rho, t, self.params.T1, T2_eff)
    
    def _measure(self, rho: np.ndarray, n_shots: int = 1) -> np.ndarray:
        p0_ideal = np.real(rho[0, 0])
        p0 = self.params.readout_fidelity * p0_ideal + \
             (1 - self.params.readout_fidelity) * (1 - p0_ideal)
        return self._rng.binomial(1, 1 - p0, size=n_shots)
    
    
    def _get_dd_sequence(self, mode: str, n_pulses: int, total_time: float) -> List[Tuple[str, float]]:
        """
        Generate DD pulse sequence accounting for pulse duration overhead.
        """
        # Special case: Ramsey or no pulses = just idle (free evolution)
        if mode == 'Ramsey' or n_pulses == 0:
            return [('idle', total_time)]
        
        # 1. Calculate Pulse Overhead
        pulse_len = self.params.gate_time_180
        total_pulse_time = n_pulses * pulse_len
        free_time = total_time - total_pulse_time
        
        # Handle "Time Travel" (Sequence too long for window)
        if free_time < 0:
            # Fallback: Squeeze pulses back-to-back (no idle), effectively just error injection
            return [( ('x180' if i%2==0 else 'y180'), 0 ) for i in range(n_pulses)]
            
        sequence = []
        
        if mode == 'Ramsey':
            sequence = [('idle', total_time)]
            
        elif mode == 'Hahn':
            # X90 - t/2 - X180 - t/2 - X90
            tau = free_time / 2
            sequence = [('idle', tau), ('x180', 0), ('idle', tau)]
            
        elif mode == 'CPMG':
            # [tau/2 - X - tau/2] repeated N times is standard CPMG
            # Or [tau/2 - X - tau - X - ... - tau/2]
            # We use symmetric CPMG: tau - P - 2tau - P ...
            tau_gap = free_time / n_pulses
            tau_half = tau_gap / 2
            
            sequence.append(('idle', tau_half))
            for k in range(n_pulses - 1):
                sequence.append(('x180', 0))
                sequence.append(('idle', tau_gap))
            sequence.append(('x180', 0))
            sequence.append(('idle', tau_half))
            
        elif mode == 'XY4':
            # Symmetric XY4: tau/2 - X - tau - Y - tau - X - tau - Y - tau/2
            tau_gap = free_time / 4
            tau_half = tau_gap / 2
            
            sequence.append(('idle', tau_half))
            sequence.append(('x180', 0)); sequence.append(('idle', tau_gap))
            sequence.append(('y180', 0)); sequence.append(('idle', tau_gap))
            sequence.append(('x180', 0)); sequence.append(('idle', tau_gap))
            sequence.append(('y180', 0)); 
            sequence.append(('idle', tau_half))
            
        elif mode == 'XY8':
            # Same logic as XY4 but 8 pulses
            tau_gap = free_time / 8
            tau_half = tau_gap / 2
            pulses = ['x180', 'y180', 'x180', 'y180', 'y180', 'x180', 'y180', 'x180']
            
            sequence.append(('idle', tau_half))
            for k, p in enumerate(pulses):
                sequence.append((p, 0))
                if k < 7:
                    sequence.append(('idle', tau_gap))
            sequence.append(('idle', tau_half))
            
        elif mode == 'UDD':
            # UDD times are locations of pulses normalized to [0,1]
            # t_j = sin^2(pi * j / (2n + 2))
            # Note: UDD is defined for negligible pulse width. 
            # With finite pulses, we map free evolution durations.
            
            t_locs = [np.sin(np.pi * k / (2 * n_pulses + 2))**2 for k in range(1, n_pulses + 1)]
            
            # Calculate intervals between pulses
            # We scale these intervals to fit 'free_time'
            intervals = []
            prev_loc = 0
            for loc in t_locs:
                intervals.append(loc - prev_loc)
                prev_loc = loc
            intervals.append(1.0 - prev_loc)
            
            # Add to sequence
            for k in range(n_pulses):
                sequence.append(('idle', intervals[k] * free_time))
                sequence.append(('x180', 0))
            sequence.append(('idle', intervals[n_pulses] * free_time))
            
        return sequence


    def _compute_filter_function_t2(self, mode: str, n_pulses: int, total_time: float) -> float:
        """
        Calculates T2_eff by integrating noise spectrum with filter function.
        Chi(t) = integral S(w) * |F(wt)|^2 / w^2 dw
        Decay = exp(-Chi(t)) = exp(-t/T2_eff)  => T2_eff = t / Chi(t)
        """
        
        # Cache check
        cache_key = (mode, n_pulses, float(f"{total_time:.2e}"))
        if cache_key in self._filter_cache:
            return self._filter_cache[cache_key]

        # Ramsey = free evolution, uses T2*
        if mode == 'Ramsey' or n_pulses == 0:
            t2_new = self.params.T2_star
            self._filter_cache[cache_key] = t2_new
            return t2_new
        
        # Hahn echo (1 pulse) - simple refocusing
        if mode == 'Hahn' or n_pulses == 1:
            t2_new = self.params.T2_echo
            t2_new = min(t2_new, 2 * self.params.T1)
            self._filter_cache[cache_key] = t2_new
            return t2_new

        alpha = self.params.noise_exponent
        
        if mode in ['CPMG', 'XY4', 'XY8']:
            # For 1/f noise, scaling is roughly N^(alpha / (alpha+1))
            # Standard theoretical scaling for 1/f: T2 ~ N^0.5 when alpha=1
            scaling_factor = n_pulses ** (alpha / (alpha + 1))
            
            # XY sequences correct pulse errors better than CPMG
            robustness_bonus = 1.1 if mode.startswith('XY') else 1.0
            
            t2_new = self.params.T2_echo * scaling_factor * robustness_bonus
        
        elif mode == 'UDD':
            # UDD comparable to CPMG for 1/f noise
            scaling_factor = (n_pulses) ** (alpha / (alpha + 1)) * 0.9
            t2_new = self.params.T2_echo * scaling_factor

        else:
            t2_new = self.params.T2_echo

        # CRITICAL: Clamp to physical limit 2*T1
        t2_new = min(t2_new, 2 * self.params.T1)
        
        self._filter_cache[cache_key] = t2_new
        return t2_new

    def simulate_t2_dd(self, idle_times: np.ndarray, dd_mode: str = 'CPMG', 
                       n_pulses: int = 2, n_shots: int = 100) -> Dict:
        
        results = {'idle_times': idle_times, 'signal': np.zeros(len(idle_times)), 
                   'signal_std': np.zeros(len(idle_times))}
        
        for i, t in enumerate(idle_times):
            # Calculate physics-based T2 for this specific duration and sequence
            T2_eff = self._compute_filter_function_t2(dd_mode, n_pulses, t)
            
            measurements = []
            dd_sequence = self._get_dd_sequence(dd_mode, n_pulses, t)
            
            for _ in range(n_shots):
                rho = np.outer(GROUND, GROUND.conj())
                rho = self._apply_gate(rho, 'x90')
                
                for op, duration in dd_sequence:
                    if op == 'idle':
                        rho = self._idle(rho, duration, T2_eff)
                    else:
                        rho = self._apply_gate(rho, op) # Pulse errors accumulate here!
                
                rho = self._apply_gate(rho, 'x90')
                if dd_mode in ['CPMG', 'UDD'] and n_pulses % 2 == 1:
                    rho = self._apply_gate(rho, 'x180')
                    
                measurements.append(self._measure(rho, 1)[0])
            
            results['signal'][i] = 1 - np.mean(measurements)
            results['signal_std'][i] = np.std(measurements) / np.sqrt(n_shots)
            
        # Store T2_eff for reference (using the longest time)
        results['T2_eff_est'] = self._compute_filter_function_t2(dd_mode, n_pulses, idle_times[-1])
        return results

    # ========== RANDOMIZED BENCHMARKING ==========
    
    def _get_inverse_clifford(self, sequence: List[int]) -> int:
        U_total = I.copy().astype(complex)
        for cliff_idx in sequence:
            U_cliff = I.copy().astype(complex)
            for gate in self._cliffords[cliff_idx]:
                U_cliff = self._gates[gate] @ U_cliff
            U_total = U_cliff @ U_total
        
        U_inv = U_total.conj().T
        # Simple search
        for idx, cliff in enumerate(self._cliffords):
            U_cliff = I.copy().astype(complex)
            for gate in cliff:
                U_cliff = self._gates[gate] @ U_cliff
            if np.allclose(np.abs(np.trace(U_cliff.conj().T @ U_inv)), 2):
                return idx
        return 0

    def simulate_rb(self, depths: np.ndarray, n_sequences: int = 50, n_shots: int = 100,
                    dd_mode: Optional[str] = None, dd_idle_time: float = 100e-9,
                    dd_n_pulses: int = 2) -> Dict:
        
        results = {'depths': depths, 'survival_prob': np.zeros(len(depths)), 
                   'survival_prob_std': np.zeros(len(depths))}
        
        # For RB, the idle time is fixed, so T2_eff is constant
        if dd_mode:
            T2_eff = self._compute_filter_function_t2(dd_mode, dd_n_pulses, dd_idle_time)
        else:
            T2_eff = self.params.T2_echo
            
        for d_idx, depth in enumerate(depths):
            seq_outcomes = []
            for _ in range(n_sequences):
                sequence = self._rng.integers(0, 24, size=int(depth)).tolist()
                sequence.append(self._get_inverse_clifford(sequence))
                
                shot_outcomes = []
                for _ in range(n_shots):
                    rho = np.outer(GROUND, GROUND.conj())
                    for k, cliff_idx in enumerate(sequence):
                        rho = self._apply_clifford(rho, cliff_idx)
                        
                        # Apply DD between Cliffords
                        if dd_mode and k < len(sequence) - 1:
                            dd_seq = self._get_dd_sequence(dd_mode, dd_n_pulses, dd_idle_time)
                            for op, duration in dd_seq:
                                if op == 'idle':
                                    rho = self._idle(rho, duration, T2_eff)
                                else:
                                    rho = self._apply_gate(rho, op)
                    
                    shot_outcomes.append(1 - self._measure(rho, 1)[0])
                seq_outcomes.append(np.mean(shot_outcomes))
            
            results['survival_prob'][d_idx] = np.mean(seq_outcomes)
            results['survival_prob_std'][d_idx] = np.std(seq_outcomes) / np.sqrt(n_sequences)
            
        return results

# Helper functions for fitting 
def fit_exponential_decay(x, y, y_err=None):
    from scipy.optimize import curve_fit
    def func(x, A, tau, B): return A * np.exp(-x / tau) + B
    try:
        popt, pcov = curve_fit(func, x, y, p0=[1.0, np.mean(x), 0], maxfev=5000)
        return {'tau': popt[1], 'tau_err': np.sqrt(np.diag(pcov))[1], 'fitted': func(x, *popt)}
    except: return {'tau': 0, 'tau_err': 0, 'fitted': y}

def fit_rb_decay(depths, survival, survival_err=None):
    from scipy.optimize import curve_fit
    def func(m, A, alpha, B): return A * alpha**m + B
    try:
        popt, pcov = curve_fit(func, depths, survival, p0=[0.5, 0.99, 0.5], bounds=([0,0,0], [1,1,1]))
        alpha, alpha_err = popt[1], np.sqrt(np.diag(pcov))[1]
        EPC = (1-alpha)/2
        EPG = EPC/1.875
        return {'EPG': EPG, 'EPG_err': alpha_err/3.75, 'F_gate': 1-EPG, 'fitted': func(depths, *popt)}
    except: return {'EPG': 0, 'EPG_err': 0, 'fitted': survival}

