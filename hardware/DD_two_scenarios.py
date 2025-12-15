"""
DD Two Scenarios Experiment

Shows when DD helps vs hurts:
- Scenario A (400ns idle): DD overhead dominates -> DD hurts
- Scenario B (4us idle): Coherence benefit dominates -> DD helps

Needs calibrated qubit with x90/x180/y90/y180 and state discrimination.

Author: Kunal Sinha
"""

from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_decay_exp
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import time


class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q3"]
    
    # T2 settings
    t2_num_averages: int = 100
    t2_min_wait_ns: int = 16
    t2_max_wait_ns: int = 40000
    t2_num_points: int = 20
    
    # RB settings
    rb_num_sequences: int = 50
    rb_num_averages: int = 1
    rb_max_depth: int = 200
    rb_delta_clifford: int = 20
    rb_seed: int = 345324
    
    # DD idle times
    dd_idle_short_ns: int = 400   # Scenario A
    dd_idle_long_ns: int = 4000   # Scenario B
    
    # Flags
    run_t2_measurements: bool = False
    run_scenario_a: bool = True
    run_scenario_b: bool = True
    
    # Hardware
    flux_point: Literal["joint", "independent"] = "independent"
    reset_type: Literal["active", "thermal"] = "active"
    use_state_discrimination: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 10000
    timeout: int = 100


node = QualibrationNode(
    name="DD_two_scenarios",
    description="DD helps vs hurts comparison",
    parameters=Parameters()
)

u = unit(coerce_to_integer=True)
machine = QuAM.load()
config = machine.generate_config()
qmm = machine.connect()

if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]


def generate_sequence(max_depth, seed):
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=max_depth + 1)
    inv_gate = declare(int, size=max_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < max_depth, i + 1):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])

    return sequence, inv_gate


def play_clifford(qubit: Transmon, cliff_idx):
    with switch_(cliff_idx, unsafe=True):
        with case_(0):
            qubit.xy.wait(qubit.xy.operations["x180"].length // 4)
        with case_(1):
            qubit.xy.play("x180")
        with case_(2):
            qubit.xy.play("y180")
        with case_(3):
            qubit.xy.play("y180")
            qubit.xy.play("x180")
        with case_(4):
            qubit.xy.play("x90")
            qubit.xy.play("y90")
        with case_(5):
            qubit.xy.play("x90")
            qubit.xy.play("-y90")
        with case_(6):
            qubit.xy.play("-x90")
            qubit.xy.play("y90")
        with case_(7):
            qubit.xy.play("-x90")
            qubit.xy.play("-y90")
        with case_(8):
            qubit.xy.play("y90")
            qubit.xy.play("x90")
        with case_(9):
            qubit.xy.play("y90")
            qubit.xy.play("-x90")
        with case_(10):
            qubit.xy.play("-y90")
            qubit.xy.play("x90")
        with case_(11):
            qubit.xy.play("-y90")
            qubit.xy.play("-x90")
        with case_(12):
            qubit.xy.play("x90")
        with case_(13):
            qubit.xy.play("-x90")
        with case_(14):
            qubit.xy.play("y90")
        with case_(15):
            qubit.xy.play("-y90")
        with case_(16):
            qubit.xy.play("-x90")
            qubit.xy.play("y90")
            qubit.xy.play("x90")
        with case_(17):
            qubit.xy.play("-x90")
            qubit.xy.play("-y90")
            qubit.xy.play("x90")
        with case_(18):
            qubit.xy.play("x180")
            qubit.xy.play("y90")
        with case_(19):
            qubit.xy.play("x180")
            qubit.xy.play("-y90")
        with case_(20):
            qubit.xy.play("y180")
            qubit.xy.play("x90")
        with case_(21):
            qubit.xy.play("y180")
            qubit.xy.play("-x90")
        with case_(22):
            qubit.xy.play("x90")
            qubit.xy.play("y90")
            qubit.xy.play("x90")
        with case_(23):
            qubit.xy.play("-x90")
            qubit.xy.play("y90")
            qubit.xy.play("-x90")


def build_rb_program(idle_time_ns: int = 0, use_dd: bool = False):
    num_sequences = node.parameters.rb_num_sequences
    n_avg = node.parameters.rb_num_averages
    max_depth = node.parameters.rb_max_depth
    delta_clifford = node.parameters.rb_delta_clifford
    seed = node.parameters.rb_seed
    idle_cc = idle_time_ns // 4
    num_depths = max_depth // delta_clifford + 1
    
    with program() as prog:
        depth = declare(int)
        depth_target = declare(int)
        saved_gate = declare(int)
        m = declare(int)
        n = declare(int)
        j = declare(int)
        t_seg = declare(int)
        
        I, I_st, Q, Q_st, _, n_st = qua_declaration(num_qubits=num_qubits)
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        m_st = declare_stream()
        
        assign(t_seg, idle_cc // 8)
        
        for i, qubit in enumerate(qubits):
            align()
            machine.set_all_fluxes(flux_point=node.parameters.flux_point, target=qubit)
            
            with for_(m, 0, m < num_sequences, m + 1):
                sequence_list, inv_gate_list = generate_sequence(max_depth, seed)
                assign(depth_target, 0)
                
                with for_(depth, 1, depth <= max_depth, depth + 1):
                    assign(saved_gate, sequence_list[depth])
                    assign(sequence_list[depth], inv_gate_list[depth - 1])
                    
                    with if_((depth == 1) | (depth == depth_target)):
                        with for_(n, 0, n < n_avg, n + 1):
                            if node.parameters.reset_type == "active":
                                active_reset(qubit, "readout")
                            else:
                                qubit.resonator.wait(qubit.thermalization_time * u.ns)
                            qubit.align()
                            
                            with for_(j, 0, j <= depth, j + 1):
                                play_clifford(qubit, sequence_list[j])
                                
                                with if_(j < depth):
                                    if idle_cc > 0:
                                        if use_dd:
                                            with if_(t_seg >= 4):
                                                qubit.xy.wait(t_seg)
                                                qubit.xy.play("x180")
                                                qubit.xy.wait(t_seg * 2)
                                                qubit.xy.play("y180")
                                                qubit.xy.wait(t_seg * 2)
                                                qubit.xy.play("x180")
                                                qubit.xy.wait(t_seg * 2)
                                                qubit.xy.play("y180")
                                                qubit.xy.wait(t_seg)
                                            with else_():
                                                qubit.xy.wait(idle_cc)
                                        else:
                                            qubit.xy.wait(idle_cc)
                            
                            qubit.align()
                            readout_state(qubit, state[i])
                            save(state[i], state_st[i])
                        
                        assign(depth_target, depth_target + delta_clifford)
                    
                    assign(sequence_list[depth], saved_gate)
                
                save(m, m_st)
        
        with stream_processing():
            m_st.save("iteration")
            for i in range(num_qubits):
                state_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_sequences).save(
                    f"state{i + 1}"
                )
    
    return prog, num_depths, num_sequences


def run_rb(idle_ns: int, use_dd: bool, label: str):
    print(f"  Running: {label}")
    prog, num_depths, num_sequences = build_rb_program(idle_ns, use_dd)
    
    start = time.time()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(prog)
        results = fetching_tool(job, ["iteration"], mode="live")
        while results.is_processing():
            m = results.fetch_all()[0]
            progress_counter(m, num_sequences, start_time=results.start_time)
    elapsed = time.time() - start
    
    depths = np.arange(0, node.parameters.rb_max_depth + 0.1, node.parameters.rb_delta_clifford)
    depths[0] = 1
    ds = fetch_results_as_xarray(
        job.result_handles, qubits,
        {"depths": depths, "sequence": np.arange(num_sequences)},
    )
    
    da_state = 1 - ds["state"].mean(dim="sequence")
    da_state = da_state.assign_coords(depths=da_state.depths - 1).rename(depths="m")
    
    da_fit = fit_decay_exp(da_state, "m")
    alpha = np.exp(da_fit.sel(fit_vals="decay"))
    EPC = (1 - alpha) - (1 - alpha) / 2
    EPG = EPC / 1.875
    
    result = {
        'label': label,
        'idle_ns': idle_ns,
        'use_dd': use_dd,
        'EPG': {q.name: float(EPG.sel(qubit=q.name).values) for q in qubits},
        'runtime_s': elapsed,
    }
    
    for q in qubits:
        print(f"    {q.name}: EPG = {result['EPG'][q.name]*100:.4f}%")
    
    return result


if __name__ == "__main__":
    print(f"\nRunning on: {[q.name for q in qubits]}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'qubits': [q.name for q in qubits],
    }
    
    total_start = time.time()
    
    if node.parameters.run_scenario_a:
        print("\nScenario A (short idle):")
        idle_a = node.parameters.dd_idle_short_ns
        results['RB_baseline'] = run_rb(0, False, "Baseline")
        results['RB_A_no_dd'] = run_rb(idle_a, False, f"No DD ({idle_a}ns)")
        results['RB_A_xy4'] = run_rb(idle_a, True, f"XY-4 ({idle_a}ns)")
    
    if node.parameters.run_scenario_b:
        print("\nScenario B (long idle):")
        idle_b = node.parameters.dd_idle_long_ns
        results['RB_B_no_dd'] = run_rb(idle_b, False, f"No DD ({idle_b}ns)")
        results['RB_B_xy4'] = run_rb(idle_b, True, f"XY-4 ({idle_b}ns)")
    
    results['total_runtime_s'] = time.time() - total_start
    
    save_dir = Path("../data/hardware")
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / f"dd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone. Saved to {filename}")
