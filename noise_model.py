"""
Noise model for reproducing Fig. 5 of Blank, Park, Rhee & Petruccione,
"Quantum classifier with tailored quantum kernel", npj QI 6, 41 (2020).

All device parameters are from Supplementary Table I: ibmq_ourense,
calibration 2019-09-29 11:48:14 UTC.

Reproduction target (Supp. Note III D). Fitting
    f(theta) = a * ( sin^2((theta + vartheta)/2 + pi/4) - w2 )
to the noisy simulation should give roughly

    a  ~ 0.8213      (amplitude damping vs theory)
    vartheta ~ 0     (negligible phase shift)
    w2 ~ 0.5023      (ordinate shift)

The hardware run gave a ~ 0.6515, vartheta ~ 2/51 * pi, w2 ~ 0.5414.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import average_gate_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)

# ---------------------------------------------------------------------------
# Supplementary Table I (a): per-qubit data
#   T1, T2 in microseconds; t_u2 (single-qubit gate duration) in nanoseconds
#   readout = eps_r, gate_err = eps (average single-qubit gate infidelity)
#   Device temperature is quoted as 0.0 K, so excited-state population = 0.
# ---------------------------------------------------------------------------
QUBITS = {
    0: dict(T1=94.9785,  T2=93.2334,  f=4.8195, readout=0.0180, gate_err=0.000318, t_1q=36),
    1: dict(T1=101.3888, T2=36.7808,  f=4.8911, readout=0.0180, gate_err=0.000376, t_1q=36),
    2: dict(T1=179.9052, T2=154.4074, f=4.7109, readout=0.0110, gate_err=0.000290, t_1q=50),
    3: dict(T1=126.7049, T2=112.0796, f=4.7690, readout=0.0260, gate_err=0.000909, t_1q=50),
    4: dict(T1=73.0632,  T2=39.0842,  f=5.0237, readout=0.0310, gate_err=0.000337, t_1q=36),
}

# ---------------------------------------------------------------------------
# Supplementary Table I (b): CNOT data. err = average infidelity, t in ns.
#
# NOTE: the gate time for the q3-q1 pair is not legible in the PDF. 300 ns is
# an interpolation between the neighbouring pairs -- REPLACE IT with the real
# value from the authors' data file before quoting any result:
#   github.com/carstenblank/Quantum-classifier-with-tailored-quantum-kernels---Supplemental
#   -> experiment_results/exp_sim_regular_20190929T114806Z.py
# ---------------------------------------------------------------------------
CX = {
    (1, 0): dict(err=0.005685, t=235),
    (2, 1): dict(err=0.007304, t=391),
    (3, 1): dict(err=0.011624, t=300),   # <-- unverified gate time
    (4, 3): dict(err=0.006492, t=210),
}

# ourense topology: a T shape. q1 is the hub, q3 hangs off it, q4 off q3.
COUPLING_MAP = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3]]

# The 2019 native set was {u1, u2, u3, cx}. Those gate names no longer exist in
# current Qiskit, so we use the modern equivalent: rz is virtual (zero duration,
# error-free), and sx / x are each one physical pulse -- the same pulse that u2
# was built from. So the tabulated u2 duration and u2 error attach to sx and x.
BASIS_GATES = ["rz", "sx", "x", "cx"]


def _depolarizing_param(target_err, relax_err, num_qubits):
    """
    Choose the depolarizing strength so that depolarizing composed with thermal
    relaxation reproduces the measured average gate infidelity `target_err`.
    This mirrors qiskit-aer's basic_device_gate_errors.
    """
    relax_fid = average_gate_fidelity(relax_err.to_quantumchannel())
    relax_infid = 1.0 - relax_fid

    if target_err <= relax_infid:
        # Relaxation alone already accounts for (or exceeds) the reported error;
        # nothing left to attribute to depolarizing.
        return 0.0

    dim = 2 ** num_qubits
    p = dim * (target_err - relax_infid) / (dim * relax_fid - 1.0)
    p_max = 4 ** num_qubits / (4 ** num_qubits - 1)
    return float(np.clip(p, 0.0, p_max))


def ourense_noise_model():
    """Build the basic device noise model from Supplementary Table I."""
    nm = NoiseModel(basis_gates=BASIS_GATES)

    # --- single-qubit gates + readout ---
    for q, d in QUBITS.items():
        t1_ns, t2_ns = d["T1"] * 1e3, d["T2"] * 1e3
        relax = thermal_relaxation_error(
            t1_ns, t2_ns, d["t_1q"], excited_state_population=0.0
        )
        p = _depolarizing_param(d["gate_err"], relax, 1)
        err = depolarizing_error(p, 1).compose(relax)

        for gate in ("sx", "x"):
            nm.add_quantum_error(err, gate, [q])
        # rz is virtual: no duration, no error. Deliberately left untouched.

        e_r = d["readout"]
        nm.add_readout_error(
            ReadoutError([[1 - e_r, e_r], [e_r, 1 - e_r]]), [q]
        )

    # --- two-qubit gates ---
    for (qa, qb), d in CX.items():
        relax = thermal_relaxation_error(
            QUBITS[qa]["T1"] * 1e3, QUBITS[qa]["T2"] * 1e3, d["t"], 0.0
        ).expand(
            thermal_relaxation_error(
                QUBITS[qb]["T1"] * 1e3, QUBITS[qb]["T2"] * 1e3, d["t"], 0.0
            )
        )
        p = _depolarizing_param(d["err"], relax, 2)
        err = depolarizing_error(p, 2).compose(relax)

        # Table I does not state the CNOT direction, so we apply the same error
        # to both. On real hardware only one direction is native; the other is
        # the same pulse wrapped in Hadamards, so the difference is small.
        nm.add_quantum_error(err, "cx", [qa, qb])
        nm.add_quantum_error(err, "cx", [qb, qa])

    return nm


def toy_circuit(theta, w2=0.5):
    """
    The paper's Fig. 6 / Supplementary Fig. 2 circuit, on the authors'
    hand-picked physical layout:  a->q0, d->q1, in->q2, m->q3, l->q4.

    Note there is NO quantum forking here. Because x2 is x1 with a sign flip on
    |1>, the whole entangled training register comes from one cz(index, data).
    """
    a, d, x_in, m, l = 0, 1, 2, 3, 4
    qc = QuantumCircuit(5, 2)

    # index qubit carries the weights: Ry(alpha)|0> = sqrt(w1)|0> + sqrt(w2)|1>
    qc.ry(2 * np.arcsin(np.sqrt(w2)), m)

    # prepare |x1> = (i|0> + |1>)/sqrt(2) on the data qubit (up to global phase)
    qc.h(d)
    qc.rz(-np.pi, d)
    qc.s(d)

    qc.cz(m, d)      # index=|1> flips the sign -> |x2>
    qc.cx(m, l)      # label qubit: y_m = m
    qc.rx(theta, x_in)  # test datum: Rx(theta)|0> = cos(t/2)|0> - i sin(t/2)|1>

    # swap test
    qc.h(a)
    qc.cswap(a, x_in, d)
    qc.h(a)

    qc.measure(a, 0)
    qc.measure(l, 1)
    return qc


def expectation_from_counts(counts, shots):
    """<sigma_z^(a) sigma_z^(l)> = (c00 - c01 - c10 + c11) / shots.

    Classical bit 0 is the ancilla, bit 1 the label. Qiskit prints keys
    most-significant-first, so key[-1] is the ancilla and key[-2] the label.
    """
    total = 0
    for key, n in counts.items():
        a, l = int(key[-1]), int(key[-2])
        total += n * (-1) ** (a + l)
    return total / shots


def theory(theta, w2=0.5):
    return (1 - w2) * np.sin(theta / 2 + np.pi / 4) ** 2 - w2 * np.cos(theta / 2 + np.pi / 4) ** 2


def run_sweep(thetas=None, shots=8192, noisy=True, seed=1234):
    """Sweep theta and return the measured expectation values."""
    if thetas is None:
        thetas = np.arange(0.0, 2 * np.pi, 0.1)   # the paper's 63 points

    if noisy:
        nm = ourense_noise_model()
        sim = AerSimulator(
            noise_model=nm,
            basis_gates=nm.basis_gates,
            coupling_map=COUPLING_MAP,
            seed_simulator=seed,
        )
    else:
        sim = AerSimulator(seed_simulator=seed)

    circuits = [toy_circuit(t) for t in thetas]
    tqcs = transpile(
        circuits,
        basis_gates=BASIS_GATES,
        coupling_map=COUPLING_MAP if noisy else None,
        optimization_level=1,
        seed_transpiler=seed,
    )

    # sanity check against the paper's stated gate count
    n_cx = tqcs[0].count_ops().get("cx", 0)
    print(f"transpiled CNOT count: {n_cx}  (paper reports 13)")

    result = sim.run(tqcs, shots=shots).result()
    return np.asarray(thetas), np.array(
        [expectation_from_counts(result.get_counts(i), shots) for i in range(len(tqcs))]
    )


def fit_amplitude(thetas, values):
    """Fit a * (sin^2((theta + vartheta)/2 + pi/4) - w2). Returns (a, vartheta, w2)."""
    from scipy.optimize import curve_fit

    def model(th, a, vartheta, w2):
        return a * (np.sin((th + vartheta) / 2 + np.pi / 4) ** 2 - w2)

    popt, _ = curve_fit(model, thetas, values, p0=[1.0, 0.0, 0.5])
    return popt


if __name__ == "__main__":
    th, vals = run_sweep()
    a, vartheta, w2 = fit_amplitude(th, vals)
    print(f"amplitude a  = {a:.4f}   (paper: 0.8213)")
    print(f"phase shift  = {vartheta:.4f} rad  (paper: ~0)")
    print(f"offset w2    = {w2:.4f}   (paper: 0.5023)")