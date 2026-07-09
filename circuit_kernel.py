import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import SwapGate, XGate


def swap_test_circuit(test_vec, train_vec):
    qc = QuantumCircuit(3)
    qc.initialize(test_vec, [1])
    qc.initialize(train_vec, [2])
    qc.h(0)
    qc.cswap(0, 1, 2)
    qc.h(0)
    return qc


def fidelity_from_circuit(test_vec, train_vec):
    qc = swap_test_circuit(test_vec, train_vec)
    sv = Statevector.from_instruction(qc)
    p_ancilla = sv.probabilities([0])
    return 2 * p_ancilla[0] - 1


# ---------------------------------------------------------------------------
# Step 2b -- quantum forking classifier (arbitrary M training points)
# ---------------------------------------------------------------------------

def prepare_training_register(states, labels, weights):
    """
    Prepare Σ_m sqrt(w_m) |m⟩_idx |x_m⟩_data |y_m⟩_lbl via quantum forking.

    Routing: initialise all M data registers, put index in weighted superposition,
    then for each m ≥ 1 do CSWAP(index == m, active_data, aux_data[m]).
    After routing, active_data qubit holds x_m when the index register is |m⟩.

    Qubit layout (n = 1 qubit per training state):
        q[0 .. k-1]    index           (k = ceil(log2(M)))
        q[k]           active data     (the qubit swapped with test in classifier)
        q[k+1 .. k+M-1] auxiliary data (initialised to x_1 .. x_{M-1}, 0-indexed)
        q[k+M]         label
    """
    M = len(states)
    k = max(1, int(np.ceil(np.log2(M))))
    M_pad = 2 ** k

    qc = QuantumCircuit(k + M + 1)

    # Initialise all data registers
    qc.initialize(states[0], [k])
    for i in range(1, M):
        qc.initialize(states[i], [k + i])

    # Index superposition: amplitude sqrt(w_m) for |m⟩, 0 for m >= M
    idx_amps = np.zeros(M_pad)
    idx_amps[:M] = np.sqrt(weights)
    qc.initialize(idx_amps, list(range(k)))

    # Quantum forking: CSWAP active_data <-> aux[m] when index == m
    for m in range(1, M):
        gate = SwapGate().control(k, ctrl_state=m)
        qc.append(gate, list(range(k)) + [k, k + m])

    # Label: initialise to y_0, flip for each m where y_m differs
    label_q = k + M
    if labels[0] == 1:
        qc.x(label_q)
    for m in range(1, M):
        if labels[m] != labels[0]:
            gate = XGate().control(k, ctrl_state=m)
            qc.append(gate, list(range(k)) + [label_q])

    return qc


def full_classifier_circuit(test_vec, states, labels, weights):
    """
    Full swap-test classifier for M training points.

    Returns (circuit, label_qubit_index).  label_qubit_index is needed by
    expectation_value_from_circuit to extract <sigma_z^(a) sigma_z^(l)>.

    Global qubit layout:
        q[0]            ancilla
        q[1]            test
        q[2 .. 2+k-1]  index
        q[2+k]          active data
        q[2+k+1 ..]    auxiliary data
        q[2+k+M]        label
    """
    M = len(states)
    k = max(1, int(np.ceil(np.log2(M))))
    n_train = k + M + 1

    qc = QuantumCircuit(2 + n_train)
    qc.initialize(test_vec, [1])

    training_reg = prepare_training_register(states, labels, weights)
    qc.compose(training_reg, qubits=list(range(2, 2 + n_train)), inplace=True)

    active_data_q = 2 + k
    qc.h(0)
    qc.cswap(0, 1, active_data_q)
    qc.h(0)

    return qc, 2 + k + M


def expectation_value_from_circuit(theta, w1=0.5, w2=0.5):
    from closed_form_kernel import training_data, test_state
    x1, y1, x2, y2 = training_data()
    qc, label_q = full_classifier_circuit(
        test_state(theta), [x1, x2], [int(y1), int(y2)], [w1, w2]
    )
    sv = Statevector.from_instruction(qc)
    p = sv.probabilities([0, label_q])
    return p[0] - p[1] - p[2] + p[3]
