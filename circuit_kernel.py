import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def swap_test_circuit(test_vec, train_vec):
    qc = QuantumCircuit(3)
    qc.initialize(test_vec, [1])
    qc.initialize(train_vec,[2])
    qc.h(0)
    qc.cswap(0,1,2)
    qc.h(0)
    return qc


def fidelity_from_circuit(test_vec, train_vec):
    qc = swap_test_circuit(test_vec, train_vec)
    sv = Statevector.from_instruction(qc)
    p_ancilla = sv.probabilities([0])
    return 2 * p_ancilla[0] - 1

# ---------------------------------------------------------------------------
# Step 2b -- full toy-example classifier circuit (5 qubits)
# ---------------------------------------------------------------------------
 
def prepare_training_register(x1, x2, w1=0.5, w2=0.5):
    qc = QuantumCircuit(3)
    alpha = 2 * np.arccos(np.sqrt(w1))
    qc.ry(alpha, 0)
    qc.initialize(x1, [1])
    qc.cz(0, 1)
    qc.cx(0, 2)
    return qc
 
 
def full_classifier_circuit(test_vec, x1, x2, w1=0.5, w2=0.5):
    qc = QuantumCircuit(5)
    qc.initialize(test_vec, [1])
    training_reg = prepare_training_register(x1, x2, w1, w2)
    qc.compose(training_reg, qubits=[4, 2, 3], inplace=True)
    qc.h(0)
    qc.cswap(0, 1, 2)
    qc.h(0)
    return qc
 
 
def expectation_value_from_circuit(theta, w1=0.5, w2=0.5):
    from closed_form_kernel import training_data, test_state
    x1, y1, x2, y2 = training_data()
    qc = full_classifier_circuit(test_state(theta), x1, x2, w1, w2)
    sv = Statevector.from_instruction(qc)
    p = sv.probabilities([0, 3])
    return p[0] - p[1] - p[2] + p[3]
 