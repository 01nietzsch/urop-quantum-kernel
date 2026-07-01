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