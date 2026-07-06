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
    """
    Stage A: prepare the entangled index + data + label register (3 qubits).
 
    After this circuit, the joint state of [index, data, label] is:
        sqrt(w1)|0>_m |x1>_d |0>_l  +  sqrt(w2)|1>_m |x2>_d |1>_l
 
    which encodes both training points, their weights, and their labels
    in a single superposition -- this is what lets the swap test compute
    the weighted fidelity difference in one shot.
 
    Circuit (3 qubits: index=q0, data=q1, label=q2):
        1. Ry(alpha) on index,  alpha = 2*arccos(sqrt(w1))
           -> sqrt(w1)|0> + sqrt(w2)|1>  on index
        2. initialize(x1) on data  (unconditionally)
        3. controlled-Z on data, controlled on index
           -> Z|x1> = |x2> in the index=|1> branch
              (works because x1 and x2 differ only by a sign on |1>)
        4. controlled-X on label, controlled on index
           -> label flips to |1> = y2  in the index=|1> branch
 
    NOTE: step 3 is specific to this toy dataset where Z|x1>=|x2>.
    The general approach (quantum forking, Eq. 15 / Fig. 4 of the paper)
    uses controlled-SWAPs instead and works for arbitrary training states.
 
    Returns
    -------
    QuantumCircuit on 3 qubits [index(q0), data(q1), label(q2)]
    """
    # TODO: implement the four steps above.
    #
    qc = QuantumCircuit(3)
    alpha = 2 * np.arccos(np.sqrt(w1))
    qc.ry(alpha, 0)
    qc.initialize(x1, [1])
    qc.cz(0, 1)
    qc.cx(0, 2)
    return qc
 
 
def full_classifier_circuit(test_vec, x1, x2, w1=0.5, w2=0.5):
    """
    Full 5-qubit swap-test classifier: Stage A (training register prep)
    followed by Stage B (swap test between test qubit and data qubit).
 
    Qubit layout: 0=ancilla, 1=test, 2=data, 3=label, 4=index.
 
    Returns
    -------
    QuantumCircuit (5 qubits, no measurements -- statevector is read directly)
    """
    # TODO: combine prepare_training_register with the swap-test sandwich.
    #
    qc = QuantumCircuit(5)
    qc.initialize(test_vec, [1])
    training_reg = prepare_training_register(x1, x2, w1, w2)
    qc.compose(training_reg, qubits=[4, 2, 3], inplace=True)
    qc.h(0)
    qc.cswap(0, 1, 2)
    qc.h(0)
    return qc
 
 
def expectation_value_from_circuit(theta, w1=0.5, w2=0.5):
    """
    Stage C: extract <sigma_z^(a) sigma_z^(l)> from the statevector.
 
    sigma_z eigenvalues: +1 for |0>, -1 for |1>.
    So sigma_z^(a) x sigma_z^(l) has eigenvalue:
        +1 when ancilla and label are the same  (|00> or |11>)
        -1 when ancilla and label differ        (|01> or |10>)
 
    sv.probabilities([0, 3]) returns [p(a=0,l=0), p(a=0,l=1), p(a=1,l=0), p(a=1,l=1)]
    in little-endian order (qubit 0 is least-significant bit):
        index 0 = |a=0, l=0>  -> eigenvalue +1
        index 1 = |a=0, l=1>  -> eigenvalue -1
        index 2 = |a=1, l=0>  -> eigenvalue -1
        index 3 = |a=1, l=1>  -> eigenvalue +1
 
    So: <sigma_z^(a) sigma_z^(l)> = p[0] - p[1] - p[2] + p[3]
 
    This should match expectation_value(theta) from closed_form_kernel.py.
 
    Returns
    -------
    float
    """
    from closed_form_kernel import training_data, test_state
    x1, y1, x2, y2 = training_data()
    qc = full_classifier_circuit(test_state(theta), x1, x2, w1, w2)
    sv = Statevector.from_instruction(qc)
 
    # TODO: apply the formula above.
    #
    # p = sv.probabilities([0, 3])
    # return p[0] - p[1] - p[2] + p[3]
    raise NotImplementedError
 