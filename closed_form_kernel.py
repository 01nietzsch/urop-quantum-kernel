import numpy as np
import matplotlib.pyplot as plt


def training_data():
    x1 = np.array([1j/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    y1 = 0
    x2 = np.array([1j/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
    y2 = 1
    return x1, y1, x2, y2
    
def test_state(theta):
    return np.array([np.cos(theta/2), - 1j * np.sin(theta/2)], dtype=complex)

def fidelity(a, b, n=1):
    inner_product = np.vdot(a, b)         
    return np.abs(inner_product) ** (2 * n)

def expectation_value(theta, w1=0.5, w2=0.5, n=1):
    test_states = test_state(theta)
    x1, y1, x2, y2 = training_data()
    expected_sigma = w1*fidelity(test_states, x1, n) - w2*fidelity(test_states, x2, n)
    return expected_sigma

def classify(theta, w1=0.5, w2=0.5, n=1):
    exp = expectation_value(theta, w1, w2, n)
    y_pred = 1/2 * (1- np.sign(exp))
    return y_pred 

def hadamard_classifier(theta, w1=0.5, w2=0.5, n=1):
    test_states = test_state(theta)
    x1, y1, x2, y2 = training_data()
    expected_sigma = w1 * np.real(np.vdot(test_states, x1)) - w2*np.real(np.vdot(test_states, x2)) 
    return expected_sigma

if __name__=="__main__" : 
    thetas = np.linspace(0,6,num=8)
    expectation_values = np.array([expectation_value(t) for t in thetas])
    y_preds  = np.array([classify(t) for t in thetas])

    # test with the hadamard: 
    expectation_values_hadamard = np.array([hadamard_classifier(t) for t in thetas])
    plt.figure()
    plt.scatter(thetas, expectation_values, color = 'red', label = r"$\langle \sigma_z^{(a)} \sigma_z^{(l)} \rangle$")
    plt.scatter(thetas, expectation_values_hadamard, color = 'green', label = 'Hadamard classifier')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.xlabel('theta')
    plt.title('Swap-test classifier (fidelity kernel), n=1')
    plt.legend()
    plt.show()