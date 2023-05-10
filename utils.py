# Content: functions for experiments

import numpy as np
import paddle
from paddle_quantum.ansatz import Circuit


def haar_unitary(dim : int) -> paddle.Tensor:
    r""" randomly generate a unitary following Haar random, referenced by arXiv:math-ph/0609050v2

    Args:
        num_qubits: number of qubits :math:`n`

    Returns:
        a :math:`2^n \times 2^n` unitary
        
    """
    # Step 1: sample from Ginibre ensemble
    ginibre = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2)
    # Step 2: perform QR decomposition of G
    mat_q, mat_r = np.linalg.qr(ginibre)
    # Step 3: make the decomposition unique
    mat_lambda = np.diag(mat_r) / np.abs(np.diag(mat_r))
    mat_u = mat_q @ np.diag(mat_lambda)
    return mat_u

def haar_subspace_state(basis: np.ndarray) -> np.ndarray:
    r"""create a Haar random state in the subspace spanned by the basis.
    
    Args:
        basis (np.ndarray): an isometry with columns being the basis.
    """
    dim_subspace = basis.shape[1]
    # generate haar random coefficient vector
    coef_array = haar_unitary(dim_subspace)[:,0]
    # linear combination of the basis using the coefficients
    sample_state = basis @ coef_array
    return sample_state

def random_state_fixed_overlap(unitary: np.ndarray, overlap: float) -> np.ndarray:
    r"""create a random state that has a fixed overlap with the given state.
 
    Args:
        unitary (np.ndarray): a unitary with the first column being the given state.
        overlap (float, optional): the given overlap.
    """
    # record the given state
    given_state = unitary[:, 0]
    # create a Haar random state in the orthogonal complement subspace of the given state
    perp_state = haar_subspace_state(unitary[:, 1:]) 
    # combine together
    sample_state = overlap * given_state + np.sqrt(1-overlap**2) * perp_state
    return sample_state


def Alternating_Layer_ansatz(num_qubits: int, depth=1) -> Circuit:
    r"""Generate alternating layer ansatz.

    Args:
        num_qubits (int): Number of qubits.
        depth (int, optional): Depth. Defaults to 1.

    Returns:
        Circuit: ALT ansatz.
    """
    assert num_qubits % 2 == 0, "The number of qubits is not even"
    cir = Circuit(num_qubits)
    cir.ry()
    cir.rz()

    for _ in range(depth):
        
        for qubit_idx in range(num_qubits):
            if qubit_idx % 2 == 0:
                cir.cz([qubit_idx, qubit_idx + 1])
        cir.ry()
        cir.rz()
        for qubit_idx in range(num_qubits):
            if qubit_idx % 2 == 1 and (qubit_idx + 1) < num_qubits:
                cir.cz([qubit_idx, qubit_idx + 1])                
        cir.ry(range(1, num_qubits - 1))
        cir.rz(range(1, num_qubits - 1))
    return cir


def u_theta(num_qubits: int, depth: int):
    """
    get the circuit and corresponding \theta^*

    Args:
        num_qubits
        depth

    """
    cir = Alternating_Layer_ansatz(num_qubits, depth)

    # initialise the parameters
    cir.randomize_param()
    param = cir.param
    cir.update_param(param)
    print(cir)
    print(param)

    return cir, cir(), cir.param, cir.unitary_matrix(), cir.param[0]


def M_direction_generator(old_param, M):
    """
    generate an arbitrary direction with M elements

    Args:
        old_param
        M: M elements

    """

    if M > len(old_param):
        M = len(old_param)

    new_param = paddle.to_tensor(np.random.uniform(low=0, high=2*np.pi, size=(M,))) 

    return new_param