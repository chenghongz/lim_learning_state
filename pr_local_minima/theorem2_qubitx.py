# Direct verification of Theorem 2 with respect to overlap.

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorcircuit as tc
import jax
from utils import *
K = tc.set_backend("jax")


def get_hessian_func(num_qubits, cir_depth, target_state, argnums):
    # define loss function, use * to unpack so that 'argnums' works
    def f(*params):
        c = tensorcircuit_Alternating_Layer_ansatz(num_qubits, cir_depth, params)
        output_state = c.state()
        overlap = target_state.conj().T @ output_state
        return 1 - tc.backend.real(overlap.conj() * overlap)
    # create the hessian function and jit the function
    hessian_func = tc.backend.jit(jax.hessian(f, argnums))
    # using tc.backend.hessian would lead to a bug
    return hessian_func

def get_grad_func(num_qubits, cir_depth, target_state, argnums):
    # define loss function, use * to unpack so that 'argnums' works
    def f(*params):
        c = tensorcircuit_Alternating_Layer_ansatz(num_qubits, cir_depth, params)
        output_state = c.state()
        overlap = target_state.conj().T @ output_state
        return 1 - tc.backend.real(overlap.conj() * overlap)
    # create the grad function and jit the function
    sampled_gradient = tc.backend.jit(jax.grad(f, argnums))
    return sampled_gradient

def is_positive(mat: np.ndarray, tolerance_error: float = 1e-4):
    min_eig_val = np.linalg.eigvalsh(mat)[0]
    if_positive = (min_eig_val >= -tolerance_error)
    return if_positive, min_eig_val


if __name__ == "__main__":

    np.random.seed(0)

    # some variables
    num_sample_states = 200  # the number of initial state to be sampled
    list_num_qubits = list(range(1, 12))  # list of qubits to be verified
    cir_depth = 5  # the depth of the circuit
    num_params_differentiated = 1 # the number parameters to be considered
    overlap = 0.8 # the overlap to the target state

    for num_qubits in list_num_qubits:
        
        # calculating the required parameters
        if num_qubits != 1:
            num_params = 2 * num_qubits + 4 * (num_qubits-1) * cir_depth
        else:
            num_params = 2 + 2 * num_qubits * cir_depth
        params = np.random.random(num_params) * 2 * np.pi
        argnums = list(range(num_params-num_params_differentiated, num_params))
        cir_matrix = tensorcircuit_Alternating_Layer_ansatz(num_qubits, cir_depth, params).matrix()

        print(f'num_qubits = {num_qubits}')
        print(f'num_params = {num_params_differentiated}')

        gradient_norm = []
        hessian_l_eachq = []
        for _ in range(num_sample_states):
            target_state = random_state_fixed_overlap(cir_matrix, overlap)

            # get gradients
            grad_func = get_grad_func(num_qubits, cir_depth, target_state, argnums)
            sample_gradient = np.array(grad_func(*params))
            gradient_norm.append(np.linalg.norm(sample_gradient))
        
            # get hessian matrix
            hessian_func = get_hessian_func(num_qubits, cir_depth, target_state, argnums)
            sample_hessian = np.array(hessian_func(*params))
            hessian_l_eachq.append(sample_hessian)

        # save the files (Caution: remember to change directory)
        np.save(f"./data_qubitx/gradient_norm_num_qubit{num_qubits}_depth{5}_sample{num_sample_states}_diff{num_params_differentiated}_overlap{overlap}_bottom", gradient_norm)
        np.save(f"./data_qubitx/hessian_l_eachq_num_qubit{num_qubits}_depth{5}_sample{num_sample_states}_diff{num_params_differentiated}_overlap{overlap}_bottom", hessian_l_eachq)
