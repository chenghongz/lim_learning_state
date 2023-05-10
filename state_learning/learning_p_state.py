# Content: Special example of theorem7 showing loss cannot be optimised

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import paddle
import paddle_quantum as pq
from paddle_quantum.state import State, zero_state
from paddle_quantum.loss import StateFidelity
from utils import *

pq.set_dtype('complex64')
pq.set_backend("state_vector")


def state_learning(input_cir, num_qubits, target_state, itr: int = 200, lr: float = 0.01):

    cir = input_cir
    print(target_state)
    loss_func = StateFidelity(State(target_state))
    opt = paddle.optimizer.Adam(learning_rate = lr, parameters = cir.parameters())

    # training...
    loss_list = []
    for _ in range(itr):
        output_state = cir(zero_state(num_qubits))
        loss = 1 - loss_func(output_state) ** 2
        loss_list.append(loss.item())
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()
    
    return output_state, loss_list


if __name__ == "__main__":
    
    np.random.seed(0)

    depth = 3 # Circuit depth
    sample = 3 # Number of sampled random states
    overlap = 0.2 # Initial overlap
    N = 4 # Number of Qubits
    
    gradient = []
    variance = []

    input_cir, input_state, input_param, input_u, init_theta = u_theta(N, depth)

    for i in (range(sample)):

        # sample target state
        target_state = random_state_fixed_overlap(unitary=input_u.numpy(), overlap=overlap)
        print(f"initial : ", paddle.abs(State(target_state).bra @ input_state.ket))

        # calculate loss
        output_state, loss_list = state_learning(input_cir, N, target_state)

        print(f"final : ", paddle.abs(State(target_state).bra @ output_state.ket))
        np.save("test_loss_list", loss_list)