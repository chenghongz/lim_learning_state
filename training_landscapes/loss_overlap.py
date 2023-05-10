# Content: Training landscapes via sampling p-overlap states 

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import paddle
import paddle_quantum as pq
from paddle_quantum.state import State
from tqdm import tqdm
from paddle_quantum.state import State
from paddle_quantum.loss import StateFidelity
from utils import *

pq.set_dtype('complex128')
pq.set_backend("state_vector")


def calculation_process(cir, param, param_change, input_u, lam_list, overlap):

    # initialise the p-overlap state
    target_state = random_state_fixed_overlap(unitary=input_u.numpy(), overlap=overlap)
    loss_func = StateFidelity(State(target_state))

    loss_l = []

    # direction vector
    normalizer = paddle.norm(param_change, p='fro')       
    
    # update to the specific direction theta + lambda * new_theta
    for lam in lam_list:
        lam_param = []
        for each_param in range(len(param_change)):
            lam_param.append(param[each_param] + lam * param_change[each_param]/normalizer)

        # optimise the param to the new direction
        lam_param = paddle.to_tensor(lam_param)
        cir.update_param(lam_param)

        # calculate loss
        loss = loss_func(cir())
        loss_l.append( 1 - (loss.numpy()[0] ** 2))

    return loss_l


# sample loss fix num_qubit and sample lambda
def loss_sample_lambda(input_cir, input_u, sample_num: int, param, overlap):

    # record for every p-overlap state
    total_loss = []

    # generate new parameter 
    new_param = M_direction_generator(param, len(param))

    # sampling loss
    for _ in tqdm(range(sample_num)):
        
        # lambda range
        lam_range = np.linspace(-np.pi, np.pi, 100)

        loss_l = calculation_process(input_cir, param, new_param, input_u=input_u, lam_list=lam_range, overlap=overlap)

        # save the loss
        loss_l = list(loss_l)
        total_loss.append(loss_l)

    # Caution: change directory for saving files
    np.savetxt(f"./data1/loss_overlap_n{N}_p{overlap}_sub_depth{depth}_sample{sample_num}.txt", total_loss)
    return


if __name__ == "__main__":
    
    np.random.seed(0)

    depth = 2 # Circuit Depth 
    p_sample = 200 # Nubmer of sampled p-overlap state
    overlap = 0.8 # Overlap to the target state
    
    gradient = []
    variance = []

    # create U(theta) with overlap p to the target state
    N = 4
    input_cir, input_state, input_param, input_u, init_theta = u_theta(N, depth)

    # sample loss wrt different theta value
    loss_sample_lambda(input_cir = input_cir, input_u = input_u, sample_num=p_sample, param = input_param, overlap = overlap)