# Statistical Analysis of Quantum State Learning Process in Quantum Neural Networks

This GitHub repository contains a demonstration for the main theorem verification and practical variational state learning examples. Please refer to [our paper](to be added) for further information.

## Install Paddle Quantum and Tensorcircuit

The experiment is carried in Paddle Quantum version `2.3.0` and Tensorcircuit version `0.8.0`. To run the codes in this repository, you need to install Paddle Quantum and Tensorcircuit first.

```bash
pip install paddle-quantum==2.3.0
pip install tensorcircuit==0.8.0
```

## File Description

`training_landscapes` demonstrates local minima by sampling from p-overlap states as well as different optimization directions.

`pr_local_minima` provides the code for directly verifying Theorem 2 in terms of the number of qubits, the number of trainable parameters, and overlap information.

`state_learning` provides the code for the varational training of quantum neural networks on the arbitrary states beginning with initial p-overlap states.

`utils.py` includes helper functions.