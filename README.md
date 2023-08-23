# qiro 
Quantum-Informed Recursive Optimization Algorithms. Original Author: Judi 

I only modify the cost to implement some new features. This is a newer version than the one in my IFX public folder since it has pennylane simulator. 

Contains most of the code; stuff to be added: parallel tempering, simulated annealing. 

### Installation

Set up a new conda environment with the following command:

```
conda create -n qiro python=3.9.13
```
Activate the environment:
```
conda activate qiro
```
Then install dependencies:
```
pip install -r requirements.txt

Currently missing qiskit and pennylane. There is some complication due to they using old version of numpy. 
```


### File descriptions:

#### Testing_Notebook_().ipynb
Jupyter Notebook where I play around with different idea as well as obtaining numerical results. It is a good starting place for you to learn how to use the script 

#### AnsatzGenerator.py 
Contains generation of QAOA. 

#### Generating_Problems.py

Contains generation of MIS, MaxCut, and MAX-2-SAT problems, and transforming them in the correct shape for QIRO. For MIS problem, we also include method to generate Hamiltonian for pennylane simulator 

#### Calculating_Expectation_Values.py

Contains functions for calculating the expectation values of the correlations from p=1 QAOA for MIS and MAX-2-SAT problems. Should in principle work for any quadratic Hamiltonian.

#### Expectation_values_PennyLane.py

Contains functions for calculating the expectation values of the correlations for MIS and MaxCut problems using PennyLane simulator. Could have some bugs due to I was rushing to code it. 

#### OrdinaryQAOA.py 

Contains function to train QAOA 

#### QIRO_MAX_2_SAT.py

Contains the QIRO algorithm for MAX-2-SAT problems.

#### QIRO_MIS.py

Contains the QIRO algorithm for MIS problems.

#### QIRO_MC.py

Contains the QIRO algorithm for MaxCut problems.

#### RQAOA.py
This has been removed and replaced by QIRO_MaxCut.py 

#### aws_quera.py

Contains the code for running the QIRO algorithm to solve MIS on QuEra Aquila (AWS Braket).
### Classical Benchmarks
#### greedy_mis.py

Contains the code for the greedy algorithm for MIS.

#### Parallel_Tempering.py

Contains the code for the parallel tempering algorithm.

#### Simulated_Annealing.py

Contains the code for the simulated annealing algorithm.

