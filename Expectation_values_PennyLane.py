import timeit
import numpy as np
import itertools as it
import copy
import random
import pennylane as qml

from scipy.optimize import fsolve
import Generating_Problems as Generator
from AnsatzGenerator import QAOAAnsatz
from OrdinaryQAOA import train_qaoa

class ExpectationValues():
    """
    :param problem: input problem
    this class is responsible for the whole RQAOA procedure
    """
    def __init__(self, problem):
        self.problem = problem
        self.energy = None
        self.best_energy = None
        self.gamma = 0
        self.beta = 0
        self.fixed_correl = []
        self.max_exp_dict = {}
    

    def optimize(self,strategy,no_correlation,temperature):
        exp_value_coeffs = [] 
        exp_value_signs = [] 
        exp_values = [] 
        self.max_exp_dict = {}
        rounding_list = []
        
        # The following is a very inefficient way to do it. For who ever reading this, I am so sorry, please optimize it further 
        for index in range(1, len(self.problem.matrix)):
            self.max_exp_dict[frozenset({index})] = 0           
        for index_large in range(1, len(self.problem.matrix)):
            for index_small in range(1, index_large):
                if self.problem.matrix[index_large, index_small] != 0:
                    self.max_exp_dict[frozenset({index_large, index_small})] = 0
        rounding_element = list([matrix_index,exp] for matrix_index, exp in self.max_exp_dict.items())
        for element in rounding_element: 
            if len(element[0]) == 2: 
                index_large = list(element[0])[0] 
                index_small = list(element[0])[1] 
                exp_value_coeffs.append([self.problem.position_translater[index_large],self.problem.position_translater[index_small]])
            else: 
                index = list(element[0])[0]
                exp_value_coeffs.append([ self.problem.position_translater[index]])
        # 
        
        
        ansatz_kwargs = {'num_qubits':self.problem.num_qubits, 'num_layers':1, 'wires':list(self.problem.graph.nodes)}
        history = train_qaoa(self.problem.H_driver, self.problem.H_mixer, ansatz_kwargs, meas_terms=self.problem.pauli_words)
        self.gamma = history['gamma'][-1]
        self.beta = history['beta'][-1]
        self.energy = history['energy'][-1]
        exp_values = history['measurement'][-1] 
        
        for index,value in enumerate(exp_values): 
            exp_value_signs.append( np.sign(element[1]).astype(int) )
        return exp_value_coeffs, exp_value_signs, exp_values

    def brute_force(self):
        """calculate optimal solution of the remaining variables (according to the remaining
        optimization problem) brute force"""
        x_in_dict = {}
        brute_forced_solution = {}
        count = 0
        single_energy_vector = copy.deepcopy(self.problem.matrix.diagonal())
        correl_energy_matrix = copy.deepcopy(self.problem.matrix)
        np.fill_diagonal(correl_energy_matrix, 0)

        for iter_var_list in it.product([-1, 1], repeat=(len(self.problem.position_translater)-1)):
            vec = np.array([0])
            vec = np.append(vec, iter_var_list)
            E_current = self.problem.calc_energy(vec, single_energy_vector, correl_energy_matrix)

            for i in range(1, len(vec)):
                x_in_dict[self.problem.position_translater[i]] = iter_var_list[i-1]
            if count == 0:
                E_best = copy.deepcopy(E_current)
                brute_forced_solution = copy.deepcopy(x_in_dict)
                count += 1
            if float(E_current) < float(E_best):
                brute_forced_solution = copy.deepcopy(x_in_dict)
                E_best = copy.deepcopy(E_current)
        return brute_forced_solution
    
    # TODO: in the future, maybe add QAOA with suboptimal parameters.