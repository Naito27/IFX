import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import copy
# import Calculating_Expectation_Values as Expectation_Values
import Expectation_values_PennyLane as Expectation_Values

from Generating_Problems import MIS
import networkx as nx
import random 
from copy import deepcopy 
from classical_solver import find_mis

class QIRO(Expectation_Values.ExpectationValues):
    """
    :param problem_input: The problem object that shall be solved
    :param nc: size of remaining subproblems that are solved by brute force
    This class is responsible for the whole QIRO procedure; the output represents the optimized bitstring solution in the form
    of a dictionary as well as a list of optimal parameters from each elimination step
    """

    def __init__(self, problem_input, nc, strategy,no_correlation,temperature, radius = 2):
        super().__init__(problem=problem_input)
        # let us use the problem graph as the reference, and this current graph as the dynamic
        # object from which we will eliminate nodes:
        self.graph = copy.deepcopy(self.problem.graph)
        self.nc = nc
        self.assignment = []
        self.solution = []
        self.strategy = strategy
        self.no_correlation = no_correlation
        self.temperature = temperature
        self.radius = radius

    
    def update_single(self, variable_index, exp_value_sign):
        """Updates Hamiltonian according to fixed single point correlation"""
        node = variable_index - 1
        fixing_list = []
        assignments = []
        # if the node is included in the IS we remove its neighbors
        if exp_value_sign == 1:
#             ns = copy.deepcopy(self.graph.neighbors(node))
            ns = set(self.graph.neighbors(node))
            for n in ns:
                self.graph.remove_node(n)
                fixing_list.append([n + 1])
                assignments.append(-1)
        
        # in any case we remove the node which was selected by correlations:
        self.graph.remove_node(node)
        fixing_list.append([variable_index])
        assignments.append(exp_value_sign)

        # reinitailize the problem object with the new, updated, graph:
        self.problem = MIS(self.graph, self.problem.alpha)

        return fixing_list, assignments
    
    def update_correlation(self, variables, exp_value_sign):
        """Updates Hamiltonian according to fixed two point correlation -- RQAOA (for now)."""
        
        #     """This does the whole getting-of-coupled-vars mumbo-jumbo."""
        fixing_list = []
        assignments = []
        if exp_value_sign == 1:
            # if variables are correlated, then we set both to -1 
            # (as the independence constraint prohibits them from being +1 simultaneously). 
            for variable in variables:
                fixing_list.append([variable])
                assignments.append(-1)
                self.graph.remove_node(variable - 1)                
        else:
            # we remove the things we need to remove are the ones connected to both node, which are not both node.
            mutual_neighbors = set(self.graph.neighbors(variables[0] - 1)) & set(self.graph.neighbors(variables[1] - 1))
            fixing_list = [[n + 1] for n in mutual_neighbors]
            assignments = [-1] * len(fixing_list)
            for n in mutual_neighbors:
                self.graph.remove_node(n)

        # reinitailize the problem object with the new, updated, graph:
        self.problem = MIS(self.graph, self.problem.alpha)

        return fixing_list, assignments
    

    def prune_graph(self):
        """Prunes the graph by removing all connected components that have less than nc nodes. The assignments are determined
        to be the maximum independent sets of the connected components. The self.graph is updated correspondingly."""

        # get connected components
        connected_components = copy.deepcopy(list(nx.connected_components(self.graph)))
        prune_assignments = {}
        for component in connected_components:
            if len(component) <= self.nc:
                subgraph = self.graph.subgraph(component)
                _, miss = find_mis(subgraph)
                prune_assignments.update({n: 1 if n in miss[0] else -1 for n in subgraph.nodes}) 

        # remove component from graph
        for node in prune_assignments.keys():
            self.graph.remove_node(node)

        self.problem = MIS(self.graph, self.problem.alpha)

        fixing_list = [[n + 1] for n in sorted(prune_assignments.keys())]
        assignments = [prune_assignments[n] for n in sorted(prune_assignments.keys())]

        return fixing_list, assignments
    
    """ These are failed strategy """
#     def inverse_degree(self, exp_value_coeffs,exp_values): 
#         copy_exp_values = deepcopy(exp_values)
#         for index in range(len(exp_value_coeffs)):
#             coeff = exp_value_coeffs[index] 
#             match len(coeff):
#                 case 2:
#                     node_a = coeff[0] - 1 
#                     node_b = coeff[1] - 1 
#                     common_neighbors = set(self.graph[node_a]) & set(self.graph[node_b])
#                     weight = len(common_neighbors) + 1e-4
#                     copy_exp_values[index] = copy_exp_values[index]/weight
# #                     copy_exp_values[index] = copy_exp_values[index]/(self.graph.degree[node_a] + self.graph.degree[node_b]) 
#                 case 1: 
#                     node = coeff[0] - 1 
#                     copy_exp_values[index] = copy_exp_values[index]/(self.graph.degree[node] + 1e-4)
#         chosen_index = np.argmax(np.abs(copy_exp_values))
#         return chosen_index
    
#     def min_degree(self,exp_value_coeffs, exp_values):
#         copy_exp_values = deepcopy(exp_values)
#         minimum_degree = self.find_minimum_degree()
#         for index in range(len(exp_value_coeffs)):
#             coeff = exp_value_coeffs[index] 
#             match len(coeff):
#                 case 2:
#                     node_a = coeff[0] - 1 
#                     node_b = coeff[1] - 1 
#                     common_neighbors = set(self.graph[node_a]) & set(self.graph[node_b])
#                     copy_exp_values[index] = copy_exp_values[index]*np.tanh(minimum_degree/(len(common_neighbors)+1))/np.tanh(1) 
#                 case 1: 
#                     node = coeff[0] - 1 
#                     copy_exp_values[index] = copy_exp_values[index]*np.tanh(minimum_degree/(self.graph.degree[node]+0.1))/np.tanh(1)
#         chosen_index = np.argmax(np.abs(copy_exp_values))
#         return chosen_index        
#     def find_minimum_degree(self):
#         minimum_degree = float('inf')  # Initialize with infinity
#         for node in self.graph.nodes():
#             degree = self.graph.degree(node)
#             minimum_degree = min(minimum_degree, degree)
#         return minimum_degree    
    
    
    
    def probability_inference(self, exp_value_coeffs, exp_values): 
        """ This strategy is based on probability inference."""
        copy_exp_values = deepcopy(exp_values) 
        coeff_value_dict = self.coeff_value_dict
        for index,value in enumerate(exp_values):
            value = exp_values[index] 
            coeff = exp_value_coeffs[index] 
            match [len(coeff), np.sign(value).astype(int)]:
                case [1,1]:
                    node = coeff[0] - 1
                    neighbor_array = set(self.graph.neighbors(node))
                    p_total = 1/2**(1+len(neighbor_array))
                    p_total += value/2**(1+len(neighbor_array))
                    for nn1 in neighbor_array:
                        one_point = coeff_value_dict.get(tuple([nn1+1]),0)
                        p_total -= one_point/2**(1+len(neighbor_array))
                        two_point = coeff_value_dict.get(tuple(sorted([node+1,nn1+1])),0)
                        p_total -= two_point/2**(1+len(neighbor_array))
                        for nn2 in neighbor_array:
                            two_point = coeff_value_dict.get(tuple(sorted([nn2+1,nn1+1])),0)/2 
                            p_total += two_point/2**(1+len(neighbor_array))
                    copy_exp_values[index] = max(p_total,0)   
                    
                case [1,-1]:
#                     node = coeff[0] - 1
#                     p_node = (1-value)/2 
#                     p_total = p_node 
#                     copy_exp_values[index] = p_total
                    node = coeff[0] - 1
                    neighbor_array = set(self.graph.neighbors(node))
                    p_total = 1/2**(1+len(neighbor_array))
                    p_total += value/2**(1+len(neighbor_array))
                    for nn1 in neighbor_array:
                        one_point = coeff_value_dict.get(tuple([nn1+1]),0)
                        p_total -= one_point/2**(1+len(neighbor_array))
                        two_point = coeff_value_dict.get(tuple(sorted([node+1,nn1+1])),0)
                        p_total -= two_point/2**(1+len(neighbor_array))
                        for nn2 in neighbor_array:
                            two_point = coeff_value_dict.get(tuple(sorted([nn2+1,nn1+1])),0)/2 
                            p_total += two_point/2**(1+len(neighbor_array))
                    copy_exp_values[index] = 1-max(p_total,0)   
                    
                case [2, 1]: 
                    node_1 = coeff[0] - 1 
                    node_2 = coeff[1] - 1 
                    p_total = 1/4 
                    one_point = coeff_value_dict.get(tuple([node_1+1]),0)
                    p_total -= one_point/4
                    one_point = coeff_value_dict.get(tuple([node_2+1]),0)
                    p_total -= one_point/4
                    two_point = coeff_value_dict.get(tuple(sorted([node_1+1,node_2+1])),0)
                    p_total += two_point/4 
                    copy_exp_values[index] = p_total
                    
                    
                case [2,-1]: 
                    node_1 = coeff[0] - 1 
                    node_2 = coeff[1] - 1 
                    mutual_neighbors = set(self.graph.neighbors(node_1)) & set(self.graph.neighbors(node_2))
                    p_total = (1-value)/2 
                    for nn in mutual_neighbors:
                        term_a = coeff_value_dict.get(tuple(sorted([node_1+1,nn+1])),0 )
                        term_b = coeff_value_dict.get(tuple(sorted([node_2+1,nn+1])),0 )
                        prob_a = (1+term_a)/2 
                        prob_b = (1+term_b)/2 
                        p_total = p_total*(prob_a*(1-prob_b) + prob_b*(1-prob_a))
#                     copy_exp_values[index] = max(p_total,0)
        chosen_index = np.argmax(np.abs(copy_exp_values))
        return chosen_index 
    
    def picking_correlation_functions(self,exp_value_coeffs,exp_value_signs,exp_values): 
        """ This function is used to pick the one-point or two-point functions based the strategy. It takes into account sphere of influence based on radius of P-QAOA  """
        
        # This graph is a copy of the problem graph in which we used to keep track of which nodes has been chosen so that we can remove the sphere of influence of that nodes from the copy_graph 
        copy_graph = deepcopy(self.graph)
        copy_exp_value_coeffs = [] 
        copy_exp_value_signs = [] 
        copy_exp_values = [] 
        counter = self.no_correlation
        
        if (self.strategy == "Greedy"):
            for index,value in enumerate(exp_values):
                if len(exp_value_coeffs[index]) == 1:
                    exp_value_signs[index] = 1
                    exp_values[index] = (1+value)/2 
                else:
                    exp_value_signs[index] = -1 
                    exp_values[index] = (1-value)/2 
            
        while counter > 0 and not nx.is_empty(copy_graph): 
            # We pick out one correlation function for each loop based on the chosen strategy 
            match self.strategy: 
                case "Max":
                    chosen_index = np.argmax(np.abs(exp_values))                
                case "Soft_Max":    
                    weight_vector = [np.exp(abs(value)*self.temperature) for value in exp_values]
                    weight_vector = weight_vector/np.sum(weight_vector) 
                    chosen_index = np.random.choice(range(len(exp_values)), 1, False, weight_vector)[0] 
#                 case "Inverse_Degree": 
#                     chosen_index = self.inverse_degree(exp_value_coeffs,exp_values) 
#                 case "Min_Degree":
#                     chosen_index = self.min_degree(exp_value_coeffs, exp_values)
                case "Probability_Inference":
                    chosen_index = self.probability_inference(exp_value_coeffs,exp_values)
                case "Greedy": 
                    chosen_index = np.argmax(np.abs(exp_values)) 
            chosen_coeff = exp_value_coeffs[chosen_index] 
            chosen_sign = exp_value_signs[chosen_index] 
            chosen_value = exp_values[chosen_index] 
            
            # Appending the chosen element 
            copy_exp_value_coeffs.append(chosen_coeff) 
            copy_exp_value_signs.append(chosen_sign) 
            copy_exp_values.append(chosen_value) 
            

            # The following mumbo jumbo is to deal with sphere of influence of each node in p-QAOA 
            nodes_in_radius = set()  
            for label in chosen_coeff: 
                node = label - 1
                nodes_in_radius = nodes_in_radius.union( set( nx.ego_graph(copy_graph,node,distance = self.radius) ) )
            copy_graph.remove_nodes_from(nodes_in_radius)
            
            # Remove all the one-point and two-points correlations related to the nodes_in_radius set 
            remove_index_list = [index for index, coeff in enumerate(exp_value_coeffs) if any( (label-1) in nodes_in_radius for label in coeff)]
            
            exp_value_coeffs = [coeff for index,coeff in enumerate(exp_value_coeffs) if index not in remove_index_list]
            exp_value_signs = [sign for index,sign in enumerate(exp_value_signs) if index not in remove_index_list]            
            exp_values = [value for index,value in enumerate(exp_values) if index not in remove_index_list]       
            counter -= 1 
                
        return copy_exp_value_coeffs, copy_exp_value_signs, copy_exp_values 
    
    def picking_back_up(self, exp_value_coeffs, exp_values):
        """ This function is used to pick a one-point correlation so we always have a back-up in case that the same negative two-point functions are picked consecutively"""
        number_of_nodes = self.graph.number_of_nodes()
        abs_exp_values = np.abs(exp_values)
        chosen_index = np.argmax(abs_exp_values[:number_of_nodes])
        back_up_element = [exp_value_coeffs[chosen_index], exp_values[chosen_index]]
        return back_up_element
    
    def statistic_approach(self,exp_value_coeffs,exp_value_signs,exp_values):
        """ This approach is based on Goemans-Williamson algorithm. From the one-point and two-point functions, we can calculate the mean and covariance matrix. We assume that the underlying assumption is of the multivariables gaussian distribution N(mu,M) """
        # Covariance matrix initialization 
        no_nodes = self.graph.number_of_nodes()
        node_index_dict = dict( zip(self.graph.nodes(),range(no_nodes)) ) 
        single_term_dict = dict(zip(map(tuple, exp_value_coeffs[:no_nodes]), exp_values[:no_nodes]))
        M = np.zeros([no_nodes, no_nodes]) 
        mean_array = np.array(exp_values[:no_nodes])
        # Copy of exp_arrays 
        copy_exp_value_coeffs = [] 
        copy_exp_value_signs = [] 
        copy_exp_values = [] 
        
        # Filling the covariance matrix 
        for index,coeff in enumerate(exp_value_coeffs):
            match len(coeff):
                case 1:
                    node = coeff[0] - 1
                    index_i = node_index_dict[node] 
                    value = exp_values[index] 
                    M[index_i,index_i] = 1-value**2 
                case 2:
                    node_1 = coeff[0] - 1
                    node_2 = coeff[1] - 1
                    index_i = node_index_dict[node_1] 
                    index_j = node_index_dict[node_2] 
                    value_ij = exp_values[index] 
                    value_i = single_term_dict[ (coeff[0],) ]
                    value_j = single_term_dict[ (coeff[1],) ]
                    M[index_i,index_j] = value_ij - value_i*value_j
                    M[index_j,index_i] = value_ij - value_i*value_j
            
        # Sampling 
        sample = np.random.multivariate_normal(mean_array,M) 
        # Reconstruct exp_array based on the sample   
        for index,coeff in enumerate(exp_value_coeffs): 
            match len(coeff):
                case 1:
                    node = coeff[0] - 1 
                    index_i = node_index_dict[node] 
                    copy_exp_value_coeffs.append(coeff) 
                    value = sample[index_i]
                    copy_exp_values.append(value) 
                    copy_exp_value_signs.append(np.sign(value).astype(int)) 
                case 2:
                    node_1 = coeff[0] - 1 
                    node_2 = coeff[1] - 1 
                    index_i = node_index_dict[node_1] 
                    index_j = node_index_dict[node_2]       
                    copy_exp_value_coeffs.append(coeff) 
                    value = sample[index_i]*sample[index_j] 
                    copy_exp_values.append(value) 
                    copy_exp_value_signs.append(np.sign(value).astype(int)) 
                                          
        return copy_exp_value_coeffs,copy_exp_value_signs,copy_exp_values 
    
    def execute(self, energy='best'):
        """Main QIRO function which produces the solution by applying the QIRO procedure."""
        self.opt_gamma = []
        self.opt_beta = []
        self.fixed_correlations = []
        step_nr = 0
        self.coeff_value_dict = {}

        while self.graph.number_of_nodes() > 0:
            step_nr += 1
#             print(f"Step: {step_nr}. Number of nodes: {self.graph.number_of_nodes()}.")
#             nx.draw(self.graph, with_labels = True)
#             plt.show()
            fixed_variables = []            
            
            # Obtain ALL the one-point and two-point correlation functions 
            exp_value_coeffs, exp_value_signs, exp_values = self.optimize()
            
            self.coeff_value_dict = {tuple(sorted(coeff)): value for coeff,value in zip(exp_value_coeffs,exp_values)}
            # Alternating between picking from one-point correlation functions and two point correlation functions 
            number_of_nodes = self.graph.number_of_nodes()
            if (step_nr % 2) != 0:
                exp_value_coeffs = exp_value_coeffs[:number_of_nodes]
                exp_value_signs = exp_value_signs[:number_of_nodes]
                exp_values = exp_values[:number_of_nodes]
            else:
                exp_value_coeffs = exp_value_coeffs[number_of_nodes:]
                exp_value_signs = exp_value_signs[number_of_nodes:]
                exp_values = exp_values[number_of_nodes:]
                
            # Picking out back up element in case no variable is fixed (dont need it if we are alternating between one-point and two-point correlation functions
#             back_up_element = self.picking_back_up(exp_value_coeffs, exp_values) 

  
            # Picking out correlations function based on self.strategy
            exp_value_coeffs, exp_value_signs, exp_values = self.picking_correlation_functions(exp_value_coeffs,exp_value_signs,exp_values) 
                     
            for index in range(len(exp_value_coeffs)): 
                exp_value_coeff = exp_value_coeffs[index]
                exp_value_sign = exp_value_signs[index]
                exp_value = exp_values[index]
                if len(exp_value_coeff) == 1: 
                    holder_fixed_variables, assignments = self.update_single(*exp_value_coeff,exp_value_sign)
                    fixed_variables += holder_fixed_variables 
                    for var, assignment in zip(holder_fixed_variables,assignments): 
                        self.fixed_correlations.append([var,int(assignment),exp_value])
                else:
                    holder_fixed_variables, assignments = self.update_correlation(exp_value_coeff,exp_value_sign) 
                    fixed_variables += holder_fixed_variables 
                    for var, assignment in zip(holder_fixed_variables,assignments):
                        self.fixed_correlations.append([var,int(assignment),exp_value])

            # perform pruning.
            pruned_variables, pruned_assignments = self.prune_graph()
            for var, assignment in zip(pruned_variables, pruned_assignments):
                if var is None:
                    raise Exception("Variable to be eliminated is None. WTF?")
                self.fixed_correlations.append([var, assignment, None])
            fixed_variables += pruned_variables
            
            # Backup procedure in case no variable is fixed 
#             if len(fixed_variables) == 0: 
#                 backup_coeff = back_up_element[0]
#                 backup_sign  = np.sign(back_up_element[1]).astype(int) 
#                 backup_value = back_up_element[1]
# #                 print(f"No fixed variables have been found, attempting with {backup_coeff}. Sign: {backup_sign}. Value: {backup_value}")
#                 holder_fixed_variables, assignments = self.update_single(*backup_coeff,backup_sign)
#                 fixed_variables += holder_fixed_variables 
#                 for var, assignment in zip(holder_fixed_variables,assignments): 
#                     self.fixed_correlations.append([var,int(assignment),exp_value])
                        
        solution = [var[0] * assig for var, assig, _ in self.fixed_correlations]
        sorted_solution = sorted(solution, key=lambda x: abs(x))
        # print(f"Solution: {sorted_solution}")
        self.solution = np.array(sorted_solution).astype(int)

