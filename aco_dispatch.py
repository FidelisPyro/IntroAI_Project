
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import os


# Path to csv file
"""
Change path to where the generator_cost_emission_coefficients.csv file is located for your computer
"""
# path = "/home/fidelispyro/IntroAI/project/generator_cost_emission_coefficients.csv"

# This works for me (Cody), so hopefully it works for you guys too
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generator_cost_emission_coefficients.csv")

# Read csv file into a pandas dataframe
df = pd.read_csv(path)


class ACO:
    def __init__(self, num_ants, num_iterations, alpha, beta, rho, Q):
        self.num_ants = num_ants                # Number of ants
        self.num_iterations = num_iterations    # Number of iterations
        self.alpha = alpha                      # Influences pheromone trails
        self.beta = beta                        # Influences heuristic information
        self.rho = rho                          # Pheromon evaporation rate
        self.Q = Q                              # Amount of pheromone each ant deposits

    def run(self, df, load_demand):
        # Number of generators
        num_generators = len(df)
        
        # Initialize pheromone trails
        pheromone_trails = np.ones(num_generators)
        
        # Calculate heuristic information (minimize cost)
        eta = 1 / (df['a'] * df['Pmax (MW)']**2 + df['b'] * df['Pmax (MW)'] + df['c'])

        # Initialize best cost and best solution
        best_cost = float('inf')    
        best_solution = None

        for _ in range(self.num_iterations):
            for _ in range(self.num_ants):
                solution = []       # Array of current generator outputs
                total_output = 0    # Current total output of all generators 
                cost = 0            # Current total cost of all generators 

                for gen in range(num_generators):
                    # Possible power levels for current generator
                    power_levels = np.arange(df['Pmin (MW)'][gen], df['Pmax (MW)'][gen] + 1)
                    
                    # Calculate probabilities for each power level
                    probabilities = (pheromone_trails[gen]**self.alpha) * (eta[gen]**self.beta)
                    # Makes sure that the probabilities array has the same length as the power_levels array
                    probabilities = np.ones_like(power_levels) * probabilities   
                    # Normalize probabilities   
                    probabilities = probabilities / probabilities.sum()                    

                    # Select power level based on probabilities
                    selected_level = np.random.choice(power_levels, p=probabilities)
                    solution.append(selected_level)

                    # Adds selected power level to total output and calculates cost of selected power level
                    total_output += selected_level
                    cost += df['a'][gen] * selected_level**2 + df['b'][gen] * selected_level + df['c'][gen]

                # If total output is >= load demand and new cost is less than current best, update bests to current
                if total_output >= load_demand and cost < best_cost:    #### May need to change to >= to = to not have wasted power
                    best_cost = cost
                    best_solution = solution

                # Updates pheromone trails 
                pheromone_trails = (1 - self.rho) * pheromone_trails + self.Q / best_cost

        return best_solution, best_cost

def find_load_range(num, randomize = False):
    """
    Will return a list of load demands to use in the aco function. The values returned will be evely spread across the board by default,
    but if the second parameter is set to true, then the values will be random. The number of values in the returned list will be 
    whatever num is equal to.
    """
    if num <= 1:
        return [283.5]
    
    load_demand_range = []

    if randomize:
        for _ in range(num):
            load_demand_range.append(round(random.uniform(117,283.5), 2))
    else:
        inc_value = (283.5 - 117) / (num-1)
        load_demand = 117
        
        for _ in range(num):
            load_demand_range.append(load_demand)
            load_demand += inc_value

    return load_demand_range

# Initialize ACO with parameters
aco = ACO(num_ants = 40, num_iterations = 1000, alpha = 1, beta = 1, rho = 0.5, Q = 1)

# This is how many times we will run aco on a different load demand
number_of_load_demands = 3
# Do you want random values? Or 5 evenly spread out values?
randomize = True
# Get range of load demands
load_demands = find_load_range(number_of_load_demands, randomize)

for ld in load_demands:
    # Run ACO with set load demand
    best_solution, best_cost = aco.run(df, load_demand = ld)     # 283.5 MW is the total P (MW) of all 30 buses
    print(f"Running ACO using load demand of {ld}:")
    print(f"Best solution: {best_solution}")
    print(f"Best cost: {best_cost}")

""" 
Can still be optimized.
Probably need to try to change load demand to a range of values to find the best solution for each load demand up to 283.5 MW.
117 MW is the minimum output the generators can produce so load demand range should be from 117 MW to 283.5 MW.
I want to run some matplotlib function to create some graphs we can use to show the current results and results of any updates we make.
# """
