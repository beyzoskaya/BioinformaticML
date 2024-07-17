"""
In the book, there is no special informations about implementation of Genetic Algorithm.
I implemented Genetic Algorithm to TSP.
"""

import numpy as np
import random
import matplotlib.pyplot as plt

#distance_matrix = np.array([
#    [0, 10, 15, 20],
#    [10, 0, 35, 25],
#    [15, 35, 0, 30],
#    [20, 25, 30, 0]
#])

def generate_distance_matrix(num_cities):
    np.random.seed(42)
    matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
    matrix = (matrix + matrix.T) // 2 
    np.fill_diagonal(matrix, 0)
    return matrix

num_cities = 10
distance_matrix = generate_distance_matrix(num_cities)
print(distance_matrix)
"""
printed distance matrix: 

[[ 0 28 75 69 21 57 76 48 70 79]
 [28  0 30 75 35 57 41 55 17 53]
 [75 30  0 54 41 65 92 65 74 43]
 [69 75 54  0 50 59 80 43 75 85]
 [21 35 41 50  0 29 36 56 97 69]
 [57 57 65 59 29  0 64 66 62 33]
 [76 41 92 80 36 64  0 27 32 50]
 [48 55 65 43 56 66 27  0 82 23]
 [70 17 74 75 97 62 32 82  0 72]
 [79 53 43 85 69 33 50 23 72  0]]
"""

def calculate_total_distance(route,distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i+1]]
    total_distance += distance_matrix[route[-1], route[0]]
    return total_distance

def create_initial_population(population_size,num_cities):
    population = []
    for _ in range(population_size):
        individual = np.random.permutation(num_cities)
        #print(f"Individual: {individual}")
        population.append(individual)
    return np.array(population)

def evaluate_population(population,distance_matrix):
    fitness_score = []
    for individual in population:
        fitness = calculate_total_distance(individual,distance_matrix)
        fitness_score.append(fitness)
    return np.array(fitness_score)

def select_parents(population,fitness_scores,num_parents):
    sorted_indices = np.argsort(fitness_scores)
    best_indices = sorted_indices[:num_parents]
    parents = population[best_indices]
    return parents

def crossover(parent1,parent2):
    size = len(parent1) # size of the parents
    start,end = sorted(random.sample(range(size),2)) # random selection of two crossover points
    child = [-1]*size # empty child solution
    child[start:end]=parent1[start:end] # copy the segment from parent1 to child
    ptr = 0 # pointer to track position of parent2
    for i in range(size): # iterate over each position for child
        if child[i] == -1: # has not been filled yet
            while parent2[ptr] in child:
                ptr += 1
            child[i] = parent2[ptr]
    return child

def mutate(individual,mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) -1)
            individual[i],individual[j] = individual[j],individual[i]
    return individual

def genetic_algorithm(distance_matrix,population_size,num_generations,mutation_rate,num_parents):
    num_cities = distance_matrix.shape[0]
    population = create_initial_population(population_size,num_cities)
    best_route = None
    best_distance = float('inf')
    history = []

    for generation in range(num_generations):
        fitness_scores = evaluate_population(population,distance_matrix)
        parents = select_parents(population,fitness_scores,num_parents)
        new_population = []
        for i in range(population_size):
            parent1,parent2 = parents[random.randint(0, num_parents - 1)], parents[random.randint(0, num_parents - 1)]
            child = crossover(parent1,parent2)
            child = mutate(child,mutation_rate)
            new_population.append(child)
        population = np.array(new_population)

        min_distance = np.min(fitness_scores)
        if min_distance < best_distance:
            best_distance = min_distance
            best_route = population[np.argmin(fitness_scores)]
        history.append(best_distance)
        print(f"Generation {generation}: Best distance = {best_distance}")

    return best_route,best_distance,history

population_size = 200
#num_cities = distance_matrix.shape[0]
#print(f"distance matrix shape: {distance_matrix.shape}")
#print(f"distance matrix shape first dimension: {distance_matrix.shape[0]}")
#population = create_initial_population(population_size,num_cities)

num_generations = 500
mutation_rate = 0.02
num_parents = 20

best_route,best_distance,history = genetic_algorithm(distance_matrix,population_size,num_generations,mutation_rate,num_parents)

print(f"Best route found: {best_route}")
print(f"Total distance: {best_distance}")

plt.plot(history)
plt.xlabel('Generation')
plt.ylabel('Distance')
plt.title('Genetic Algorithm Convergence')
plt.grid(True)
plt.savefig('genetic_algorithm_TSP_convergence.png')
plt.show()