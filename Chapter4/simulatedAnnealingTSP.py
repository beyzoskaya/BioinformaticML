import numpy as np
import matplotlib.pyplot as plt

distance_matrix = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

def energy_function(route,distance_matrix):
    total_distance = 0
    N = len(route)

    for i in range(N-1):
        total_distance += distance_matrix[route[i], route[i+1]]
    total_distance += distance_matrix[route[N-1],route[0]] # last city visited then turn to the starting position
    return total_distance

def simulated_annealing_tsp(distance_matrix,initial_solution,initial_temperature,cooling_rate,num_iterations):
    current_solution = initial_solution.copy()
    best_solution = current_solution.copy()
    current_energy = energy_function(current_solution,distance_matrix)
    best_energy = current_energy

    temperature = initial_temperature

    all_routes = [current_solution]
    all_energies = [current_energy]

    for iteration in range(num_iterations):
        new_solution = current_solution.copy()
        idx1,idx2 = np.random.choice(len(new_solution), size=2, replace=False)
        new_solution[idx1],new_solution[idx2] = new_solution[idx2],new_solution[idx1]

        new_energy = energy_function(new_solution,distance_matrix)
        delta_energy = new_energy-current_energy

        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy/temperature):
            current_solution = new_solution
            current_energy = new_energy
        
        if current_energy < best_energy:
            best_solution = current_solution
            best_energy = current_energy

        temperature = temperature*cooling_rate

        all_routes.append(current_solution)
        all_energies.append(current_energy)
    return best_solution,best_energy,all_routes,all_energies

initial_solution = np.array([0, 1, 2, 3]) 
initial_temperature = 1000.0
cooling_rate = 0.95
num_iterations = 1000

best_solution, best_energy, all_routes, all_energies = simulated_annealing_tsp(distance_matrix, initial_solution, initial_temperature, cooling_rate, num_iterations)

print(f"Best route found: {best_solution}")
print(f"Total distance: {best_energy}")

print("\nAll evaluated routes and total distances:")
for i in range(len(all_routes)):
    route = all_routes[i]
    distance = all_energies[i]
    print(f"Iteration {i}: Route = {route}, Total distance = {distance}")


plt.figure(figsize=(10, 6))
plt.plot(range(len(all_energies)), all_energies, marker='o', linestyle='-', color='b')
plt.xlabel('Iterations')
plt.ylabel('Total Distance')
plt.title('Simulated Annealing for TSP: Energy Convergence')
plt.grid(True)
plt.tight_layout()
plt.savefig('simulated_annealing_for_TSP_1000_iteration.png')
plt.show()
