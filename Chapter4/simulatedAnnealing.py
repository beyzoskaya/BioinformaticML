import numpy as np
import matplotlib.pyplot as plt

# energy function is objective to minimize f(x,y) = x^2 + y^2
def energy_function(state):
    x,y = state
    return x**2 + y**2

def generate_neighbor(current_state,scale=1.0):
    print(np.random.randn(*current_state.shape))
    return current_state + scale * np.random.randn(*current_state.shape)


# geometric cooling = T = µ * T
def geometric_cooling(current_temperature, cooling_rate=0.9):
    return current_temperature*cooling_rate

def simulated_annealing(initial_state,initial_temperature,cooling_schedule, energy_function,max_iterations=1000):
    current_state = initial_state
    current_temperature = initial_temperature
    best_state = current_state
    best_energy = energy_function(best_state)
    energies = [best_energy]

    for _ in range(max_iterations):
        proposed_state = generate_neighbor(current_state)
        energy_current = energy_function(current_state)
        energy_proposed = energy_function(proposed_state)
        delta_energy = energy_proposed-energy_current

        if delta_energy <=0:
            acceptance_prob = 1.0
        else:
            acceptance_prob = np.exp(-delta_energy/current_temperature)
        
        if np.random.rand() < acceptance_prob:
            current_state = proposed_state
            energy_current = energy_proposed

        if energy_current < best_energy:
            best_state = current_state
            best_energy = energy_current
        
        current_temperature = cooling_schedule(current_temperature)
        energies.append(best_energy)
    
    return best_state,energies

initial_state = np.array([1.0, -2.0])
initial_temperature = 10.0
max_iterations = 1000

best_state,energies = simulated_annealing(initial_state,initial_temperature,geometric_cooling,energy_function,max_iterations)

print("Best state found:", best_state)
print("Minimum energy found:", energy_function(best_state))

plt.figure()
plt.plot(energies)
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.title('Energy Convergence')
plt.grid(True)
plt.savefig('energy_graph_simulated_annealing.png')
plt.show()

        

    
    
