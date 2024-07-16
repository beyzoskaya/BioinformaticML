
import numpy as np
import matplotlib.pyplot as plt

def energy_function(s, mean, covariance):
    diff = s - mean
    return np.dot(diff, np.dot(np.linalg.inv(covariance), diff))

def metropolis_algorithm(initial_state, num_steps, mean, covariance, temperature=1.0):
    current_state = initial_state
    states = [current_state]
    energies = [energy_function(current_state, mean, covariance)]

    for _ in range(num_steps):
        proposed_state = current_state + np.random.multivariate_normal(mean=np.zeros_like(current_state), cov=np.eye(len(current_state)))

        # calculate the energy difference ∆ijE
        delta_energy = energy_function(proposed_state, mean, covariance) - energy_function(current_state, mean, covariance)

        # calculate acceptance probability rij
        if delta_energy <= 0:
            acceptance_prob = 1.0
        else:
            acceptance_prob = np.exp(-delta_energy / temperature)

        # accept or reject the proposed state based on acceptance probability
        if np.random.rand() < acceptance_prob:
            current_state = proposed_state

        states.append(current_state)
        energies.append(energy_function(current_state, mean, covariance))

    return np.array(states), np.array(energies)

mean = np.array([0.0, 0.0])  # Mean vector
covariance = np.array([[1.0, 0.5], [0.5, 1.0]])  # Covariance matrix
initial_state = np.array([0.0, 0.0])  # Initial state vector
num_steps = 1000  # Number of iterations
temperature = 1.0  # Temperature parameter

states, energies = metropolis_algorithm(initial_state, num_steps, mean, covariance, temperature)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(states[:, 0], label='State x1')
plt.plot(states[:, 1], label='State x2')
plt.title('Metropolis Algorithm: Sampled States')
plt.xlabel('Iteration')
plt.ylabel('State Value')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(energies, label='Energy')
plt.title('Metropolis Algorithm: Energy Function')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.grid(True)

plt.tight_layout()
plt.savefig('energy_function_metropolis.png')
plt.show()
