"""
process of constructing a Markov chain to sample from a desired probability distribution 𝑃

Objective: sample from a complex probability distribution 𝑃
"""

import numpy as np
import matplotlib.pyplot as plt

# 1D Gaussian with mean 0 and standard deviation 1
# This is the distribution for sampling P(x) in Monte Carlo
def target_distribution(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# Gaussian with mean x and standard deviation 0.5
# auxilary distribution for exploring state space of target distribution
def proposal_distribution(x):
    return np.random.normal(x, 0.5)

def metropolis_hastings(target, proposal, num_samples, initial_value):
    samples = np.zeros(num_samples)
    samples[0] = initial_value # first sample is an arbitrary initial state
    
    for t in range(1, num_samples): # iterate over all samples
        current = samples[t-1] 
        print(f"Current at iteration {t}: {current}")
        proposed = proposal(current)
        print(f"Proposed at iteration {t}: {proposed}")
        
        acceptance_ratio = target(proposed) / target(current)
        acceptance_probability = min(1, acceptance_ratio)
        
        if np.random.rand() < acceptance_probability:
            samples[t] = proposed
        else:
            samples[t] = current
            
    return samples

num_samples = 10000
initial_value = 0

samples = metropolis_hastings(target_distribution, proposal_distribution, num_samples, initial_value)

x = np.linspace(-4, 4, 1000)
plt.plot(x, target_distribution(x), label='Target Distribution')
plt.hist(samples, bins=50, density=True, alpha=0.6, label='MCMC Samples')
plt.legend()
plt.title('Metropolis-Hastings Sampling')
plt.savefig('metropolisHasting.png')
plt.show()
