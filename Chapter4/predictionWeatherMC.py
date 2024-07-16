import numpy as np
import matplotlib.pyplot as plt

# Probable state of weather
states = ['Sunny','Rainy','Cloudy']
state_index = {state: i for i, state in enumerate(states)}
print(f"State indexes: {state_index}")
num_states = len(states)

# transition table
transition_matrix = np.array([
    [0.8, 0.1, 0.1],  # from Sunny to Sunny, Rainy, Cloudy
    [0.2, 0.6, 0.2],  # from Rainy to Sunny, Rainy, Cloudy
    [0.2, 0.3, 0.5]   # from Cloudy to Sunny, Rainy, Cloudy
])

# target distribution --> equilibrium distribution
def target_distribution(state_probs):
    return state_probs / np.sum(state_probs)

# go for another state randomly with equal prob
def proposal_distribution(current_state):
    return np.random.choice(states)

def metropolis_hastings(initial_state,num_samples):
    samples = [initial_state]
    current_state = initial_state
    state_counts = np.zeros(num_states)
    # state_counts keep track of the number of times each state has been visited 

    for _ in range(num_samples):
        proposed_state = proposal_distribution(current_state)
        current_index = state_index[current_state]
        proposed_index = state_index[proposed_state]

        acceptance_ratio = (
            transition_matrix[current_index,proposed_index]/
            transition_matrix[proposed_index,current_index]
        )
        
        acceptance_probability = min(1,acceptance_ratio)

        if np.random.rand() < acceptance_probability:
            current_state = proposed_state
        
        samples.append(current_state)
        state_counts[state_index[current_state]] +=1 # freq of visit each state 
        # state_index[current_state] index of current state from the state index
        #print(f"Index of current state: {state_index[current_state]}")
        print(f"state_counts[state_index[current_state]]: {state_counts[state_index[current_state]]}")
    
    return samples,state_counts/num_samples

initial_state = 'Sunny'
num_samples = 1000

samples, state_probs = metropolis_hastings(initial_state,num_samples)

plt.figure(figsize=(10, 6))
plt.hist(samples, bins=np.arange(len(states)+1)-0.5, density=True, alpha=0.7, rwidth=0.8)
plt.xticks(range(len(states)), states)
plt.xlabel('Weather States')
plt.ylabel('Probability')
plt.title('Weather State Distribution using Metropolis-Hastings')
plt.savefig('metropolisHastingWeather.png')
plt.show()

print("Estimated state probabilities:")
for state, prob in zip(states, state_probs):
    print(f"{state}: {prob:.3f}")

