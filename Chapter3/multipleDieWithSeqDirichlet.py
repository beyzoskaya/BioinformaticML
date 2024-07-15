import numpy as np
import random

def generate_random_sequences(length, num_sequences):
    sequences = []
    for _ in range(num_sequences):
        seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=length))
        sequences.append(seq)
    return sequences
K = 100  # Number of sequences
N = 50   # Length of each sequence
sequences = generate_random_sequences(N, K)

"""

sequences = [
    "ACGT",
    "ACGG",
    "TCGT",
    "TCGA"
]

K = len(sequences) 
N = len(sequences[0])

"""


A = ['A', 'C', 'G', 'T']

counts = {}

for i in range(N):
    counts[i] = {}
    for letter in A:
        counts[i][letter]=0

for seq in sequences:
    for i,letter in enumerate(seq):
        counts[i][letter] += 1

print("Counts Dictionary:")
for position, letter_counts in counts.items():
    print(f"Position {position}: {letter_counts}")

# Dirichlet prior parameters
alpha = {letter: 2 for letter in A} 
# alpha = {'A': 1, 'C': 1, 'G': 1, 'T': 1}

posterior_counts = {}
for i in range(N):
    posterior_counts[i] = {}

    for letter in A:
        posterior_counts[i][letter]=counts[i][letter]+alpha[letter]

print("\nPosterior Counts Dictionary:")
for position, letter_counts in posterior_counts.items():
    print(f"Position {position}: {letter_counts}")

probabilities = {}
for i in range(N):
    probabilities[i]={}
    total_count = sum(posterior_counts[i].values())
    for letter in A:
        probabilities[i][letter] = (posterior_counts[i][letter] - alpha[letter]) / (total_count - len(A))

print(f"Total count: {total_count}")
print(f"Normalized total count: {total_count - len(A)}")
        # remove this pseudocount to get the adjusted count for each letter --> subtracting 1 means 
        # normalizing all counts to the total count with substraction of len(A)

print("\nMAP Estimates (Probabilities) Dictionary:")
for position, letter_probs in probabilities.items():
    print(f"Position {position}: {letter_probs}")

log_likelihood = 0
likelihood = 1
for i in range(N):
    for letter in A:
        if counts[i][letter] > 0:
            log_likelihood += counts[i][letter] * np.log(probabilities[i][letter])
            likelihood *= probabilities[i][letter] ** counts[i][letter]

exp_log_likelihood = np.exp(log_likelihood)

print(f"\nLog-Likelihood: {log_likelihood}")
print(f"Likelihood from exp(log-likelihood): {exp_log_likelihood}")
print(f"Likelihood: {likelihood}")

print(f"--------------------------------")

# For comparing two different priors

# Check counts
print("Counts Dictionary:")
for position, letter_counts in counts.items():
    print(f"Position {position}: {letter_counts}")

# Calculate probabilities with uniform prior
probabilities_uniform = {}
for i in range(N):
    probabilities_uniform[i] = {}
    for letter in A:
        probabilities_uniform[i][letter] = counts[i][letter] / K

# Calculate probabilities with Dirichlet prior
probabilities_dirichlet = {}
for i in range(N):
    probabilities_dirichlet[i] = {}
    total_count = sum(posterior_counts[i].values())
    for letter in A:
        probabilities_dirichlet[i][letter] = (posterior_counts[i][letter] - 1) / (total_count - len(A))

# Verify likelihood calculation
def calculate_log_likelihood(probabilities):
    log_likelihood = 0
    for i in range(N):
        for letter in A:
            if counts[i][letter] > 0:
                log_likelihood += counts[i][letter] * np.log(probabilities[i][letter])
    return log_likelihood

log_likelihood_uniform = calculate_log_likelihood(probabilities_uniform)
log_likelihood_dirichlet = calculate_log_likelihood(probabilities_dirichlet)

print(f"Log-Likelihood (Uniform Prior): {log_likelihood_uniform}")
print(f"Log-Likelihood (Dirichlet Prior with alpha=1): {log_likelihood_dirichlet}")