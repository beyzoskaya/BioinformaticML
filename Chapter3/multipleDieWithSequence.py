import numpy as np

sequences = [
    "ACGT",
    "ACGG",
    "TCGT",
    "TCGA"
]

K = len(sequences)  # Number of sequences
N = len(sequences[0])  # Length of each sequence
A = ['A', 'C', 'G', 'T']  # Alphabet

# Initialize counts
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

probabilities = {}
for i in range(N):
    probabilities[i]={}
    for letter in A:
        probabilities[i][letter]=counts[i][letter]/K

print("\nProbabilities Dictionary:")
for position,letter_probs in probabilities.items():
    print(f"Position {position}: {letter_probs}")

log_likelihood = 0
likelihood = 1
for i in range(N):
    for letter in A:
        if counts[i][letter] > 0:
            log_likelihood += counts[i][letter]*np.log(probabilities[i][letter])
            likelihood *= probabilities[i][letter] ** counts[i][letter]
exp_log_likelihood = np.exp(log_likelihood)
print(f"\nLog-Likelihood: {log_likelihood}")
print(f"Likelihood from exp(log-likelihood): {exp_log_likelihood}")
print(f"Likelihood: {likelihood}")

"""
Counts for each position:
    Position 0: 
        A:2  p(A) = 2/4 = 0.5
        T:2  p(T) = 2/4 = 0.5
        C:0  p(C) = 0
        G:0  p(G) = 0
    Position 1:
        C:4  p(C) = 1
        A:0  p(A) = 0
        T:0  p(T) = 0
        G:0  p(G) = 0
    Position 2:
        G:4  p(G) = 1
        A:0  p(A) = 0
        T:0  p(T) = 0
        C:0  p(C) = 0
    Position 3:
        A:1  p(A) = 1/4 = 0.25
        G:1  p(G) = 1/4 = 0.25
        T:2  p(T) = 2/4 = 0.50
        C:0  p(C) = 0
"""
    