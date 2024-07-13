import numpy as np
from scipy.special import gammaln
# gammaln is for natural logarithm of gamma function

#Example sequence 
sequence_data = "ACGTCGATCGATCGATCGTAGCTAGCTAGCTGACT"
counts = {
    'A': sequence_data.count('A'),
    'C': sequence_data.count('C'),
    'G': sequence_data.count('G'),
    'T': sequence_data.count('T')
}

A = ['A', 'C', 'G', 'T'] # nucleotids in the formula (summation over all A)
n = np.array([counts[x] for x in A]) # observed count of nucleotid x (nx in book)
N = np.sum(n)

def lagrangian(p, n, N, lmbda):
    return -np.sum(n * np.log(p)) - lmbda * (1 - np.sum(p))
def find_pX(n, N, lmbda):
    return n / lmbda # partial derivate ∂L/∂pX = 0 gives us pX = nX/λ pX--> calculated probs under uniform priors
def find_lambda(n):
    return np.sum(n)
lmbda = find_lambda(n)
p_optimal = find_pX(n, N, lmbda)

p_A, p_C, p_G, p_T = 0.25, 0.25, 0.25, 0.25 # example priors for uniform case
p = {'A': p_A, 'C': p_C, 'G': p_G, 'T': p_T}
likelihood = np.prod([p[X]**counts[X] for X in A])
print("Likelihood P(D|M):", likelihood)

# negative log likelihood calculation
negative_log_likelihood = -np.sum([counts[X] * np.log(p[X]) for X in A])
print("Negative Log-Likelihood:", negative_log_likelihood)

alpha = 1.0
q = np.array([1/4] * 4)  # Uniform prior


def dirichlet_log_normalization(alpha, q):
    return np.sum(gammaln(alpha * q)) - gammaln(np.sum(alpha * q))

log_prior = dirichlet_log_normalization(alpha, q)

print("Negative Log-Prior:", log_prior) # we need to calculate negative log-prior with dirichlet function because of multivariate case (A,T,G,C)--> we have four options instead of two(binomial)

n_alpha_q = n + alpha * q

# Posterior normalization constant B(n + alpha * q)
B_n_alpha_q = np.exp(dirichlet_log_normalization(1, n_alpha_q))

log_evidence = np.log(B_n_alpha_q / np.exp(log_prior))

print("Log Evidence:", log_evidence)

negative_log_posterior = negative_log_likelihood - log_prior + log_evidence

print("Negative Log-Posterior:", negative_log_posterior)
print("--------------------------------------------------")
print("Observed counts:", counts)
print("Total number of observations (N):", N)
print("Optimal probabilities (p*):", dict(zip(A, p_optimal)))
print("Sum of probabilities (should be 1):", np.sum(p_optimal))