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
N = len(sequence_data)


# Dirichlet Prior params
# Dirichlet is parametrized by alpha and q (q is prior prob)
alpha = 1.0  # concentration parameter
q = np.array([1/4] * 4)  # uniform prior in uniform case, Dirichlet is a symmetric and same for all alpabet elements

# logarithm of normalization constant
# normaliaztion function for dirichlet distribution is gamma 

# Takes a concentration parameter alpha and prior probs
def dirichlet_log_normalization(alpha, q):
    return np.sum(gammaln(alpha * q)) - gammaln(np.sum(alpha * q))

B_alpha_q = np.exp(dirichlet_log_normalization(alpha, q))

# update parameters with observed data
n_alpha_q = n + alpha*q

# posterior constant B(n+alpha*q)
B_n_alpha_q = np.exp(dirichlet_log_normalization(n_alpha_q, np.ones_like(q)))

def multinomial_log_likelihood(n, p):
    return np.sum(n * np.log(p))

likelihood = np.exp(multinomial_log_likelihood(n, q))

# evidence P(D)
P_D = B_n_alpha_q / B_alpha_q * likelihood

# calculate posterior distribution P(w|D)
posterior_distribution = np.exp(dirichlet_log_normalization(n_alpha_q, np.ones_like(q)))

print("Counts:", counts)
print("n:", n)
print("Alpha:", alpha)
print("q:", q)
print("B_alpha_q:", B_alpha_q)
print("n_alpha_q:", n_alpha_q)
print("B_n_alpha_q:", B_n_alpha_q)
print("Likelihood:", likelihood)
print("P_D (Evidence):", P_D)
print("Posterior Distribution:", posterior_distribution)

# Printed output for the sequence "ACGTCGATCGATCGATCGTAGCTAGCTAGCTGACT" should be
# Counts: {'A': 8, 'C': 9, 'G': 9, 'T': 9}
# n: [8 9 9 9]
# Alpha: 1.0
# q: [0.25 0.25 0.25 0.25]
# B_alpha_q: 172.79226606366026
# n_alpha_q: [8.25 9.25 9.25 9.25]
# B_n_alpha_q: 2.675359387101763e-22
# Likelihood: 8.470329472543041e-22
# P_D (Evidence): 1.3114693141337646e-45
# Posterior Distribution: 2.675359387101763e-22