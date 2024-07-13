import math

def factorial(n):
    return math.factorial(n)

# Takes the set of alphabet as an argument 
def uniform_prior(A):
    num_nucleotides = len(A)
    return {x: 1 / num_nucleotides for x in A}

def likelihood_from_counts(counts,probabilities):
    N = sum(counts.values()) # Total length of sequence
    print(f"Number of N: {N}")
    factorial_N = factorial(N)
    print(f"Factorial for total number of counts: {factorial_N}")
    factorial_counts_product = math.prod(factorial(counts[x]) for x in counts)
    print(f"Factorial for product of counts(in denominatior): {factorial_counts_product} ")
    probabilites_product = math.prod(probabilities[x] ** counts[x] for x in counts)
    print(f"Product of probabilities (second multiplier in the equation): {probabilites_product}")
    likelihood = (factorial_N/factorial_counts_product)*probabilites_product
    return likelihood

def factorial_approx(n):
    """ Stirling's approximation for factorial used induce on vector P"""
    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)

def multinomial_likelihood(counts,probabilities):
    N = sum(counts.values())
    factorial_N = factorial_approx(N)
    factorial_counts_product = sum(factorial_approx(counts[x]) for x in counts)
    probabilities_product = math.prod(probabilities[x] ** counts[x] for x in counts)
    likelihood = math.exp(factorial_N - factorial_counts_product) * probabilities_product
    return likelihood

def dirichlet_posterior(counts, alpha, prior_probabilities):
    N = sum(counts.values())
    beta = N + alpha
    posterior_probabilities = {x: (counts[x] + alpha * prior_probabilities[x]) / beta for x in counts}
    return posterior_probabilities

def relative_entropy(observed_probabilities, model_probabilities):
    return -sum(observed_probabilities[x] * math.log(model_probabilities[x]) for x in observed_probabilities)


if __name__ == "__main__":
    counts = {'A': 3, 'C': 2, 'G': 4, 'T': 1}
    probabilities = {'A': 0.2, 'C': 0.3, 'G': 0.4, 'T': 0.1}
    
    # We can also assign a distribution again instead of givin each alphabet element different prior probs
    nucleotides = ['A', 'C', 'G', 'T']
    prior_probabilities = uniform_prior(nucleotides)
    likelihood = multinomial_likelihood(counts, prior_probabilities)
    
    print(f"Prior Probabilities: {prior_probabilities}")
    print(f"Multinomial Likelihood: {likelihood}")
    alpha = 1.0
    posterior_probabilities = dirichlet_posterior(counts=counts,alpha=alpha,prior_probabilities=prior_probabilities)
    print(f"Dirichlet Posterior Probabilities: {posterior_probabilities}")
    
    #likelihood = likelihood_from_counts(counts, probabilities)
    #print(f"Likelihood: {likelihood}")

    # Calculate relative entropy (KL divergence)
    # (KL) divergence, measures the difference between two probability distributions observed and model probs

    observed_probabilities = {x: counts[x] / sum(counts.values()) for x in nucleotides}
    kl_divergence = relative_entropy(observed_probabilities, prior_probabilities)
    
    print(f"Observed Probabilities: {observed_probabilities}")
    print(f"Relative Entropy (KL Divergence): {kl_divergence}")