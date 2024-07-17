import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

"""
Implements the Expectation Maximization Algorithm to synthetic Gaussian Mixture Distributions
"""


def initialize_parameters(data, K):
    n, d = data.shape
    np.random.seed(0)
    means = data[np.random.choice(n, K, False)]
    covariances = np.array([np.eye(d)] * K)
    weights = np.ones(K) / K
    return means, covariances, weights

def e_step(data, weights, means, covariances):
    n = data.shape[0]
    K = weights.shape[0]
    responsibilities = np.zeros((n, K))
    for k in range(K):
        responsibilities[:, k] = weights[k] * multivariate_normal.pdf(data, mean=means[k], cov=covariances[k])
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

def m_step(data, responsibilities):
    n, d = data.shape # n is the number of data points, d is the dimensionality of the data
    K = responsibilities.shape[1] # K is the number of components
    # effective number of points for each component
    Nk = responsibilities.sum(axis=0)

    # update mixture weights
    weights = Nk / n

    # update means
    means = np.dot(responsibilities.T, data) / Nk[:, np.newaxis]

    # update covariances
    covariances = np.zeros((K, d, d))
    for k in range(K):
        diff = data - means[k]
        covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]
    return weights, means, covariances

# Because we use Gaussian Mixture, we need weights,means and covariances
def log_likelihood(data, weights, means, covariances):
    n = data.shape[0]
    K = weights.shape[0]
    log_likelihood = 0.0
    for i in range(n):
        likelihood_i = 0
        for k in range(K):
            likelihood_i += weights[k] * multivariate_normal.pdf(data[i], mean=means[k], cov=covariances[k])
        log_likelihood += np.log(likelihood_i)
    return log_likelihood

def em_algorithm(data,K,max_iter=100,tol=1e-4):
    means, covariances, weights = initialize_parameters(data, K)
    log_likelihoods = []

    for i in range(max_iter):
        responsibilities = e_step(data,weights,means,covariances)
        weights, means, covariances = m_step(data, responsibilities)
        ll = log_likelihood(data, weights, means, covariances)
        log_likelihoods.append(ll)
        if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            print(f"Converged at iteration {i}")
            break
        print(f"Iteration {i}: Log-Likelihood = {ll}")
    
    return means, covariances, weights, log_likelihoods

mean1 = [-2, 0]
cov1 = [[1, 0.5], [0.5, 1]]
mean2 = [3, 3]
cov2 = [[1, -0.5], [-0.5, 1]]
weights = [0.4, 0.6]

def generate_data(num_samples):
    num_samples_comp1 = int(num_samples * weights[0])
    num_samples_comp2 = num_samples - num_samples_comp1
    X1 = np.random.multivariate_normal(mean1, cov1, num_samples_comp1)
    X2 = np.random.multivariate_normal(mean2, cov2, num_samples_comp2)
    return np.vstack([X1, X2])

num_samples = 200
X = generate_data(num_samples)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=30, marker='o', label='Data Points')
plt.title('Generated Data Points')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()

K = 2
estimated_means, estimated_covariances, estimated_weights, log_likelihoods = em_algorithm(X, K)


print("Estimated means:")
print(estimated_means)
print("Estimated covariances:")
print(estimated_covariances)
print("Estimated weights:")
print(estimated_weights)