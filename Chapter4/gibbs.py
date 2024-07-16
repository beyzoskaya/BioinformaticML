import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# Synthetic data Gaussian Mixture 
np.random.seed(0)
mean1 = [-2, 0]
cov1 = [[1, 0.5], [0.5, 1]]
mean2 = [3, 3]
cov2 = [[1, -0.5], [-0.5, 1]]
weights = [0.4, 0.6]

def generate_data(num_samples):
    num_samples_comp1  = int(num_samples*weights[0])
    num_samples_comp2 =  num_samples - num_samples_comp1
    X1 = np.random.multivariate_normal(mean1, cov1, num_samples_comp1)
    X2 = np.random.multivariate_normal(mean2, cov2, num_samples_comp2)
    return np.vstack([X1,X2])

num_samples = 200
X = generate_data(num_samples)

K=2 # number of components
n,d = X.shape 
print(f"Shape of synthetic data: {X.shape}")
print(f"dimensionality d: {d}")
print(f"number of data points in each dim: {n}")

def initialize_parameters_sklearn(data, K):
    gmm = GaussianMixture(n_components=K, random_state=0).fit(data)
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_
    return means, covariances, weights

means_init, covs_init, weights_init = initialize_parameters_sklearn(X, K)
print(f"Initial means: {means_init}")
print(f"Initial covariances:\n{covs_init}")
print(f"Initial weights: {weights}")

#means_init = np.array([[0, 0], [1, 1]])
#print(f"Initial means: {means_init}")

#covs_init =np.array([np.eye(d)] * K)
#print(f"Initial covs: {covs_init}")

#weights_init = np.ones(K) / K 
#print(f"Initial weights: {weights_init}")

def sample_means(X,covariances,weights):
    means = np.zeros_like(means_init)
    for k in range(K):
        # Mahalanobis Distance calculation for point X between mean 
        # Formula for Mahalanobis Distance: D_Mahalanobis(X,μ) = √(X-μ)^T∑^-1(X-μ)
        #print(f"Covariance matrix: {covariances}")
        cov_inv = np.linalg.inv(covariances[k])
        #print(f"Inverse of covariance matrix for Mahalanobis Distance: {cov_inv}")
        cov_inv_dot_X = np.dot(cov_inv,X.T).T
        means[k] = np.sum(weights[k] * cov_inv_dot_X, axis=0) / np.sum(weights[k])
    return means

def sample_covariances(X,means,weights):
    covariances = np.zeros_like(covs_init)
    for k in range(K):
        diff = X - means[k]
        covariances[k] = np.dot(weights[k]*diff.T,diff) / np.sum(weights[k])
    return covariances

def sample_weights(weights,responsibilities):
    weights = responsibilities.sum(axis=0) / n
    return weights


max_iter = 100
for i in range(max_iter):
    means = sample_means(X,covs_init,weights_init)
    covariances = sample_covariances(X,means,weights_init)
    responsibilities = np.zeros((n,K))
    for k in range(K):
        responsibilities[:,k] = weights_init[k] * multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    weights = sample_weights(weights_init,responsibilities)

    means_init = means
    covs_init = covariances
    weights_init = weights

    print(f"Iteration {i}: Means = {means}, Covariances = {covariances}, Weights = {weights}")


estimated_means = means_init
estimated_covariances = covs_init
estimated_weights = weights_init

print("\nFinal Estimated Means:")
print(estimated_means)
print("Final Estimated Covariances:")
print(estimated_covariances)
print("Final Estimated Weights:")
print(estimated_weights)