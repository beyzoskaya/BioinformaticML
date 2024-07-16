import numpy as np
import matplotlib.pyplot as plt

# parameters for bivariate distribution
mean = np.array([0, 0])
cov = np.array([[1, 0.5], [0.5, 1]])

# f(x1, x2) = x1**2 + x2
def f(x1, x2):
    return x1**2 + x2

# Number of samples T to use Monte Carlo approximation for calculation of expectation
T = 10000

# sample generation from bivariate distribution
samples = np.random.multivariate_normal(mean, cov, T)

# each sample calculated with f function
f_values = np.array([f(x1, x2) for x1, x2 in samples])

# Monte Carlo approximation of the expectation E(f)
expectation_f = np.sum(f_values) / T

print(f"Approximate expectation E(f): {expectation_f}")

plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], s=1, marker='o', label='Generated Samples')
plt.title('Samples from Bivariate Normal Distribution')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.savefig('samples_Monte_Carlo.png')
plt.show()
