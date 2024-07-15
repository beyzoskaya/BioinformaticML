import numpy as np
from scipy.optimize import minimize_scalar

# need to define function f(s) and constraint D
def f(s):
    return 0.5 * s**2

D = 10
s_range = np.arange(1,6)
def negative_Lagrangian(lam):

    # Partititon function Z(lambda)
    Z_lambda = np.sum(np.exp(-lam*f(s_range))) # example range for s

    # entropy term
    entropy_term = np.sum(np.exp(-lam * f(s_range)) * (-lam * f(s_range) - np.log(np.exp(-lam * f(s_range)))))
    print(f"function f in range (1,6): {f(s_range)}")
    # -∑s ps log ps --> -ps log ps = exp(-λf(s))(-λf(s)-log(exp(-λf(s))))

    # Lagrangian
    lagrangian = entropy_term + lam * (np.sum(f(s_range)) - D) + np.log(Z_lambda)
    return -lagrangian

result = minimize_scalar(negative_Lagrangian)
#optimal_lambda = result.x
optimal_lambda = 1

partititon_function = np.sum(np.exp(-optimal_lambda*f(s_range)))
probabilities = np.exp(-optimal_lambda*f(s_range)) / partititon_function

print("Optimal Lambda:", optimal_lambda)
print("Probabilities p_s:")
for s, p in enumerate(probabilities):
    print(f"s = {s+1}: p_s = {p}")
