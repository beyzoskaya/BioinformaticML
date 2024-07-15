import numpy as np

# Define function f(s)
def f(s):
    return np.sin(s)

def g(s):
    return np.cos(s)

# Example distributions Q(s) and R(s)
Q = np.array([0.1, 0.2, 0.3, 0.2, 0.2])  
print(f"Distribution Q: {Q}")
R = np.array([0.2, 0.1, 0.1, 0.4, 0.2]) 
print(f"Distribution R: {R}")
print(f"Summation of dist Q: {np.sum(Q)}")
print(f"Summation of dist R: {np.sum(R)}")
lambda_value = 1

E_Q_f = np.sum(Q * f(np.arange(len(Q)))) # expected value calculation ∑Qxf(s)
print(f"Expected value calc for distribution Q: {E_Q_f}")
print(f"arrange function start from 1 to 6: {f(np.arange(1, 6))}")
H_Q = -np.sum(Q * np.log(Q)) # entropy --> -∑Q(s) logQ(S)
print(f"Entropy for distribution Q: {H_Q}")

E_R_f = np.sum(R * f(np.arange(len(R))))
print(f"Expected value calc for distribution R: {E_R_f}")
H_R = -np.sum(R * np.log(R))
print(f"Entropy for distribution R: {H_R}")

# Calculate the free energies F(Q, lambda) and F(R, lambda)
F_Q_lambda = E_Q_f - (1 / lambda_value) * H_Q
print(f"Free energy F(Q, lambda): {F_Q_lambda}")
F_R_lambda = E_R_f - (1 / lambda_value) * H_R
print(f"Free energy F(R, lambda): {F_R_lambda}")

# Calculate the relative entropy H(Q, R)
H_Q_R = np.sum(Q * np.log(Q / R))
print(f"Relative entropy H(Q, R): {H_Q_R}")

free_energy_difference = F_Q_lambda - F_R_lambda
print(f"Difference in free energy F(Q, lambda) - F(R, lambda): {free_energy_difference}")

def calculate_free_energy(Q,R,f,lam):
    term1 = np.sum((Q - R) * (f + (1 / lam) * np.log(R)))
    term2 = (1 / lam) * np.sum(Q * np.log(Q / R))
    free_energy_difference = term1+term2
    return free_energy_difference

free_energy_diff_function = calculate_free_energy(Q, R, f(np.arange(len(Q))), lambda_value)
print(f"Free energy difference calculated by function: {free_energy_diff_function}")


