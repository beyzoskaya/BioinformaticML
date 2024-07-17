import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def compute_probability(d,weights):
    """ probability y(d) using sigmoid function"""
    x = np.dot(d,weights)
    return sigmoid(x)

def compute_negative_log_likelihood(t,y):
    return -(t*np.log(y) + (1-t) * np.log(1-y))

def derivative_E_with_respect_to_y(t, y):
    return - (t-y) / (y * (1-y))

def derivative_E_with_respect_to_x(t, y):
    return - (t-y)

def predict(d, weights):
    y = compute_probability(d, weights)
    if y >= 0.5:
        return 1  # Predict Class A
    else:
        return 0  # Predict Class A¯

weights = np.array([0.5, -0.3, 0.1])  
d = np.array([1.0, 2.0, 3.0])  
t = 1

y = compute_probability(d,weights)
print(f"Probability y(d) = {y}")

E = compute_negative_log_likelihood(t, y)
print(f"Negative Log-Likelihood Error E = {E}")

dE_dy = derivative_E_with_respect_to_y(t, y)
print(f"∂E/∂y = {dE_dy}")


dE_dx = derivative_E_with_respect_to_x(t, y)
print(f"∂E/∂x = {dE_dx}")

predicted_class = predict(d,weights)
if predicted_class == 1:
    print("Predicted: Class A")
else:
    print("Predicted: Class A¯")