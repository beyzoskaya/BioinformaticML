import numpy as np

def softmax(x):
    e_x = np.exp(x-max(x))
    return e_x / e_x.sum(axis=0)

def negative_log_likelihood(t,y):
    nll = -np.sum(t * np.log(y))
    return nll

def derivative_nll_y(t,y):
    return -(t / y)

def predict_class(y):
    return np.argmax(y)


t = np.array([1, 0, 0]) # one hot encoded version like class 0 is classified and the others are 0
y = np.array([0.8, 0.1, 0.1]) # softmax output 

nll_value = negative_log_likelihood(t, y)
print(f"Negative Log-Likelihood (NLL): {nll_value}")

grad_y = derivative_nll_y(t, y)
print(f"Gradient of NLL with respect to y: {grad_y}")

predicted_class = predict_class(y)
print(f"Predicted Class: {predicted_class}")