import numpy as np

def gaussian_pdf(t,y,sigma):
    """
    t: target values
    y: predicted values
    sigma: standard deviation for Gaussian 
    """

    exponent = -0.5 * ((t - y) / sigma)**2
    normalization_factor = 1/np.sqrt(2*np.pi*sigma**2)
    pdf = np.prod(normalization_factor*np.exp(exponent))
    return pdf

def negative_log_likelihood(t,y,sigma):
    error_term = (t-y)**2 / (2*sigma**2)
    constant_term = 0.5 * np.log(2 * np.pi * sigma**2)
    nll = np.sum(error_term+constant_term)
    return nll

def derivative_nll_y(t,y,sigma):
    grad_y =  -(t - y) / (sigma**2)
    return grad_y

t = np.array([1.0, 2.0, 3.0])
y = np.array([0.8, 2.5, 2.9])
sigma = np.array([0.1, 0.2, 0.15])

pdf_value = gaussian_pdf(t,y,sigma)
print(f"Gaussian PDF: {pdf_value}")

nll_value = negative_log_likelihood(t,y,sigma)
print(f"Negative Log-Likelihood (NLL): {nll_value}")

grad_y = derivative_nll_y(t,y,sigma)
print(f"Gradient of NLL with respect to y: {grad_y}")
