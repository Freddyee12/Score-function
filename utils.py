import os

def hyvarinen_score(X, distribution_params):
    """ Calculate the Hyvärinen score for a given data point X and distribution parameters.
        For simplicity, assume a Gaussian distribution and use a basic approximation.
    """
    mu, sigma = distribution_params
    score = ((X - mu) ** 2) / (2 * sigma ** 2) - 1 / sigma ** 2
    return score