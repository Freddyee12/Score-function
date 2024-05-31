import numpy as np
from tqdm import tqdm
from utils import *

def rscusum(data_stream, p_infinity_params, q1_params, lambda_, threshold):
    """ Implements the RSCUSUM algorithm.
        data_stream: array of data points
        p_infinity_params: parameters (mean, variance) of the pre-change distribution P∞
        q1_params: parameters (mean, variance) of the least favorable distribution Q1
        lambda_: the pre-selected multiplier
        threshold: the stopping threshold to control false alarms
    """
    z = 0
    time = 0
    for X in data_stream:
        SH_p_inf = hyvarinen_score(X, p_infinity_params)
        SH_q1 = hyvarinen_score(X, q1_params)
        z_lambda = lambda_ * (SH_p_inf - SH_q1)
        z = max(z + z_lambda, 0)  # Ensure z stays non-negative
        time += 1
        if z >= threshold:
            print(f"Change detected at time {time} with score {z}")
            break
    else:
        print("Change is not occured!")


# Example usage:
# np.random.seed(42)
data_stream = np.random.randn(1000)  # Random data simulating pre-change scenario

#parameter setting
p_infinity_params = (0, 1)  # Mean 0, variance 1 (standard normal distribution)
q1_params = (0.5, 1.5)  # Slightly different parameters for Q1
lambda_ = 0.5
threshold = 2

rscusum(data_stream, p_infinity_params, q1_params, lambda_, threshold)