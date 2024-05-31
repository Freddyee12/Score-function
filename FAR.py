import numpy as np
from tqdm import tqdm
from utils import *

def rscusum_TF(data_stream, p_infinity_params, q1_params, lambda_, threshold):
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
            return True
    else:
        return False
    

p_infinity_params = (0, 1)  # Mean 0, variance 1 (standard normal distribution)
q1_params = (0.5, 1.5)  # Slightly different parameters for Q1

lambda_ = 0.5
threshold = 3
trials = 10000
detect_success = 0

for _ in tqdm(range(trials), desc="Processing"):
    data_stream = np.random.randn(1000)  # Random data simulating pre-change scenario
    if(rscusum_TF(data_stream, p_infinity_params, q1_params, lambda_, threshold)):
        detect_success += 1
print(f"Threshold = {threshold}, and FAR = {detect_success / trials}")
print(f"Number of successful detection: {detect_success}")