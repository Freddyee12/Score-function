import numpy as np
import matplotlib.pyplot as plt
from utils import *

def rscusum(data, mu_pre, sigma_pre, mu_post, sigma_post, lambda_, threshold):
    scores = []
    z = 0
    for X in data:
        score_pre = hyvarinen_score(X, mu_pre, sigma_pre)
        score_post = hyvarinen_score(X, mu_post, sigma_post)
        z += lambda_ * (score_pre - score_post)
        z = max(z, 0)  # reset if negative
        scores.append(z)
        if z >= threshold:
            break
    return scores

# Parameters
np.random.seed(42)
mu_pre = 0
sigma_pre = 1
mu_post = 0.5  # Post-change mean shifted
sigma_post = 1  # Post-change sigma, not used for non-robust SCUSUM
lambda_ = 0.5
threshold = 5
data_length = 1000
change_point = 500

# Generating synthetic data
data = np.random.normal(mu_pre, sigma_pre, data_length)
data[change_point:] = np.random.normal(mu_post, sigma_pre, data_length - change_point)

# Running RSCUSUM
rscusum_scores = rscusum(data, mu_pre, sigma_pre, mu_post, sigma_post, lambda_, threshold)

# Non-robust SCUSUM (assuming same as pre-change post-change)
non_robust_scores = rscusum(data, mu_pre, sigma_pre, mu_pre, sigma_pre, lambda_, threshold)

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(rscusum_scores, label='RSCUSUM', color='blue')
plt.plot(non_robust_scores, label='Non-robust SCUSUM', color='red')
plt.axvline(x=change_point, color='green', linestyle='--', label='Change Point')
plt.axhline(y=threshold, color='grey', linestyle='--', label='Threshold')
plt.xlabel('Time')
plt.ylabel('Detection Score')
plt.title('Detection Score vs. Time')
plt.legend()
plt.show()
