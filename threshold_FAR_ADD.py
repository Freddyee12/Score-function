""" 
    This code is describing the trade-off behavior between threshold to ADD & FAR
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import *

def run_rscusum_simulation(data_stream, p_infinity_params, q1_params, lambda_, threshold):
    """ Simulate the RSCUSUM and return the detection time or None if no detection. """
    z = 0
    for time, X in enumerate(data_stream):
        SH_p_inf = hyvarinen_score(X, p_infinity_params)
        SH_q1 = hyvarinen_score(X, q1_params)
        z_lambda = lambda_ * (SH_p_inf - SH_q1)
        z = max(z + z_lambda, 0)  # Ensure z stays non-negative
        if z >= threshold:
            return time  # Detection time
    return None  # No detection within the data stream

def simulate_rscusum(trials, data_length, change_point, p_infinity_params, q1_params, lambda_, threshold):
    """ Simulate the RSCUSUM algorithm over a number of trials to estimate false alarm rate and detection delay. """
    detections = []
    for _ in range(trials):
        # Generate pre-change data
        data_stream = np.random.normal(p_infinity_params[0], np.sqrt(p_infinity_params[1]), data_length)
        # Introduce change at change_point
        if change_point < data_length:
            data_stream[change_point:] = np.random.normal(q1_params[0], np.sqrt(q1_params[1]), data_length - change_point)
        
        detection_time = run_rscusum_simulation(data_stream, p_infinity_params, q1_params, lambda_, threshold)
        detections.append(detection_time)
    
    false_alarms = sum(1 for dt in detections if dt is not None and dt < change_point) / trials
    average_detection_delay = np.mean([dt - change_point for dt in detections if dt is not None and dt >= change_point])
    
    return false_alarms, average_detection_delay

# Parameters
# np.random.seed(42)
data_length = 10000
change_point = 500
p_infinity_params = (0, 1)  # Mean 0, variance 1
q1_params = (0.5, 1.5)  # Different mean and variance
lambda_ = 0.5
thresholds = np.linspace(2, 10, 20)
trials = 1000

false_alarm_rates = []
average_detection_delays = []

for threshold in thresholds:
    far, add = simulate_rscusum(trials, data_length, change_point, p_infinity_params, q1_params, lambda_, threshold)
    false_alarm_rates.append(far)
    average_detection_delays.append(add)

# Plotting the trade-off curve
print(false_alarm_rates)
plt.figure(figsize=(10, 6))
plt.plot(thresholds, false_alarm_rates, marker='o', label = 'FAR')
plt.plot(thresholds, average_detection_delays, marker='v', label = 'ADD')
plt.xlabel('Threshold')
plt.ylabel('ADD & FAR')
plt.title('Trade-off between False Alarm Rate and Average Detection Delay')
plt.grid(True)
plt.show()
