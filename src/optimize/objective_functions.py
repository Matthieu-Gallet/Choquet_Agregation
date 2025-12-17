

import numpy as np
from fuzzy_measure.classical import fuzzy_power, power_prime, fuzzy_weight, weight_prime
from fuzzy_measure.tnorm import fuzzy_power_tnorm, gradient_power_tnorm, fuzzy_weight_tnorm, gradient_weight_tnorm


def objective(theta, x, y, m, methode="Power"):
    """Objective function for classical Choquet integral"""
    if methode == "Power":
        ye = fuzzy_power(x, m, theta)
        grad = power_prime(x, y, ye, m, theta)
    elif methode == "Weight":
        theta = theta.reshape(m, 1)
        ye = fuzzy_weight(x, m, theta)
        grad = weight_prime(x, y, ye, m, theta)
    cost = (0.5 * np.linalg.norm(y - ye) ** 2) / len(y)
    return cost, grad


def objective_tnorm(theta, x, y, m, tnorm_c, methode="Power"):
    """Objective function for Choquet integral with t-norms"""
    if methode == "Power":
        b = theta[0]
        alpha = theta[1]
        ye = fuzzy_power_tnorm(x, m, b, alpha, tnorm_c)
        grad = gradient_power_tnorm(x, y, ye, m, b, alpha, tnorm_c)
    elif methode == "Weight":
        weights = theta[:-1].reshape(m, 1)
        alpha = theta[-1]
        ye = fuzzy_weight_tnorm(x, m, weights, alpha, tnorm_c)
        grad = gradient_weight_tnorm(x, y, ye, m, weights, alpha, tnorm_c)
    cost = (0.5 * np.linalg.norm(y - ye) ** 2)/len(y)
    return cost, grad, ye

