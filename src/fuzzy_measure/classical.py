import numpy as np

def fuzzy_power(x, m, b):
    hm = 1 / m * np.arange(m, 0, -1)
    g = hm ** np.exp(b)
    ye = x @ g
    return ye.ravel()


def fuzzy_weight(x, m, w):
    hm = np.triu(np.ones((m, m)))
    g = hm @ (w / np.sum(w))
    ye = x @ g
    return ye.ravel()


def weight_prime(x, y, ye, m, w):
    hm = np.triu(np.ones((m, m)))
    sumw = np.sum(w)
    g0 = hm / (sumw**2)
    v = np.tile(-w, (1, m))
    v[np.arange(m), np.arange(m)] += sumw
    gp = g0 @ v
    d = ye - y
    dg = (d @ (x @ gp))/ len(y)
    return dg


def power_prime(x, y, ye, m, b):
    hm = 1 / m * np.arange(m, 0, -1)
    g = hm ** np.exp(b)
    r = np.exp(b) * np.log(hm) * g
    grad = ((ye - y) @ (x @ r)) / len(y)
    return grad
