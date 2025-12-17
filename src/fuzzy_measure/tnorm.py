

import numpy as np

def fuzzy_power_tnorm(x, m, b, alpha=0.5, choice=3):
    hm = 1 / m * np.arange(m, 0, -1)
    g = hm ** np.exp(b)
    choq = tnorm(x, g, alpha, choice)
    ye = np.sum(choq, axis=1)
    return ye.ravel()


def power_prime_tnorm(x, y, ye, m, b, alpha, choice):
    hm = 1 / m * np.arange(m, 0, -1)
    g = hm ** np.exp(b)
    r = np.exp(b) * np.log(hm) * g
    a = grad_y_tnorm(x, g.ravel(), alpha, choice)
    grad = (ye - y) @ (a @ r)
    return grad


def alpha_P_prime_tnorm(x, y, ye, m, b, alpha, choice):
    hm = 1 / m * np.arange(m, 0, -1)
    g = hm ** np.exp(b)
    z = grad_alpha_tnorm(x, g.ravel(), alpha, choice).sum(axis=1)
    grad = z @ (ye - y)
    return grad


def gradient_power_tnorm(x, y, ye, m, b, alpha, choice):
    dp = power_prime_tnorm(x, y, ye, m, b, alpha, choice)
    da = alpha_P_prime_tnorm(x, y, ye, m, b, alpha, choice)
    return np.array([dp, da])


def fuzzy_weight_tnorm(x, m, w, alpha=0.5, choice=3):
    w = w.ravel()
    hm = np.triu(np.ones((m, m)))
    g = hm @ (w / np.sum(w))
    choq = tnorm(x, g, alpha, choice)
    ye = np.sum(choq, axis=1)
    return ye.ravel()


def weighprime_tnorm(x, y, ye, m, w, alpha, choice):
    hm = np.triu(np.ones((m, m)))
    sumw = np.sum(w)
    g0 = hm / (sumw**2)
    g = hm @ (w / sumw)
    v = np.tile(-w, (1, m))
    v[np.arange(m), np.arange(m)] += sumw
    gp = g0 @ v
    z = grad_y_tnorm(x, g.ravel(), alpha, choice) @ gp
    dw = (ye - y) @ z
    return dw


def alpha_W_prime_tnorm(x, y, ye, m, w, alpha, choice):
    hm = np.triu(np.ones((m, m)))
    sumw = np.sum(w)
    g = hm @ (w / sumw)
    z = grad_alpha_tnorm(x, g.ravel(), alpha, choice).sum(axis=1)
    da = z @ (ye - y)
    return da


def gradient_weight_tnorm(x, y, ye, m, w, alpha, choice):
    dw = weighprime_tnorm(x, y, ye, m, w, alpha, choice)/len(y)
    da = alpha_W_prime_tnorm(x, y, ye, m, w, alpha, choice)/len(y)
    return np.concatenate([dw, [da]])


def tnorm_3(x, y, beta):
    alpha = 1 - np.exp(beta)
    return (x * y) / (1 - (alpha * (1 - x) * (1 - y)))


def grad_tnorm_3_y(x, y, beta):
    num = x * (-np.exp(beta) * (x - 1) + x)
    den = ((np.exp(beta) * (x - 1) * (y - 1)) - (x * y) + x + y) ** 2
    return num / den


def grad_tnorm_3_alpha(x, y, beta):
    num = -np.exp(beta) * (x - 1) * (y - 1) * (x * y)
    den = (np.exp(beta) * (x - 1) * (y - 1) - (x * y) + x + y) ** 2
    return num / den


def tnorm_6(x, y, beta):
    alpha = np.exp(beta)
    A = 1 - x
    B = 1 - y
    A = np.where(A <= 0, 1e-15, A)
    B = np.where(B <= 0, 1e-15, B)
    Z = (A**alpha) + (B**alpha) - ((A**alpha) * (B**alpha))
    return 1 - (Z ** (1 / alpha))


def grad_tnorm_6_y(x, y, beta):
    alpha = np.exp(beta)
    A = 1 - x
    B = 1 - y
    A = np.where(A <= 0, 1e-15, A)
    B = np.where(B <= 0, 1e-15, B)
    t1 = B ** (alpha - 1) * (1 - A**alpha)
    t2 = B**alpha * (1 - A**alpha) + A**alpha
    power = (1 - alpha) / alpha
    return t1 * (t2**power)


def grad_tnorm_6_alpha(x, y, beta):
    alpha = np.exp(beta)
    A = 1 - x
    B = 1 - y
    A = np.where(A <= 0, 1e-15, A)
    B = np.where(B <= 0, 1e-15, B)
    Z = (A**alpha) + (B**alpha) - (A**alpha * B**alpha)
    t1 = -(Z ** (1 / alpha)) / alpha
    t2 = (A**alpha * alpha * np.log(A) * (1 - B**alpha)) / Z
    t3 = (B**alpha * alpha * np.log(B) * (1 - A**alpha)) / Z
    return t1 * (t2 + t3 - np.log(Z))


def tnorm(x, y, alpha, choice=3):
    if choice == 3:
        return tnorm_3(x, y, alpha)
    elif choice == 6:
        return tnorm_6(x, y, alpha)
    else:
        raise ValueError("choice must be 3 or 6")


def grad_y_tnorm(x, y, alpha, choice=3):
    if choice == 3:
        return grad_tnorm_3_y(x, y, alpha)
    elif choice == 6:
        return grad_tnorm_6_y(x, y, alpha)
    else:
        raise ValueError("choice must be 3 or 6")


def grad_alpha_tnorm(x, y, alpha, choice=3):
    if choice == 3:
        return grad_tnorm_3_alpha(x, y, alpha)
    elif choice == 6:
        return grad_tnorm_6_alpha(x, y, alpha)
    else:
        raise ValueError("choice must be 3 or 6")
