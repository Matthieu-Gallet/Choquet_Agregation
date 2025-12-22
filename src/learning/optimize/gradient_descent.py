import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def gradient_update(grad):
    grad_a = grad[-1]
    grad_y = grad[:-1]
    grad_a /= np.linalg.norm(grad_a)
    grad_y /= np.linalg.norm(grad_y)
    return grad_a, grad_y


def test_convergence_nan(co, grad_a, grad_y, s1=0.05):
    df = np.diff(co)
    if np.isnan(grad_a) | np.isnan(grad_y).any():
        return True
    if (np.abs(df)[-1] < s1) | (df[-1] > 10):
        return True
    else:
        return False


def update_step(stpa, stpy, i):
    stpa *= (21 + i) / (22 + i)
    stpy *= (21 + i) / (22 + i)
    return stpa, stpy


def update_variable(stpa, stpy, theta, grad_a, grad_y):
    alpha = theta[-1]
    we = theta[:-1]
    alpha -= stpa * grad_a
    we -= stpy * grad_y
    return alpha, we


def GD_minimize(x_tr, ytr, theta, m, choice, methode="Weight", objective_func=None, verbose=False, **kwargs):
    try:
        niter = kwargs["niter"]
    except:
        niter = 500
    try:
        stpa = kwargs["stpa"]
    except:
        stpa = 0.005
    try:
        stpy = kwargs["stpy"]
    except:
        stpy = 0.005
    try:
        display = kwargs["display"]
    except:
        display = True

    co = [1e15]
    if verbose:
        pbar = tqdm(total=niter, disable=not (display), leave=False)
    for i in range(niter):
        cost, grad, ye = objective_func(theta, x_tr, ytr, m, choice, methode)
        grad_a, grad_y = gradient_update(grad)
        co.append(cost)
        if test_convergence_nan(co, grad_a, grad_y):
            break
        stpa, stpy = update_step(stpa, stpy, i)
        alpha, we = update_variable(stpa, stpy, theta, grad_a, grad_y)
        theta = np.concatenate([we, [alpha]])

        yest = np.where(ye > 0.5, 1, 0)
        train_acc = 100 * accuracy_score(ytr, yest)
        if verbose:
            pbar.set_description(f"cost : {cost:.2f} - accuracy (train) : {train_acc:.2f} ")
            pbar.update(1)
    if verbose:
        pbar.close()
    theta = np.concatenate([we, [alpha]])
    out = {
        "cost": co,
        "x": theta,
        "nit": i,
        "success": True,
        "message": "Converged (|f_n-f_(n-1)| ~= 0)",
        "accuracy": train_acc,
    }
    return out
