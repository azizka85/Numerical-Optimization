import numpy as np
from typing import Callable, Tuple
from scipy.optimize import fsolve

from modules.optimize.unconstrained.trust_region import calc_rho, calc_delta
from modules.hessian.eigenvalue_modification import eigenvalue_modification

def iterative_method(
    f: Callable[[np.ndarray[np.double]], np.double],
    df: Callable[[np.ndarray[np.double]], np.ndarray[np.double]],
    d2f: Callable[[np.ndarray[np.double]], np.ndarray[np.ndarray[np.double]]],
    x: np.ndarray[np.double],
    tol: np.double,
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], int]:
    iter = 1   

    g = df(x)
    B = d2f(x)

    delta_max = 1.
    delta = 0.9 * delta_max

    eta = 0.1

    p = find_direction(g, B, delta, 10**-4)

    rho = calc_rho(f, g, B, x, p, tol)

    delta, x = calc_delta(rho, eta, delta, delta_max, x, p, 10**-4)

    g = df(x)
    B = d2f(x)

    while np.linalg.norm(g, ord=np.inf) > tol and iter < max_iter:        
        p = find_direction(g, B, delta, 10**-4)

        rho = calc_rho(f, g, B, x, p, tol)

        delta, x = calc_delta(rho, eta, delta, delta_max, x, p, 10**-4)

        g = df(x)
        B = d2f(x)

        iter += 1

    return x, iter

def find_direction(
    g: np.ndarray[np.double],
    B: np.ndarray[np.ndarray[np.double]],
    delta: np.double,
    tol: np.double
) -> np.ndarray[np.double]:
    p = eigenvalue_modification(B, g, 10**-8)

    if np.sqrt(np.dot(p, p)) <= delta:
        return p
    
    lam, q = np.linalg.eigh(B)

    l = -lam[0]

    if np.abs(np.dot(q[0], g)) < tol:
        s = 0

        for i in range(1, len(lam)):
            d = np.dot(q[i], g)
            s += d**2 / (lam[i] + l)**2

        s -= delta**2

        def func(x: np.double) -> np.double:
            return s + x**2
        
        r = fsolve(func, 1.)

        p = abs(r[0]) * q[0]

        for i in range(1, len(lam)):
            d = np.dot(q[i], g)
            p += d * q[i] / (lam[i] + l)

        return p
    
    def func(x: np.double) -> np.double:
        s = 0

        for i in range(0, len(lam)):
            d = np.dot(q[i], g)
            s += (d / (lam[i] + x))**2

        s -= delta**2

        return s
    
    l = fsolve(func, 1.)[0]

    p = np.zeros(len(g))

    for i in range(0, len(lam)):
        d = np.dot(q[i], g)
        p -= d * q[i] / (lam[i] + l)

    return p
