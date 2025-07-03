import numpy as np
from typing import Callable, Tuple
from scipy.sparse import csr_matrix
from scipy.optimize import fsolve

from modules.optimize.unconstrained.trust_region import calc_rho, calc_delta

def newton_cg(
    f: Callable[[np.ndarray[np.double]], np.double],
    df: Callable[[np.ndarray[np.double]], np.ndarray[np.double]],
    d2f: Callable[[np.ndarray[np.double]], csr_matrix],
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

    p = find_direction(g, B, delta, tol, max_iter)

    rho = calc_rho(f, g, B, x, p, tol)

    delta, x = calc_delta(rho, eta, delta, delta_max, x, p, 10**-4)

    g = df(x)
    B = d2f(x)

    while np.linalg.norm(g, ord=np.inf) > tol and iter < max_iter:        
        p = find_direction(g, B, delta, tol, max_iter)

        rho = calc_rho(f, g, B, x, p, tol)

        delta, x = calc_delta(rho, eta, delta, delta_max, x, p, 10**-4)

        g = df(x)
        B = d2f(x)

        iter += 1

    return x, iter

def find_direction(
    g: np.ndarray[np.double],
    B: csr_matrix,
    delta: np.double,
    tol: np.double,
    max_iter: np.int32
) -> np.ndarray[np.double]:
    n = len(g)
    
    z = np.zeros(n)
    r = g.copy()
    d = -r

    xi = np.dot(r, r)

    epsilon = min(0.5, np.sqrt(np.sqrt(xi))) * np.sqrt(xi) + tol

    jiter = 0

    while np.sqrt(xi) >= epsilon and jiter < max_iter:
        v = B @ d
        gamma = np.dot(d, v)

        if gamma <= 0:
            tau = find_tau(z, d, delta)

            return  z + tau * d
        
        rho = xi / gamma
        z1 = z + rho * d

        if np.dot(z1, z1) > delta**2:
            tau = find_tau(z, d, delta)

            return  z + tau * d
        
        z = z1
        r += rho * v

        xi1 = np.dot(r, r)
        beta = xi1 / xi
        xi = xi1

        d = -r + beta * d

        jiter += 1

    return z

def find_tau(
    z: np.ndarray[np.double],
    d: np.ndarray[np.double],
    delta: np.double
):
    def func(x: np.double) -> np.double:
        p = z + x * d

        return np.dot(p, p) - delta**2
    
    return fsolve(func, 1.)[0]
