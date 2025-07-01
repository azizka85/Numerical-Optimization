import numpy as np
from typing import Callable, Tuple
    
def calc_delta(
    rho: np.double,
    eta: np.double,
    delta: np.double,
    delta_max: np.double,
    x: np.ndarray[np.double],
    p: np.ndarray[np.double],
    tol: np.double
) -> Tuple[np.double, np.ndarray[np.double]]:
    if rho < 0.25:
        delta *= 0.25
    elif rho > 0.75 and abs(np.sqrt(np.dot(p, p)) - delta) < tol:
        delta = min(2*delta, delta_max)

    if rho > eta:
        x += p

    return delta, x

def calc_rho(
    f: Callable[[np.ndarray[np.double]], np.double],
    g: np.ndarray[np.double],
    B: np.ndarray[np.ndarray[np.double]],
    x: np.ndarray[np.double],
    p: np.ndarray[np.double],
    tol: np.double
) -> np.double:
    m = np.dot(g, p) + 0.5*np.dot(p, B@p)
    
    if m <= tol:
        return 1.

    return (f(x + p) - f(x)) / m
