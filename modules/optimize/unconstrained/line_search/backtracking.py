import numpy as np
from typing import Callable

def backtracking(    
    f: Callable[[np.ndarray[np.double]], np.double],
    df: Callable[[np.ndarray[np.double]], np.ndarray[np.double]],
    c: np.double,
    alpha: np.double,
    rho: np.double,
    x: np.ndarray[np.double],
    p: np.ndarray[np.double]
) -> np.double:
    prev_f = f(x)
    d = np.dot(df(x), p)

    while f(x + alpha * p) > prev_f + c * alpha * d:
        alpha *= rho

    return alpha
    
