import numpy as np
from typing import Callable, Tuple
from modules.optimize.unconstrained.line_search.backtracking import backtracking

def conjugate_gradient(
    f: Callable[[np.ndarray[np.double]], np.double],
    df: Callable[[np.ndarray[np.double]], np.ndarray[np.double]],
    x: np.ndarray[np.double],
    tol: np.double,
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], int]:
    iter = 1
    g = df(x)
    p = -g

    alpha = backtracking(f, df, 10**-4, 1., 0.5, x, p)

    while np.linalg.norm(p, ord=np.inf) > tol and iter < max_iter:
        x += alpha * p

        beta = np.dot(g, g)

        g = df(x)

        beta = np.dot(g, g) / beta
        
        p = -g + beta*p

        alpha = backtracking(f, df, 10**-4, 1., 0.5, x, p)

        iter += 1

    return x, iter
