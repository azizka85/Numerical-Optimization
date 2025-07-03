import numpy as np
from typing import Callable, Tuple
from scipy.sparse import csr_matrix
from modules.optimize.unconstrained.line_search.backtracking import backtracking

def newton_cg(
    f: Callable[[np.ndarray[np.double]], np.double],
    df: Callable[[np.ndarray[np.double]], np.ndarray[np.double]],
    d2f: Callable[[np.ndarray[np.double]], csr_matrix],
    x: np.ndarray[np.double],
    tol: np.double,
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], int]:
    iter = 1
    
    n = len(x)
    g = df(x)

    while np.linalg.norm(g, ord=np.inf) > tol and iter < max_iter: 
        z = np.zeros(n)
        r = g
        d = -r

        B = d2f(x)

        xi = np.dot(r, r)

        epsilon = min(0.5, np.sqrt(np.sqrt(xi))) * np.sqrt(xi)

        jiter = 0

        while np.sqrt(xi) >= epsilon and jiter < max_iter:
            v = B @ d
            gamma = np.dot(d, v)
            
            if gamma <= 0:
                break

            rho = xi / gamma
            z += rho * d
            r += rho * v

            xi1 = np.dot(r, r)
            beta = xi1 / xi
            xi = xi1

            d = -r + beta * d

            jiter += 1

        if jiter > 0:
            p = z
        else:
            p = -g

        alpha = backtracking(f, df, 10**-4, 1., 0.5, x, p)

        x += alpha * p
        g = df(x)

        iter += 1

    return x, iter
