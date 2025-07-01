import numpy as np
from typing import Callable, Tuple
from modules.optimize.unconstrained.line_search.backtracking import backtracking

def bfgs(
    f: Callable[[np.ndarray[np.double]], np.double],
    df: Callable[[np.ndarray[np.double]], np.ndarray[np.double]],
    x: np.ndarray[np.double],
    tol: np.double,
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], int]:
    iter = 1
    d = df(x)
    p = -d

    n = len(p)

    H = np.eye(n)
    alpha = backtracking(f, df, 10**-4, 1., 0.5, x, p)

    while np.linalg.norm(p, ord=np.inf) > tol and iter < max_iter:
        s = alpha * p
        x += s

        d1 = df(x)
        y = d1 - d

        d = d1
        beta = np.dot(y, s)
        
        if beta > 10**-6:
            rho = 1/beta
        
            L = np.eye(n) - rho * np.outer(s, y)
            R = np.eye(n) - rho * np.outer(y, s)

            H = L @ H @ R + rho * np.outer(s, s)

        p = -H @ d

        alpha = backtracking(f, df, 10**-4, 1., 0.5, x, p)

        iter += 1

    return x, iter

def sr1(
    f: Callable[[np.ndarray[np.double]], np.double],
    df: Callable[[np.ndarray[np.double]], np.ndarray[np.double]],
    x: np.ndarray[np.double],
    tol: np.double,
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], int]:
    iter = 1
    d = df(x)
    p = -d

    n = len(p)

    H = np.eye(n)
    alpha = backtracking(f, df, 10**-4, 1., 0.5, x, p)

    while np.linalg.norm(p, ord=np.inf) > tol and iter < max_iter:
        s = alpha * p
        x += s

        d1 = df(x)
        y = d1 - d

        d = d1
        rho = 1/np.dot(y, s)
        
        v = s - H @ y        
        beta = np.dot(v, y)

        if beta > 10**-6:
            H += np.outer(v, v) / beta

        p = -H @ d

        alpha = backtracking(f, df, 10**-4, 1., 0.5, x, p)

        iter += 1

    return x, iter
