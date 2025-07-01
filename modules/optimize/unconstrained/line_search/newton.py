import numpy as np
from typing import Callable, Tuple
from modules.optimize.unconstrained.line_search.backtracking import backtracking
from modules.hessian.eigenvalue_modification import eigenvalue_modification

def newton(
    f: Callable[[np.ndarray[np.double]], np.double],
    df: Callable[[np.ndarray[np.double]], np.ndarray[np.double]],
    d2f: Callable[[np.ndarray[np.double]], np.ndarray[np.ndarray[np.double]]],
    x: np.ndarray[np.double],
    tol: np.double,
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], int]:
    iter = 1

    p = np.linalg.solve(d2f(x), -df(x))

    alpha = backtracking(f, df, 10**-4, 1., 0.5, x, p)
    x += alpha * p

    while np.linalg.norm(p, ord=np.inf) > tol and iter < max_iter:        
        p = np.linalg.solve(d2f(x), -df(x))

        alpha = backtracking(f, df, 10**-4, 1., 0.5, x, p)
        
        x += alpha * p
        iter += 1

    return x, iter

def newton_eigenvalue_modification(
    f: Callable[[np.ndarray[np.double]], np.double],
    df: Callable[[np.ndarray[np.double]], np.ndarray[np.double]],
    d2f: Callable[[np.ndarray[np.double]], np.ndarray[np.ndarray[np.double]]],
    x: np.ndarray[np.double],
    tol: np.double,
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], int]:
    iter = 1    

    p = eigenvalue_modification(d2f(x), df(x), 10**-8)

    alpha = backtracking(f, df, 10**-4, 1., 0.5, x, p)
    x += alpha * p

    while np.linalg.norm(p, ord=np.inf) > tol and iter < max_iter:        
        p = eigenvalue_modification(d2f(x), df(x), 10**-8)

        alpha = backtracking(f, df, 10**-4, 1., 0.5, x, p)

        x += alpha * p
        iter += 1

    return x, iter
