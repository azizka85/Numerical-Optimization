import numpy as np
from typing import Callable, Tuple
from scipy.optimize import minimize_scalar

def conjugate_direction(
    f: Callable[[np.ndarray[np.double]], np.double],
    x: np.ndarray[np.double],
    tol: np.double,
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], int]:
    iter = 1
    n = len(x)

    p = np.eye(n)

    def phi(alpha: np.double) -> np.double:
        y = x + alpha*p[n-1]

        return f(y)
    
    res = minimize_scalar(phi)
    x1 = x + res.x*p[n-1]

    while np.linalg.norm(x1 - x, ord=np.inf) > tol and iter < max_iter:
        x = x1
        z0 = x
        z = z0.copy()

        for i in range(n):
            def phi(alpha: np.double) -> np.double:                
                y = z + alpha*p[i]

                return f(y)
            
            res = minimize_scalar(phi)
            z += res.x * p[i]            

        for i in range(1, n):
            p[i-1] = p[i]

        p[n-1] = z - z0

        def phi(alpha: np.double) -> np.double:
            y = z + alpha*p[n-1]

            return f(y)
        
        res = minimize_scalar(phi)
        x1 = z + res.x*p[n-1]

        iter += 1

    return x1, iter
