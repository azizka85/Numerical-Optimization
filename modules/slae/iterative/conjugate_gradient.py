import numpy as np
from typing import Callable, Tuple
from scipy.sparse import csr_matrix

def preconditioned_cg(
    solve: Callable[[np.ndarray[np.double]], np.ndarray[np.double]],    
    A: csr_matrix,
    b: np.ndarray[np.double],
    x: np.ndarray[np.double],
    tol: np.double,
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], int]:
    iter = 1
    
    r = A @ x - b
    
    y = solve(r)
    p = -y        

    ry = np.dot(r, y)

    ap = A @ p

    pap = np.dot(p, ap)

    while np.linalg.norm(p, ord=np.inf) > tol and iter < max_iter:        
        alpha = ry / pap
        
        x += alpha * p
        r += alpha * ap

        y = solve(r)

        ry1 = np.dot(r, y)

        beta = ry1 / ry

        ry = ry1

        p = -y + beta * p
        ap = A @ p

        pap = np.dot(p, ap)
        
        iter += 1

    return x, iter
