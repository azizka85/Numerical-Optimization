import numpy as np
from typing import Tuple, Optional
from scipy.sparse import csr_matrix

from modules.slae.iterative.conjugate_gradient import preconditioned_cg

def projected_cg(    
    A: csr_matrix,
    G: csr_matrix,
    c: np.ndarray[np.double],
    b: np.ndarray[np.double],
    tol: np.double,
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], np.ndarray[np.double], int]:
    At = A.T
    AAt = A @ At

    daat = A.diagonal()

    iterm = 0

    def preconditioned_solve(r: np.ndarray[np.double]) -> np.ndarray[np.double]:
        y = r.copy()

        for i in range(len(y)):
            q = abs(daat[i])

            if q > tol:
                y[i] /= q

        return y    

    def projection_solve(r: np.ndarray[np.double]) -> np.ndarray[np.double]:
        ar = A @ r                             

        v, _ = preconditioned_cg(preconditioned_solve, AAt, ar, np.zeros_like(ar), tol, max_iter) 

        return r - At @ v
    
    x, iter = preconditioned_cg(preconditioned_solve, AAt, b, np.zeros_like(b), tol, max_iter)

    iterm = max(iterm, iter)
    
    x, iter = preconditioned_cg(projection_solve, G, -c, At @ x, tol, max_iter)  

    iterm = max(iterm, iter)  

    l, iter = preconditioned_cg(preconditioned_solve, AAt, A @ (c + G @ x), np.zeros_like(b), tol, max_iter)

    iterm = max(iterm, iter)

    return x, l, iterm
