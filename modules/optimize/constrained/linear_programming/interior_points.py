import numpy as np
from typing import Tuple

def starting_point(
    c: np.ndarray[np.double],
    A: np.ndarray[np.double, np.double],
    At: np.ndarray[np.double, np.double],
    b: np.ndarray[np.double]
) -> Tuple[np.ndarray[np.double], np.ndarray[np.double], np.ndarray[np.double]]:    
    B = A @ At
    d = A @ c
    l = np.linalg.solve(B, d)
    y = np.linalg.solve(B, b)
    x = At @ y
    s = c - At @ l

    x += max(-1.5 * x.min(), 0)    
    s += max(-1.5 * s.min(), 0)

    e = np.ones(len(x))

    xs = np.dot(x, s)
    es = np.dot(e, s)    
    ex = np.dot(e, x)

    return x + (0.5 * xs / es) * e, l, s + (0.5 * xs / ex) * e

def interior_points(
    c: np.ndarray[np.double],
    A: np.ndarray[np.double, np.double],
    b: np.ndarray[np.double],
    tol: np.double,
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], np.ndarray[np.double], np.ndarray[np.double], np.double, np.int32]:
    iter = 1
    
    At = A.transpose()
    
    x, l, s = starting_point(c, A, At, b)

    rxs = x * s

    while np.linalg.norm(rxs, ord=np.inf) > tol and iter < max_iter:
        X = np.diag(x)
        S = np.diag(s)
        Si = np.diag(1 / s)
        XSi = X @ Si
        AXSi = A @ XSi
        ASi = A @ Si
        AXSiAt = AXSi @ At

        rb = A @ x - b
        rc = A.transpose() @ l + s - c        

        dl = np.linalg.solve(AXSiAt, -rb - AXSi @ rc + ASi @ rxs)
        ds = -rc - At @ dl
        dx = -Si @ rxs - XSi @ ds

        qx = min(1, step_length(x, dx))
        qs = min(1, step_length(s, ds))

        m = np.dot(x, s)
        m1 = np.dot(x + qx * dx, s + qs * ds)
        t = (m1 / m)**3

        e = np.ones(len(x))

        rxs += dx * ds - t * m * e

        dl = np.linalg.solve(AXSiAt, -rb - AXSi @ rc + ASi @ rxs)
        ds = -rc - At @ dl
        dx = -Si @ rxs - XSi @ ds

        qx = min(1, 0.9 * step_length(x, dx))
        qs = min(1, 0.9 * step_length(s, ds))

        x += qx * dx
        l += qs * dl
        s += qs * ds

        rxs = x * s

        iter += 1

    return x, l, s, np.dot(c, x), iter

def step_length(
    x: np.ndarray[np.double], 
    dx: np.ndarray[np.double]
) -> np.double:
    alpha = np.inf

    for i in range(len(x)):
        if dx[i] < 0:
            alpha = min(alpha, -x[i]/dx[i])

    return alpha if alpha < np.inf else 0
