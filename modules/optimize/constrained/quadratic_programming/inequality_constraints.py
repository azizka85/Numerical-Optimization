import numpy as np
from typing import Tuple, List, Dict
from scipy.sparse import csr_matrix, eye

from modules.optimize.constrained.quadratic_programming.equality_constraints import projected_cg
from modules.slae.iterative.conjugate_gradient import preconditioned_cg

def active_set_method(
    A: csr_matrix,
    G: csr_matrix,
    c: np.ndarray[np.double],
    b: np.ndarray[np.double],
    x: np.ndarray[np.double],
    Wset: List[np.int32],
    tol: np.double,
    max_iter: np.int32        
) -> Tuple[np.ndarray[np.double], np.ndarray[np.double], int]:
    iter = 1    

    d = G.diagonal()

    def preconditioned_solve(r: np.ndarray[np.double]) -> np.ndarray[np.double]:
        y = r.copy()

        for i in range(len(y)):
            q = abs(d[i])

            if q > tol:
                y[i] /= q

        return y

    while iter < max_iter:
        n = len(Wset)

        g = G @ x + c

        if n == 0:
            l = np.array([])
            y = np.zeros_like(c)

            p, _ = preconditioned_cg(preconditioned_solve, G, -g, y, tol, max_iter)
        else:
            Wset.sort()

            W = A[Wset, :]
            y = np.zeros(len(Wset))            

            p, l, _ = projected_cg(W, G, g, y, tol, max_iter)            

        if np.linalg.norm(p, ord=np.inf) < tol:
            if is_positive(l):
                return x, l, iter
            else:
                i = np.argmin(l)
                del Wset[i]
        else:
            j = 0

            ip = -1
            alpha = 1.

            for i in range(0, A.shape[0]):
                if j < n and Wset[j] == i:
                    j += 1
                else:
                    ap = (A[i] @ p)[0]

                    if ap < 0:
                        z = (b[i] - (A[i] @ x)[0]) / ap

                        if z < tol:
                            alpha = 0.
                            ip = i
                        elif z < alpha - tol:
                            alpha = z
                            ip = i
            
            x += alpha * p           

            if alpha < 1 - tol:                                
                Wset.append(ip)    
            
        iter += 1

    return x, l, iter

def is_positive(x: np.ndarray[np.double]) -> bool:
    for e in x:
        if e < 0:
            return False
        
    return True

def starting_point(
    c: np.ndarray[np.double],
    A: np.ndarray[np.double, np.double],
    At: np.ndarray[np.double, np.double],
    G: np.ndarray[np.double, np.double],
    b: np.ndarray[np.double]
) -> Tuple[np.ndarray[np.double], np.ndarray[np.double], np.ndarray[np.double]]:   
    l = np.ones_like(b)

    x = np.linalg.solve(G, At @ l - c)
    y = A @ x - b

    y += max(-1.5 * y.min(), 0)

    ly = np.dot(l, y)
    ll = np.dot(l, l)

    l += 0.5
    y += 0.5 * ly / ll

    return x, l, y

def interior_points(
    A: np.ndarray[np.double, np.double],
    G: np.ndarray[np.double, np.double],
    c: np.ndarray[np.double],
    b: np.ndarray[np.double],
    tol: np.double,
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], np.ndarray[np.double], np.ndarray[np.double], np.int32]:
    iter = 1
    
    At = A.transpose()

    x, l, y = starting_point(c, A, At, G, b)

    m = len(y)

    e = np.ones(m)

    rly = l * y

    while np.linalg.norm(rly, ord=np.inf) > tol and iter < max_iter:
        sigma = 0.
        mu = np.dot(y, l) / m

        rd = G @ x - At @ l + c
        rp = A @ x - y - b

        L = np.diag(l)
        Y = np.diag(y)

        Yi = np.diag(1 / y)
        YiL = Yi @ L

        AtYi = At @ Yi
        AtYiL = AtYi @ L
        AtYiLA = AtYiL @ A

        dx = np.linalg.solve(G + AtYiLA, -rd - AtYiL @ (rp + y) + sigma * mu * AtYi @ e)
        dy = A @ dx + rp
        dl = -YiL @ dy - L @ e + sigma * mu * Yi @ e

        alpha = min(step_length(l, dl, y, dy, 1.), 1.)        

        mu1 = np.dot(y + alpha * dy, l + alpha * dl) / m
        sigma = (mu1 / mu)**3

        dx = np.linalg.solve(G + AtYiLA, -rd - AtYiL @ (rp + y) - AtYi @ (dl * dy) + sigma * mu * AtYi @ e)
        dy = A @ dx + rp
        dl = -YiL @ dy - L @ e - Yi @ (dl * dy) + sigma * mu * Yi @ e

        alpha = min(step_length(l, dl, y, dy, 0.9), 1.)

        x += alpha * dx
        y += alpha * dy
        l += alpha * dl

        rly = l * y

        iter += 1
        
    return x, l, y, iter

def step_length(
    l: np.ndarray[np.double], 
    dl: np.ndarray[np.double],
    y: np.ndarray[np.double], 
    dy: np.ndarray[np.double],
    t: np.double
) -> np.double:
    alpha = np.inf

    for i in range(len(l)):
        if dl[i] < 0:
            alpha = min(alpha, -t*l[i]/dl[i])

    for i in range(len(y)):
        if dy[i] < 0:
            alpha = min(alpha, -t*y[i]/dy[i])

    return alpha if alpha < np.inf else 0

def gradient_projection_method(    
    G: csr_matrix,
    c: np.ndarray[np.double],
    l: np.ndarray[np.double],
    u: np.ndarray[np.double],
    x: np.ndarray[np.double],
    tol: np.double,
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], np.ndarray[np.double], int]:
    iter = 1    

    n = len(x)

    A = eye(n, format='csr')

    d = G.diagonal()

    def preconditioned_solve(r: np.ndarray[np.double]) -> np.ndarray[np.double]:
        y = r.copy()

        for i in range(len(y)):
            q = abs(d[i])

            if q > tol:
                y[i] /= q

        return y
    
    f = False

    while iter < max_iter:
        if not f:            
            x = cauchy_point(G, c, l, u, x)
                
            g = G @ x + c

            Wset = []

            for i in range(len(x)):
                if x[i] == l[i] or x[i] == u[i]:
                    Wset.append(i)        

        if len(Wset) == 0:
            s = np.array([])
            y = np.zeros_like(c)

            p, _ = preconditioned_cg(preconditioned_solve, G, -g, y, tol, max_iter)
        else:
            W = A[Wset, :]
            y = np.zeros(len(Wset))            

            p, s, _ = projected_cg(W, G, g, y, tol, max_iter)  

        if np.linalg.norm(p, ord=np.inf) < tol:
            f = True

            if is_positive(s):
                return x, s, iter
            else:
                i = np.argmin(s)
                del Wset[i]
        else:
            t = np.inf

            for i in range(len(p)):
                if p[i] > 0 and u[i] < np.inf:
                    t = min(t, (u[i] - x[i])/p[i])
                elif p[i] < 0 and l[i] > -np.inf:
                    t = min(t, (l[i] - x[i])/p[i])

            if t > tol:
                x += t*p                
                f = False  
            else:
                if is_positive(s):
                    return x, s, iter
                else:
                    i = np.argmin(s)
                    del Wset[i]

                f = True
                          
        iter += 1

    return x, s, iter

def cauchy_point(
    G: csr_matrix,
    c: np.ndarray[np.double],
    l: np.ndarray[np.double],
    u: np.ndarray[np.double],
    x: np.ndarray[np.double]        
) -> np.ndarray[np.double]:
    s = set()
    d = np.zeros(len(x))

    g = G @ x + c

    for i in range(len(x)):
        t = 0
        if g[i] < 0 and u[i] < np.inf:
            t = (x[i] - u[i])/g[i]
        elif g[i] > 0 and l[i] > -np.inf:
            t = (x[i] - l[i])/g[i]
        else:
            t = np.inf

        d[i] = t
        s.add(t)

    tl = sorted(s)

    p = -g

    i = -1
    t = 0

    y = x.copy()

    while i < len(tl) - 1:
        update_y_and_p(t, d, g, x, y, p)

        gp = G @ p        
        ygp = np.dot(y, gp)
        cp = np.dot(c, p)
            
        df = cp + ygp

        if df > 0:
            break

        d2f = np.dot(p, gp)

        dt = -df/d2f

        if t + dt <= tl[i+1]:
            update_y_and_p(t + dt, d, g, x, y, p)

            break

        i += 1
        t = tl[i]

    if i == len(tl) - 1 and t + dt > tl[i]:
        update_y_and_p(t + dt, d, g, x, y, p)

    return y

def update_y_and_p(
    t: np.double,
    d: np.ndarray[np.double],
    g: np.ndarray[np.double],
    x: np.ndarray[np.double],
    y: np.ndarray[np.double],
    p: np.ndarray[np.double]
):
    for j in range(len(x)):
        if d[j] >= t:
            y[j] = x[j] - t * g[j]
        else:
            y[j] = x[j] - d[j] * g[j]
            p[j] = 0.
