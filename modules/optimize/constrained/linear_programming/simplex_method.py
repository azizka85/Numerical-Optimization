import numpy as np
from typing import List, Tuple

def primal(
    c: np.ndarray[np.double],
    A: np.ndarray[np.double, np.double],
    b: np.ndarray[np.double],
    Bset: List[np.int32],
    max_iter: np.int32
) -> Tuple[np.ndarray[np.double], np.double, np.ndarray[np.double], np.int32, np.bool]:
    iter = 1

    Bset.sort()

    B = A[:, Bset]
    cB = c[Bset]

    xB = np.linalg.solve(B, b)
    l = np.linalg.solve(B.transpose(), cB)

    Nset = exclude_indexes(len(c), Bset)

    d = np.zeros(len(xB))
    flag = False

    if len(Nset) > 0:
        N = A[:, Nset]
        cN = c[Nset]

        s = cN - N.transpose() @ l
        si = np.where(s < 0)[0]

        while len(si) > 0 and iter < max_iter:
            iN = np.argmin(s)
            jN = Nset[iN]

            Aq = A[:, jN]

            d = np.linalg.solve(B, Aq)
            di = np.where(d < 0)[0]

            if len(di) > 0:
                flag = True

                break                        
            
            iB = np.argmin(ratio(xB, d))
            jB = Bset[iB]

            del Bset[iB]
            del Nset[iN]

            Bset.append(jN)
            Nset.append(jB)

            B = A[:, Bset]
            cB = c[Bset]

            xB = np.linalg.solve(B, b)
            l = np.linalg.solve(B.transpose(), cB)

            N = A[:, Nset]
            cN = c[Nset]

            s = cN - N.transpose() @ l
            si = np.where(s < 0)[0]

            iter += 1

    return (
        construct_solution(len(c), xB, Bset), 
        calc_obj_fun(c, xB, Bset),
        construct_solution(len(c), d, Bset),
        iter,
        flag
    )

def calc_obj_fun(
    c: np.ndarray[np.double],
    xB: np.ndarray[np.double],
    Bset: List[np.int32]
) -> np.double:
    obj = 0.

    for i in range(len(Bset)):
        j = Bset[i]
        obj += c[j] * xB[i]

    return obj

def construct_solution(
    n: np.int32,
    xB: np.ndarray[np.double],
    Bset: List[np.int32]
) -> np.ndarray[np.double]:
    x = np.zeros(n)

    for i in range(len(Bset)):
        j = Bset[i]
        x[j] = xB[i]

    return x

def exclude_indexes(n: np.int32, Bset: List[np.int32]) -> List[np.int32]:
    Nset = []

    j = 0

    for i in Bset:
        while j < i:
            Nset.append(j)
            j += 1
            
        j += 1

    while j < n:
        Nset.append(j)
        j += 1

    return Nset

def ratio(
    xB: np.ndarray[np.double],
    d: np.ndarray[np.double]            
) -> np.ndarray[np.double]:
    r = np.zeros_like(xB)

    for i in range(len(xB)):
        r[i] =  xB[i] / d[i] if d[i] != 0 else np.inf

    return r
