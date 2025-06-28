import numpy as np

def eigenvalue_modification(
    B: np.ndarray[np.ndarray[np.double]],
    x: np.ndarray[np.double],
    delta: np.double
) -> np.ndarray[np.double]:
    lam, q = np.linalg.eig(B)

    y = np.zeros(len(x))

    for i in range(len(lam)):
        d = np.dot(q[i], x)

        if lam[i] > 0:
            y -= d*q[i]/lam[i]
        else:
            y -= d*q[i]/delta

    return y

