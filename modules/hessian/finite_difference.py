import numpy as np
from typing import Callable

class FDHessian:
    __f: Callable[[np.ndarray[np.double]], np.double]
    __epsilon: np.double

    def __init__(
        self, 
        f: Callable[[np.ndarray[np.double]], np.double],
        epsilon: np.double
    ):
        self.__f = f
        self.__epsilon = epsilon

    def __call__(self, x: np.ndarray[np.double]) -> np.ndarray[np.ndarray[np.double]]:
        n = len(x)
        H = np.zeros((n, n))         

        for i in range(n):           
            for j in range(n):
                x[i] += self.__epsilon
                x[j] += self.__epsilon

                fx3 = self.__f(x)

                x[j] -= 2*self.__epsilon

                fx2 = self.__f(x)

                x[i] -= 2*self.__epsilon
                x[j] += 2*self.__epsilon

                fx1 = self.__f(x)
                
                x[j] -= 2*self.__epsilon

                fx = self.__f(x)

                x[i] += self.__epsilon
                x[j] += self.__epsilon

                H[i][j] = (fx3 - fx2 - fx1 + fx) / 4 / self.__epsilon / self.__epsilon

        return H
