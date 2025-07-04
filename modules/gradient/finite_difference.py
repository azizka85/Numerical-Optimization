import numpy as np
from typing import Callable

class FDGradient:
    __f: Callable[[np.ndarray[np.double]], np.double]
    __epsilon: np.double

    def __init__(
        self, 
        f: Callable[[np.ndarray[np.double]], np.double],
        epsilon: np.double
    ):
        self.__f = f
        self.__epsilon = epsilon

    def __call__(self, x: np.ndarray[np.double]) -> np.ndarray[np.double]:
        n = len(x)
        p = np.zeros_like(x)                

        for i in range(n):
            x[i] += self.__epsilon

            fxp1 = self.__f(x)

            x[i] -= 2*self.__epsilon

            fxm1 = self.__f(x)

            x[i] += self.__epsilon
            
            p[i] = (fxp1 - fxm1) / 2 / self.__epsilon            

        return p

    
