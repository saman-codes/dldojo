# Thirdparty
import numpy as np

class Operations():

    @staticmethod
    def minmax_scaling(x):
        return np.divide((x - np.amin(x)),(np.amax(x) - np.amin(x)))