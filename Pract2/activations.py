
#comienzo de activation.py
#lo codeo aca y despues lo paso a otro archivo

import numpy as np

#una estructura similar a la de Loss
class Activation:
    def __init__(self):
        pass
    def __call__(self,x):
        pass
    def gradient(self,x):
        pass

class ReLu(Activation):
    def __init__(self):
        super().__init__()
    def __call__(self,x):
        return np.maximum(0,x)
    def gradient(self,x):
        res = np.where(x>=0,1,0)
        return res

class Tanh(Activation):
    def __init__(self):
        super().__init__()
    def __call__(self,x):
        return np.tanh(x)
    def gradient(self,x):
        y = (np.cosh(x))**(-2)
        return y

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
    def __call__(self,x):
        sig = (1+np.exp(-x))**(-1)
        return sig
    def gradient(self,x):
        gsig = (np.exp(-x))/((1+np.exp(-x))**2)
        return gsig
    
class Linear(Activation):
    def __init__(self):
        super().__init__()
    def __call__(self,x):
        return x
    def gradient(self,x):
        return 1
