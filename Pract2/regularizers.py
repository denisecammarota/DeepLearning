import numpy as np
import models as models
import activations as activations
import optimizers as optimizers
import losses as losses
import metrics as metrics
import layers as layers

class Regularization:
    def __init__(self,rg):
        self.rg = rg
    def __call__(self,layer):
        pass
    def gradient(self,layer):
        pass

class L2(Regularization):
    def __init__(self,rg):
        super().__init__(rg)
    def __call__(self,layer):
        res = self.rg * 0.5 * np.sum((layer.W)**2)
        return res
    def gradient(self,layer):
        res = self.rg * layer.W
        return res
    
    
class Zero(Regularization):
    def __init__(self,rg):
        super().__init__(rg)
    def __call__(self,model):
        return 0
    def gradient(self,model):
        return 0
    
