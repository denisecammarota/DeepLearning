#comienzo de loss.py
#lo codeo aca y despues lo paso a otro archivo

import numpy as np

class Loss:
    def __init__(self):
        pass
    def __call__(self,scores,y):
        pass
    def gradient(self,scores,y):
        pass
    
class MSE(Loss):
    def __init__(self):
        super().__init__()
    def __call__(self,scores,y):
        mse = np.mean(np.sum((scores-y)**2,axis=1))
        return mse
    def gradient(self,scores,y):
        gradmse = (scores-y)*2
        return gradmse
    
class CCE(Loss):
    def __init__(self):
        super().__init__()
    def __call__(self,scores,y):
        scoresmax = scores.max(axis=1) #maximo de cada fila
        scores = scores - scoresmax[:,np.newaxis] #le resto a cada fila el maximo correspondiente
        y = np.argmax(y,axis=1)
        scores_yi = scores[np.arange(scores.shape[0]),y] #estos son los f_yi
        expscores = np.exp(scores) #hace exp(scores)
        sum_expscores = expscores.sum(axis=1) #suma de los exp(scores) por fila 
        loss = np.log(sum_expscores) - scores_yi #aca estan las loss_i en vector fila
        loss = loss.mean() #vector fila, solo hago el mean, no sobre algun axis particular
        return loss
    def gradient(self,scores,y):
        y = np.argmax(y,axis=1)
        y = y.flatten()
        scoresmax = scores.max(axis=1) #maximo de cada fila
        scores = scores - scoresmax[:,np.newaxis] #le resto a cada fila el maximo correspondiente
        scores_yi = scores[np.arange(scores.shape[0]),y] #estos son los f_yi
        expscores = np.exp(scores) #hace exp(scores)
        sum_expscores = expscores.sum(axis=1) #suma de los exp(scores) por fila 
        grad = (1/sum_expscores)[:,np.newaxis]*expscores
        grad[np.arange(y.shape[0]),y] = grad[np.arange(y.shape[0]),y] - 1
        return grad
    
