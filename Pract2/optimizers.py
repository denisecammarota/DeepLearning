#comienzo de optimizers.py
#lo codeo aca y despues lo paso a otro archivo

import numpy as np

class Optimizer():
    def __init__(self,lr):
        self.lr = lr #el learning rate
        #podria eventualmente haber mas cosas aca
    def __call__(self, X, Y, model):
        pass
    def update_weights(self, W, gradW):
        pass

class SGD(Optimizer):
    def __init__(self, lr, bs):
        super().__init__(lr)
        self.bs = bs
    def randomizeMatrixRows(self,x,y):
        #x matriz de imagenes (imagenes x dimension)
        #y matriz de scores verdaderos (imagenes x categorias)
        indices = np.random.choice(x.shape[0], x.shape[0], replace=False)
        x1 = x[indices]
        y1 = y[indices]
        return x1,y1
    def __call__(self, X, Y, model,loss):
        nit = int(X.shape[0]/self.bs) #numero de iteraciones
        X1, Y1 = self.randomizeMatrixRows(X,Y) #shuffle de las imagenes
        for j in range(nit):
            X_batch = X1[j*self.bs:(j+1)*self.bs,:] #seleccion de batch
            Y_batch = Y1[j*self.bs:(j+1)*self.bs]
            scores = model.return_scores(X_batch) #variable auxiliar para el calculo de despues 
            model.grad = loss.gradient(scores,Y_batch)  #esto es para el backwards despues, gradiente del ultimo paso del loss
            model.backward(X_batch, Y_batch,self.lr) #backwards de model
    def update_weights(self, W, gradW):
        W -= self.lr * gradW # SGD, paso de optimizacion despues de calcular gradW
        return W
