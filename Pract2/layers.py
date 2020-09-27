#comienzo de layers.py
#lo codeo aca y despues lo paso a otro archivo

import numpy as np

#base para cualquier tipo de layer/capa
class BaseLayer():
    def __init__(self):
        pass
    def set_output_shape(self):
        pass
    def get_output_shape(self):
        pass
    
#clases para capas especiales sin pesos

#clase especial de Layer que es la de Input o entrada
class Input(BaseLayer):
    def __init__(self,dim_input): #donde dim_xi va a ser la dimension del input de toda la red
        super().__init__()
        self.dim_input = dim_input
    def set_output_shape(self,dim):
        self.dim_input = dim
    def get_output_shape(self):
        return self.dim_input
    
#capa de concatenacion que me va a servir para realimentacion
class Concat(BaseLayer):
    def __init__(self,input_layer):
        super().__init__()
        self.dim_1 = input_layer.get_output_shape()
        self.dim_2 = None
        self.dim_output = None
        #la creo como clase vacia en principio
    #concatena los input que le de al llamar al objeto como funcion 
    def __call__(self,x,y):
        #concateno los inputs con los que llamo a la capa de concatenacion
        xy_concat = np.hstack((x,y))
        return xy_concat
    def set_input_shape(self,dim2): #seteo el input shape que va a venir de la neurona anterior
        self.dim_2 = dim2
    def get_input1_shape(self):
        return self.dim_1
    def get_input2_shape(self):
        return self.dim_2
    def get_output_shape(self):
        return self.dim_1 + self.dim_2
    def set_output_shape(self,dim):
        self.dim_output = dim #no entiendo muy bien para que es esto, despues lo veremos
    def grad_concat(self,scores):
        scores = scores[:,self.get_input1_shape():] #creo que asi esta bien, el bias no se lo pongo
        return scores
        
#clases para capas densas con pesos 

#clase padre para capas con pesos WLayer
class WLayer(BaseLayer):
    def __init__(self,n_neuronas,activacion,mg=1,input_dim=0): 
        #input_dim es opcional, sino lo saco de otra capa
        #el resto si lo necesito y es propio de cada capa
        super().__init__()
        self.output_dim = n_neuronas #numero de neuronas, esto es output_dim
        self.activacion = activacion #funcion de activacion de la capa
        self.input_dim = input_dim #dimension del problema, no de los ejemplos
        self.W = None #estos son los pesos de la capa en cuestion
        self.mg = mg
    def get_input_shape(self):
        return self.input_dim
    def set_input_shape(self,input_dim):
        self.input_dim = input_dim
    def get_output_shape(self):
        return self.output_dim
    def set_output_shape(self,output_dim):
        self.output_dim = output_dim
    def init_weights(self): #funcion que inicializa los pesos, idem para todo tipo de cada que podria tener 
        #self.W = np.zeros(input_dim+1,output_dim) el +1 es por bias, recordatorio de las dimensiones
        self.W = np.random.rand(self.input_dim + 1,self.output_dim) * self.mg
    def get_weights(self): #devuelve la matriz de pesos
        return self.W
    def update_weights(self,W_new):
        #susceptible a modificaciones
        #actualiza los pesos, creo que asi esta ok
        self.W = W_new
    def addBias(self,x):
        aux = np.ones((x.shape[0],1))
        x_bias = np.hstack((aux,x))
        return x_bias
        
#clase heredada de WLayer, capa de neuronas densa
class Dense(WLayer):
    def __init__(self,n_neuronas,activacion,mg=1,input_dim=0): 
        #herencia de la clase WLayer, es el mismo constructor
        super().__init__(n_neuronas,activacion,mg,input_dim)
        self.y_i = None
    def dot(self,x): #esto devuelve s_i 
        x_prime = self.addBias(x) #es el x' con el bias aniadido
        y = x_prime.dot(self.W) #multiplicacion
        self.y_i = np.copy(y)
        y = self.activacion(y) #aplica la activacion
        return y
    def __call__(self,x):
        return self.dot(x)
