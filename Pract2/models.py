
#comienzo de models.py
#lo codeo aca y despues lo paso a otro archivo

import activations as activations
import optimizers as optimizers
import losses as losses
import metrics as metrics
import layers as layers
import numpy as np
import regularizers as regularizers

class Network():
    def __init__(self):
        self.list_neuronas = [] #lista de neuronas
        self.grad = None #variable auxiliar, despues del fit calculo el gradiente con la loss y despues lo termino en backward
        self.optm = None #para poder usarlo en backwards para actualizar los pesos
        self.regu = None
    def add(self,capa_neuronas):
        #chequeo si es la primera capa
        if (len(self.list_neuronas) != 0):
            output_old_neurona = (self.list_neuronas[len(self.list_neuronas)-1]).get_output_shape()
            capa_neuronas.set_input_shape(output_old_neurona) #seteo el input de la nueva neurona
        if(isinstance(capa_neuronas,layers.Concat)):
            pass
        else:
            capa_neuronas.init_weights() #inicializo
        self.list_neuronas.append(capa_neuronas) #aniado neurona a la lista
    def get_layer(self,numero_layer): #devuelve la layer dado el numero de la layer que quiero
        return self.list_neuronas[numero_layer] #empieza desde la capa 0
    def fit (self,x,y,x_test=None,y_test=None,lr = 1e-3,epochs = 100,bs = 100,acc = metrics.accuracy_xor, loss_class = losses.MSE,opt_class = optimizers.SGD,regu = regularizers.Zero,rg=1e-2):
        x_test = np.array(x_test)
        optm = opt_class(lr,bs) #creo objeto de optimizador, hace loop de batchs
        self.optm = optm
        self.regu = regu(rg)
        loss = loss_class() #creo objeto loss  
        loss_tr = [] #vector de loss de datos de training
        loss_ts = [] #vector de loss de datos de testing
        acc_tr = [] #vector de accuracy de datos de training
        acc_ts = [] #vector de accuracy de datos de testing
        for ie in range(epochs):
            print("epoch: "+str(ie))
            #forward la primera vez, devuelvo el gradiente de la loss escencialmente
            #backward path
            optm(x,y,self,loss) #hago call del optimizador, que hace backwards una vez y update de los pesos
            #calculo loss y accuracy para los datos de training
            scores = self.return_scores(x) #aniado esta funcion que me devuelve los scores solamente despues de actualizar
            loss_aux = loss(scores,y) + self.loss_rg() #calculo loss de los datos de training
            acc_aux = acc(scores,y) #calculo accuracy de los datos de training
            loss_tr.append(loss_aux) #aniado la loss de los de training 
            acc_tr.append(acc_aux)  #aniado la accuracy de los de training
            print("training data: ",loss_aux,acc_aux) #printeo loss y acc de training data
            #hago lo mismo para los datos de testing basicamente
            if(x_test != ()):
                scores = self.return_scores(x_test)
                loss_aux = loss(scores,y_test) + self.loss_rg()
                acc_aux = acc(scores,y_test)
                loss_ts.append(loss_aux)
                acc_ts.append(acc_aux)
                print("testing data: ",loss_aux,acc_aux) #printeo loss y acc de testing data
        return loss_tr,loss_ts,acc_tr,acc_ts
    def forward_upto(self,x_input,j): 
        #hace forward hasta la capa j-esima 
        s_i = np.copy(x_input)
        for i in range(j+1):
            if(isinstance(self.list_neuronas[i],layers.Concat)):
                s_i = self.list_neuronas[i](x_input,s_i)
            else:
                s_i = self.list_neuronas[i](s_i) #estos son los s_i que salen de cada neurona 
        return s_i #devuelve el resultado de forward up to j-esima capa
    def return_scores(self,x): #retorna los scores
        scores = self.forward_upto(x,len(self.list_neuronas)-1)
        return scores
    def predict(self,x): #calcula los scores con la funcion anterior y devuelve la prediccion
        scores = self.return_scores(x)
        y = np.argmax(scores,axis=1)
        return y
    def backward(self,x,y,lr):
        #le paso self, los datos x,y y lr es el learning rate porque a esta funcion la llama el opt
        #y me parece que es la manera mas facil de actualizar los pesos 
        n_capas = len(self.list_neuronas) #cantidad total de capas totales
        grad = self.grad #idem anterior
        for j in reversed(range(n_capas)): #recorro hasta la capa 0 de input
            capa_actual = self.get_layer(j) #capa actual de trabajo
            capa_anterior = self.get_layer(j-1)
            s_i_ant = self.forward_upto(x,j-1) #s_i de la capa i-1, capa anterior
            if(isinstance(capa_actual,layers.Concat)):
                grad = capa_actual.grad_concat(grad)
            else:
                #sino es una de concatenacion (o sea, es densa basicamente)
                grad = grad * capa_actual.activacion.gradient(capa_actual.y_i)
                s_i_ant = capa_actual.addBias(s_i_ant) 
                grad_wi = (s_i_ant.T).dot(grad) + self.regu.gradient(capa_actual)
                grad = grad.dot(capa_actual.W.T)
                grad = grad[:,1:]
                capa_actual.W = self.optm.update_weights(capa_actual.W,grad_wi)
    def loss_rg(self):
        res = 0
        for i in range(len(self.list_neuronas)):
            res += self.regu(self.list_neuronas[i])
        return res
 
     
            
