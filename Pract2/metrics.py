#comienzo de metrics.py
#solo necesito poder evaluar con las metricas
#las hago como funciones

import numpy as np

#ambas reciben los scores de la prediccion (scores) y los scores verdaderos (y)
#retorna la metrica correspondiente

def accuracy(scores,y): #el del ej3
    y_pred = np.argmax(scores,axis=1) #selecciona categoria correcta
    y_true = np.argmax(y,axis=1) #selecciona categoria correcta
    acc = np.mean(y_pred == y_true) #calculo accuracy
    return acc

def MSE(scores,y):
    mse = np.mean(np.sum((scores-y)**2,axis=1))
    return mse

#esta es especifica para el problema del XOR
def accuracy_xor(scores,y):
    scores[scores>0.9] = 1 #umbral que aconsejaron en clase
    scores[scores<-0.9] = -1 #idem 
    acc = np.mean(scores==y)
    return acc
