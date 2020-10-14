import keras 
import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
from keras.callbacks import History 
import matplotlib.pyplot as plt
from keras.datasets import imdb #cargo los datos del imdb
from keras.utils import to_categorical


def randomize(x,y):
    #x matriz de datos (datos x dimension)
    #y matriz de datos verdaderos (datos x categorias)
    indices = np.random.choice(x.shape[0], x.shape[0], replace=False)
    x = x[indices]
    y = y[indices]
    

#reshape como quiero a la data de la bd IMDB
def reshapeData(x_total,y_total,n_palabras):
    y = to_categorical(y_total) #queda dimensiones reviews x 2
    x = np.zeros((len(x_total), n_palabras)) #queda dimensiones reviews x n_palabras
    for i, x_total in enumerate(x_total):
        x[i, x_total] = 1 #un 1 en posicion en la que hay una palabra, queda cero en otro caso
    return x,y

def separateData(x_total,y_total):
    n_review = y_total.shape[0]
    n_train = int(n_reviews * 0.75) #cantidad de datos de train
    randomize(x_total,y_total)
    x_train = x_total[:n_train,:]
    y_train = y_total[:n_train]
    x_test = x_total[n_train:,:]
    y_test = y_total[n_train:]
    return x_train,y_train,x_test,y_test

#cargo los datos como dice en el enunciado
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000) #solo las 10k palabras mas frecuentes en reviews 
                                                                         #asi que hay 10k palabras posibles por review
                                                                         
#cuantos datos de train y test hay por default?
print('Datos de train originales:',y_train.shape[0]) #dice 25000 (1/2 del total)
print('Datos de test originales:',y_test.shape[0]) #dice 25000 (1/2 del total)
n_reviews = y_train.shape[0] + y_test.shape[0] #cantidad total de datos (= 50000)
#junto todos los datos para reformatear y separar bien despues como en los otros ejercicios
x_total = np.hstack((x_train,x_test)) #lista de palabras de reviews de las peliculas codificadas segun diccionario
y_total = np.hstack((y_train,y_test)) #calificacion positiva (1) o calificacion negativa (0)
#cuantas palabras tiene cada review? todas formateadas a 10k?
#por ejemplo...
print("Palabras en primera review: ",len(x_total[0])) #218 palabras
print("Palabras en decima review: ",len(x_total[9])) #130 palabras
#bueno, claramente no entonces (no es que tiene 1 si esta la palabra y 0 sino, que seria deseable)
n_palabras = 10000 #cantidad de palabras posibles
x_total, y_total = reshapeData(x_total,y_total,n_palabras)
x_train,y_train,x_test,y_test = separateData(x_total,y_total) #separo en datos de train y test


#defino algunas constantes que son utiles, las mismas de los problemas del ejercicio 2
n_capa1 = 100  
n_capa2 = 10 
n_capa3 = 2
lr = 1e-3
epocas = 100 #cantidad de epocas


#entonces puedo proceder a armar la red neuronal parecido al ejercicio anterior
#porque el formato de los datos es el mismo
x = keras.layers.Input(shape=(n_palabras,)) #capa de input
l1 = keras.layers.Dense(units=n_capa1, activation='relu')(x)
l2 = keras.layers.Dense(units=n_capa2, activation='relu')(l1)
l3 = keras.layers.Dense(units=n_capa3, activation='sigmoid')(l2)
model = keras.Model(inputs=x, outputs=l3)
optimizer = keras.optimizers.SGD(learning_rate=lr)
model.compile(optimizer, loss=keras.losses.MSE, metrics=['acc'])
history = model.fit(x_train, y_train, epochs=epocas, validation_data=(x_test, y_test),batch_size=100, verbose=2)                                                                       