#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler #scaling de los datos entre 0 y 1
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from os import listdir
from os.path import isfile, join
plt.style.use('seaborn')
plt.style.use('matplotlibrc.py')


# In[2]:


dic_localidades = {
      'RiesgoBariloche':'Bariloche',
      'RiesgoBuenosAires':'Buenos Aires',
      'RiesgoCABACABANA':'CABA',
      'RiesgoChacoNA':'Chaco',
      'RiesgoCórdobaCórdoba':'Córdoba',
      'RiesgoEntreRiosRíos':'Entre Ríos',
      'RiesgoJujuyJujuy':'Jujuy',
      'RiesgoLaRiojaRioja':'La Rioja',
      'RiesgoMendozaMendoza':'Mendoza',
      'RiesgoNeuquénNeuquén':'Neuquén',
      'RiesgoRioNegro':'Río Negro',
      'RiesgoSaltaSalta':'Salta',
      'RiesgoSantaCruzSantaCruz':'Santa Cruz',
      'RiesgoSantaFeSantaFe':'Santa Fe',
      'RiesgoTierradelFuegoTierradel':'Tierra del Fuego',
      'RiesgoTucumanTucuman':'Tucumán'
  }


# In[3]:


def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		if out_end_ix > len(sequence):
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


# In[4]:


def graph_predictions(l,df_original,y_train_pr,y_test_pr,name):
    plt.plot(df_original) 
    months_tr = np.arange(l,len(y_train_pr)+l) #meses de training
    months_ts = np.arange(len(y_train_pr)+(2*l)+1,len(df_original)-1) #meses de testing
    plt.plot(months_tr,y_train_pr,label='train') #grafico de train results
    plt.plot(months_ts,y_test_pr,label='test') #grafico de test results
    plt.title(dic_localidades[str(name)])
    plt.xlabel('Días')
    plt.ylabel('Casos/100 mil hab')
    plt.legend(fontsize=12)
    plt.savefig(dic_localidades[str(name)]+'_fit_2.pdf')
    plt.show()


# In[5]:


seed = 7
np.random.seed(seed)


# In[6]:


mypath = 'Datos'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))] #get all file names of that path
df_train_total = [] #aca guardamos todos los datos


# In[7]:


scaler = MinMaxScaler(feature_range=(0, 1))
n_steps_in = 10
n_steps_out = 10


# # armo scaler para los datos

# In[8]:


for file in files:
    print(file)
    data = pd.read_csv(mypath+str('/')+file,sep=",",quotechar='"',na_values=[''])
    data = data["incidenciaAcum14d"]
    df = pd.DataFrame(data)
    df = df.to_numpy()
    df = df[:-20]
    total_size = df.shape[0]
    train_size = int(0.8*total_size)
    test_size = total_size - train_size
    train_data = df[:-test_size]
    df_train_total.extend(list(train_data.flatten()))


# In[9]:


scaler = MinMaxScaler(feature_range=(0, 1))
df_train_total = scaler.fit_transform(np.array(df_train_total).reshape(-1,1))


# # ahora armo los datos de train, test y forecast

# In[10]:


df_forecast = [] #aca van a estar los datos para hacer el forecasting


# In[11]:


for file in files:
    print(file)
    data = pd.read_csv(mypath+str('/')+file,sep=",",quotechar='"',na_values=[''])
    data = data["incidenciaAcum14d"]
    df = pd.DataFrame(data)
    df = df.to_numpy()
    df_forecast.append(df[-20:-10])
    df = df[:-20]
    total_size = df.shape[0]
    train_size = int(0.8*total_size)
    test_size = total_size - train_size
    train_data = df[:-test_size]
    test_data = df[-test_size:]
    train_data = scaler.transform(train_data.reshape(-1,1))
    test_data = scaler.transform(test_data.reshape(-1,1))
    if file == 'RiesgoBariloche':
        x_train_total, y_train_total = split_sequence(train_data, n_steps_in, n_steps_out)
        x_test_total, y_test_total = split_sequence(test_data, n_steps_in, n_steps_out)
    else:
        #train
        x_train, y_train = split_sequence(train_data, n_steps_in, n_steps_out)
        x_train_total = np.vstack((x_train_total,x_train))
        y_train_total = np.vstack((y_train_total,y_train))
        #test
        x_test, y_test = split_sequence(test_data, n_steps_in, n_steps_out)
        x_test_total = np.vstack((x_test_total,x_test))
        y_test_total = np.vstack((y_test_total,y_test))
            


# In[12]:


x_train_total = x_train_total.reshape(x_train_total.shape[0],1,x_train_total.shape[1])
x_test_total = x_test_total.reshape(x_test_total.shape[0],1,x_test_total.shape[1])
y_train_total =  y_train_total.reshape(y_train_total.shape[0],y_train_total.shape[1])
y_test_total = y_test_total.reshape(y_test_total.shape[0],y_test_total.shape[1])


# In[13]:


model = keras.Sequential()
model.add(keras.layers.LSTM(units=32,activation='relu',return_sequences=True,input_shape=(1,n_steps_in)))
model.add(keras.layers.LSTM(units=32,activation='relu',return_sequences=False,input_shape=(1,n_steps_in)))
model.add(keras.layers.Dense(units=n_steps_out))
model.compile(optimizer='adam',loss=keras.losses.MSE,metrics=['mse']) 
model.summary()
history = model.fit(x_train_total, y_train_total,epochs=1000,batch_size=256,validation_data=(x_test_total,y_test_total),verbose=2) 


# In[14]:


plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend(fontsize=12)


# In[15]:


y_train_pr = model.predict(x_train_total)
y_test_pr = model.predict(x_test_total)
y_train_pr = scaler.inverse_transform(y_train_pr)
y_test_pr = scaler.inverse_transform(y_test_pr)
y_train =  scaler.inverse_transform(y_train_total)
y_test = scaler.inverse_transform(y_test_total)
print('train mse squared:',mean_squared_error(y_train_total,y_train)) 
print('test mse squared:',mean_squared_error(y_test_total,y_test)) 


# # veo como se ajustan a los train y test data para distintas localidades

# In[24]:


for file in files:
    print(file)
    data = pd.read_csv(mypath+str('/')+file,sep=",",quotechar='"',na_values=[''])
    data = data["incidenciaAcum14d"]
    df = pd.DataFrame(data)
    df = df.to_numpy()
    df = df[:-10]
    df_original = np.copy(df)
    y_forecast = np.copy(df[-n_steps_in:])
    df = df[:-n_steps_in]
    x_toforecast = np.copy(df[-n_steps_in:])
    x_toforecast = scaler.transform(x_toforecast.reshape(-1,1))
    x_toforecast = x_toforecast.flatten()
    x_toforecast = x_toforecast.reshape(1,1,n_steps_in)
    y_forecasted = model.predict(x_toforecast)
    y_forecasted = scaler.inverse_transform(y_forecasted)
    plt.title(dic_localidades[str(file)])
    plt.plot(y_forecasted.flatten(),'o',label='forecasted')
    plt.plot(y_forecast,label='true data')
    plt.legend()
    #ahora metricas de train y test
    total_size = df_original.shape[0]
    train_size = int(0.8*total_size)
    test_size = total_size - train_size
    train_data = df_original[:-test_size]
    test_data = df_original[-test_size:]
    train_data = scaler.transform(train_data.reshape(-1,1))
    test_data = scaler.transform(test_data.reshape(-1,1))
    x_train, y_train = split_sequence(train_data, n_steps_in, n_steps_out)
    x_test, y_test = split_sequence(test_data, n_steps_in, n_steps_out)
    x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])
    y_train =  y_train.reshape(y_train.shape[0],y_train.shape[1])
    y_test = y_test.reshape(y_test.shape[0],y_test.shape[1])
    y_train_pr = model.predict(x_train)
    y_test_pr = model.predict(x_test)
    y_train_pr  = scaler.inverse_transform(y_train_pr.reshape(-1,1))
    y_test_pr = scaler.inverse_transform(y_test_pr.reshape(-1,1))
    y_train = scaler.inverse_transform(y_train.reshape(-1,1))
    y_test = scaler.inverse_transform(y_test.reshape(-1,1)) 
    #imprimo mse para train y test
    print('train mse:',mean_squared_error(y_train,y_train_pr)) 
    print('test mse:',mean_squared_error(y_test,y_test_pr))
    #imprimo r2
    print('train r2:',r2_score(y_train,y_train_pr)) 
    print('test r2:',r2_score(y_test,y_test_pr)) 
    #imprimo mae
    print('train mae:',mean_absolute_error(y_train,y_train_pr)) 
    print('test mae:',mean_absolute_error(y_test,y_test_pr))
    #forecast error
    print('forecast mse:',mean_squared_error(y_forecast.flatten(),y_forecasted.flatten()))
    plt.show()


# In[17]:


tf.keras.utils.plot_model( model, to_file="model.png", show_shapes=False, show_layer_names=True, rankdir="TB")

