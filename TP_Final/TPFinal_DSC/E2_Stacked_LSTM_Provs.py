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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
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


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


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
    plt.legend(fontsize=14)
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
tw = 3


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
        x_train_total, y_train_total = create_dataset(train_data,tw)
        x_test_total, y_test_total = create_dataset(test_data,tw)
    else:
        #train
        x_train, y_train = create_dataset(train_data,tw)
        x_train_total = np.vstack((x_train_total,x_train))
        y_train_total = np.hstack((y_train_total,y_train))
        #test
        x_test, y_test = create_dataset(test_data,tw)
        x_test_total = np.vstack((x_test_total,x_test))
        y_test_total = np.hstack((y_test_total,y_test))
            


# In[12]:


x_train_total = x_train_total.reshape(x_train_total.shape[0],1,x_train_total.shape[1])
x_test_total = x_test_total.reshape(x_test_total.shape[0],1,x_test_total.shape[1])


# In[13]:


model = keras.Sequential()
model.add(keras.layers.LSTM(units=128,activation='relu',return_sequences=True,input_shape=(1,tw)))
model.add(keras.layers.LSTM(units=128,activation='relu'))
model.add(keras.layers.Dense(units=1))
model.compile(optimizer='adam',loss=keras.losses.MSE,metrics=['mse']) 
model.summary()
history = model.fit(x_train_total, y_train_total,epochs=1000,batch_size=256,validation_data=(x_test_total,y_test_total),verbose=2) 


# In[14]:


plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend(fontsize=14)


# In[15]:


y_train_pr = model.predict(x_train_total)
y_test_pr = model.predict(x_test_total)
y_train_pr = scaler.inverse_transform(y_train_pr.reshape(-1,1))
y_test_pr = scaler.inverse_transform(y_test_pr.reshape(-1,1))
y_test =  scaler.inverse_transform(y_test_total.reshape(-1,1))
y_train = scaler.inverse_transform(y_train_total.reshape(-1,1))
print('train mse squared:',mean_squared_error(y_train_pr,y_train)) 
print('test mse squared:',mean_squared_error(y_test_pr,y_test)) 


# # veo como se ajustan a los train y test data para distintas localidades

# In[21]:


def forecast(test_data_scaled,df1,name):
    n = len(test_data_scaled)
    lag = tw
    x_input=test_data_scaled[n-lag:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=lag
    i=0

    while(i<10): 
        if(len(temp_input)>lag):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, 1, lag))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, 1, lag))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    
    lst_output = scaler.inverse_transform(lst_output)
    plt.plot(lst_output,'o',label='forecast')
    plt.plot(df1,label='true data')
    plt.title(dic_localidades[str(name)])
    plt.xlabel('Días posteriores')
    plt.ylabel('Casos/100mil hab')
    plt.legend(fontsize=14)
    plt.savefig(dic_localidades[str(name)]+'_forecast_2.pdf')
    print('forecast error:',sum((lst_output-df_forecast)**2))
    plt.show()


# In[22]:


for file in files:
    print(file)
    data = pd.read_csv(mypath+str('/')+file,sep=",",quotechar='"',na_values=[''])
    data = data["incidenciaAcum14d"]
    df = pd.DataFrame(data)
    df = df.to_numpy()
    df_forecast = df[-20:-10]
    df = df[:-20]
    df_original = np.copy(df)
    total_size = df.shape[0]
    train_size = int(0.8*total_size)
    test_size = total_size - train_size
    train_data = df[:-test_size]
    test_data = df[-test_size:]
    train_data = scaler.transform(train_data.reshape(-1,1))
    test_data = scaler.transform(test_data.reshape(-1,1))
    #train
    x_train, y_train = create_dataset(train_data,tw)
    #test
    x_test, y_test = create_dataset(test_data,tw)
    x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
    x_test= x_test.reshape(x_test.shape[0],1,x_test.shape[1])
    y_train_pr = model.predict(x_train)
    y_test_pr = model.predict(x_test)
    y_train_pr = scaler.inverse_transform(y_train_pr.reshape(-1,1))
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
    graph_predictions(tw,df_original,y_train_pr,y_test_pr,file)
    forecast(test_data,df_forecast,file)


# In[ ]:




