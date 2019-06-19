''' File for testing my DLibrary! '''
from layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensor import tensor
import nn
import loss
import train
from sklearn import metrics


data = pd.read_csv('./testing_files/Regression.csv')

X = data.drop('Chance of Admit ',axis=1).values
y = data['Chance of Admit '].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)

X_train = tensor(X_train)
X_test = tensor(X_test)


layer1 = Dense(X_train,2,activation_function=nn.relu)
layer2 = Dense(layer1(),1,activation_function=nn.relu)
list_of_layers = [layer1,layer2]
lr = 0.01
assembler = train.NeuralNetwork(list_of_layers,y_train,lr)
n_epochs = 2
for epoch in range(n_epochs):
    assembler.forward()
    assembler.backward()
    print(epoch,'This is number of epochs passed!')

print(X_test.shape,'THIS IS SHAPE OF X_TEST!')
# On this step we have working neural network with 
# good updated weights and its ready to make predictions
prediction = assembler.forward_random(X_test).reshape(y_test.shape[0],1)
# Have to write another method forward_think
# in Dense class (then rewrite forward method)!
print(prediction)
metricss = metrics.mean_squared_error(y_test,prediction)
print(metricss,'This is MSE of our neural network')
print(layer1.weights_initializer)

