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
from sklearn import preprocessing
import loss


data = pd.read_csv('./testing_files/Regression.csv')

X = data.drop('Chance of Admit ',axis=1).values
y = data['Chance of Admit '].values
'''
X = data.drop('0',axis=1).values
y = data['0'].values
'''
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)

X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)


X_train = tensor(X_train)
X_test = tensor(X_test)


layer1 = Dense(X_train,3,activation_function=nn.relu)
layer2 = Dense(layer1(),2,activation_function=nn.relu)
layer3 = Dense(layer2(),1,activation_function=nn.linear)
list_of_layers = [layer1,layer2,layer3]
lr = 0.015
assembler = train.NeuralNetwork(list_of_layers,loss.MSE())
'''
n_epochs = 2
for epoch in range(n_epochs):
    print(assembler.layers[0].weights_initializer,'WEIGHTS')
    print(assembler.layers[1].weights_initializer,'WEIGHTS OF SECOND LAYER')
    assembler.forward(X_train)
    assembler.backward(y_train)
    print(epoch,'This is number of epochs passed!')
    print(assembler.layers[0].weights_initializer,'Weights')
    print(assembler.layers[1].weights_initializer,'2nd layer')
'''
assembler.train(X_train,y_train,500)

# On this step we have working neural network with 
# good updated weights and its ready to make predictions
prediction = assembler.forward_random(X_test).reshape(y_test.shape[0],1)
# Have to write another method forward_think
# in Dense class (then rewrite forward method)!
metricss = metrics.mean_squared_error(y_test,prediction)
print(metricss,'This is MSE of our neural network')
print(prediction[:7],'This is predicted')
print(y_test[:7])
print(assembler.layers[0].weights_initializer)
