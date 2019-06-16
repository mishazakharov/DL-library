''' File for testing my DLibrary! '''
from layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensor import tensor
import nn


data = pd.read_csv('Regression.csv')

X = data.drop('Chance of Admit ',axis=1).values
y = data['Chance of Admit '].values

X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.20)

X_train = tensor(X_train)


layer1 = Dense(X_train,3)
layer2 = Dense(layer1(),1)
layer3 = nn.sigmoid(layer2())

print(layer2()[:10])
print(layer3[:10])
print(X_train.shape)







