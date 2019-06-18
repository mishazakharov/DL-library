''' File for testing my DLibrary! '''
from layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensor import tensor
import nn
import loss


data = pd.read_csv('./testing_files/Regression.csv')

X = data.drop('Chance of Admit ',axis=1).values
y = data['Chance of Admit '].values

X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.20)

X_train = tensor(X_train)


layer1 = Dense(X_train,3,activation_function=nn.relu)
layer2 = Dense(layer1(),1,activation_function=nn.relu)
mse = loss.mean_squared_error(y_train,layer2())


print(X_train.shape)
print(mse,'This is my MSE')





