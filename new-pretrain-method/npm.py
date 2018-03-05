#coding=utf-8
import numpy as np
import tensorflow as tf
import random
import sys,os
import os
from tensorflow.contrib.keras.api.keras.models import Sequential,Model,load_model
from tensorflow.contrib.keras.api.keras.layers import Dense,Activation,Reshape
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.layers import Conv2D,Flatten
from tensorflow.contrib.keras.api.keras.layers import MaxPooling2D,AveragePooling2D
from tensorflow.contrib.keras.api.keras.layers import LocallyConnected2D
from tensorflow.contrib.keras.api.keras.layers import GRU,LSTM,ConvLSTM2D
from tensorflow.contrib.keras.api.keras.layers import Embedding
from tensorflow.contrib.keras.api.keras.layers import Add,Multiply,Average,Maximum,Concatenate,Dot
from tensorflow.contrib.keras.api.keras.layers import add,multiply,average,maximum,concatenate,dot
from tensorflow.contrib.keras.api.keras.layers import LeakyReLU,PReLU,ELU,ThresholdedReLU
from tensorflow.contrib.keras.api.keras.layers import BatchNormalization as BN
from tensorflow.contrib.keras.api.keras.layers import GaussianNoise,GaussianDropout,AlphaDropout
from tensorflow.contrib.keras.api.keras.layers import TimeDistributed,Bidirectional
import numpy as np
from numpy import sin,cos,tanh,sinh,random,sum
ctv=('softmax','elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear')
metrics=('mae','acc','accuracy')
optimizers=('sgd','adam','rmsprop','adagrad','adadelta','adamax','nadam',)

def samples(N=100):
	X=random.rand(N,5)
	Y=np.zeros([N,2])
	Y[:,0]=X[:,0]+X[:,1]+X[:,2]
	Y[:,1]=X[:,2]+X[:,3]+X[:,4]
	Y/=3.
	return X,Y

def LoadModel():
	i=Input(shape=(5,))
	a=Dense(10,activation='relu')(a)
	a=Dense(10,activation='relu')(a)
	a=Dense(10,activation='relu')(a)
	a=Dense(5,activation='relu')(a)
	a=Dense(5,activation='relu')(a)
	a=Dense(5,activation='relu')(a)
	o=Dense(2,activation='sigmoid')(a)
	D=Model(inputs=i,outputs=o)
	D.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
	return D

def TrainModel(D):
	for i in range(3000):
		if i%5==0:
			X,Y=samples(10000)
			err,accu=D.evaluate(X,Y)
			err=1.5e4*err**0.5
			print(i,err,accu)

if __name__=='__main__':
	D=LoadModel()
	TrainModel(D)
