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

def samples(N=120,M=10,cid=0):
    #网关相对于信号源的距离，是一个正态分布,最大传输距离为1，
    #由于正态分布在[-3delta，3delta]的概率为99.74%，故可令delta=1/3
    #极坐标下的极半径与极角
    R=np.random.normal(0,1/15.,(N*M)).reshape(N,M)
    THEATA=np.random.rand(N,M)  
    THEATA*=(2*np.pi)
    #转换成为直角坐标系
    D=np.zeros([N,M,2]) 
    D[:,:,0]=R*np.cos(THEATA)
    D[:,:,1]=R*np.sin(THEATA)
    #输出代表信号源的位置
    #其随机分布在[-1,1]之间
    Y=random.rand(N,2)
    Y*=2
    Y-=1
    #X张量最后最后一维分别表示(基站的x坐标，基站的y坐标，基站到信号源的相对距离,基站信息是否有效)
    X=np.zeros([N,M,4])
    X[:,:,:2]=D+Y[:,np.newaxis,:]
    X[:,:,2]=R+np.random.normal(0,0.002,N*M).reshape(N,M)
    #L=np.min(X[:,:,2],axis=1)
    #X[:,:,2]-=L[:,np.newaxis]
    #X[:,:,:2]+=np.random.normal(0,0.03,(N*M*2)).reshape(N,M,2)
    #随机选择几个基站为无效基站,基站有效个数是一个二项分布
    binom=np.random.binomial(10,0.75,N)
    for k in range(N):
        idx=cid if cid>=1 else binom[k]
        idx=idx if idx>0 else 10
        X[k,:idx,3]=1
        X[k,:idx,2]-=np.min(X[k,:idx,2])
        X[k,idx:,:]=0
    return X,Y

X,Y=samples(1,10)
print('X:',X.reshape(1,10,4))
print('Y:',Y)
#sys.exit()

D=None
M=10
def LoadModel():
	global D
	D=None
	D=load_model('./tdoa1.h5')
	if D==None:
		i=Input(shape=(M,4))
		print("1=====",i)
		a=Flatten()(i)
		a=Dense(200,activation='relu')(a)
		a=Dense(160,activation='relu')(a)
		a=Dense(120,activation='relu')(a)
		a=Dense(80,activation='relu')(a)
		a=Dense(80,activation='relu')(a)
		a=Dense(80,activation='relu')(a)
		a=Dense(60,activation='relu')(a)
		a=Dense(60,activation='relu')(a)
		a=Dense(60,activation='relu')(a)
		a=Dense(60,activation='relu')(a)
		a=Dense(60,activation='relu')(a)
		a=Dense(60,activation='relu')(a)
		a=Dense(60,activation='relu')(a)
		a=Dense(60,activation='relu')(a)
		a=Dense(40,activation='relu')(a)
		a=Dense(40,activation='relu')(a)
		a=Dense(40,activation='relu')(a)
		a=Dense(40,activation='relu')(a)
		a=Dense(40,activation='relu')(a)
		a=Dense(40,activation='relu')(a)
		a=Dense(20,activation='relu')(a)
		a=Dense(10,activation='relu')(a)
		o=Dense(2,activation='tanh')(a)
		D=Model(inputs=i,outputs=o)
		D.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

def TrainModel():
	for i in range(3000):
		#X,Y=samples(N=800000,M=M)
		#D.train_on_batch(X,Y)
		if i%5==0:
			#D.save('./tdoa1.h5')
			for j in range(1,M+1):
				X,Y=samples(100000,M,j)
				err,accu=D.evaluate(X,Y)
				err=1.5e4*err**0.5
				print(i,j,err,accu)

def predict():
	X,Y=samples(N=10,M=M)
	Y_=D.predict(X)
	print('X:',X.reshape(10,M,4))
	print('Y:',Y)
	print('Y_:',Y_)

if __name__=='__main__':
	LoadModel()
	TrainModel()
	predict()
