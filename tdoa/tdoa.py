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

def samples(batch=120,gw=10,gwp=20,env=20):
    #网关相对于信号源的距离，是一个正态分布,最大传输距离为1，
    #由于正态分布在[-3delta，3delta]的概率为99.74%，故可令delta=1/3
    #极坐标下的极半径与极角
    R=np.random.normal(0,1/15.,(batch*gw)).reshape(batch,gw)
    THEATA=np.random.rand(batch,gw)  
    THEATA*=(2*np.pi)
    #转换成为直角坐标系
    D=np.zeros([batch,gw,2]) 
    D[:,:,0]=R*np.cos(THEATA)
    D[:,:,1]=R*np.sin(THEATA)
    #输出代表信号源的位置
    #其随机分布在[-1,1]之间
    Y=random.rand(batch,2)
    Y*=2
    Y-=1
    #X张量最后最后一维分别表示(基站的x坐标，基站的y坐标，基站到信号源的相对距离,基站信息是否有效)
    X=np.zeros([batch,gw,gwp])
    X[:,:,:2]=D+Y[:,np.newaxis,:]
    X[:,:,2]=R+np.random.normal(0,0.002,batch*gw).reshape(batch,gw)
    #L=np.min(X[:,:,2],axis=1)
    #X[:,:,2]-=L[:,np.newaxis]
    #X[:,:,:2]+=np.random.normal(0,0.03,(N*M*2)).reshape(N,M,2)
    #随机选择几个基站为无效基站,基站有效个数是一个二项分布
    binom=np.random.binomial(10,0.75,batch)
    for k in range(batch):
        idx=binom[k]
        idx=idx if idx>0 else 10
        X[k,:idx,3]=1
        X[k,:idx,2]-=np.min(X[k,:idx,2])
        X[k,idx:,:]=0
    return np.concatenate([X.reshape(batch,gw*gwp),np.zeros([batch,env])],axis=1),Y

#sys.exit()

class TDOA():
	def __init__(self):
		self.Encoder=None
		self.Decoder=None
		self.Coder=None
		self.Classifier=None
		self.Merger=None
		self.Division=None
		self.Expecter=None
		self.Noliner=None
		self.Model=None

	def LoadEncoder(self):
		if os.path.exists('./Encoder.h5'):
			self.Encoder=load_model('./Encoder.h5')
		if self.Encoder==None:
			i=Input(shape=(220,))
			a=Dense(150,activation='relu')(i)
			a=Dense(100,activation='relu')(a)
			a=Dense(70,activation='relu')(a)
			a=Dense(50,activation='tanh')(a)
			self.Encoder=Model(inputs=i,outputs=a)
			self.Encoder.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
	
	def LoadDecoder(self):
		if os.path.exists('./Decoder.h5'):
			self.Decoder=load_model('./Decoder.h5')
		if self.Decoder==None:
			i=Input(shape=(50,))
			a=Dense(70,activation='relu')(i)
			a=Dense(100,activation='relu')(a)
			a=Dense(150,activation='relu')(a)
			a=Dense(220,activation='tanh')(a)
			self.Decoder=Model(inputs=i,outputs=a)
			self.Decoder.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
	
	def LoadClassifier(self):
		if os.path.exists('./Classifier.h5'):
			self.Classifier=load_model('./Classifier.h5')
		if self.Classifier==None:
			i=Input(shape=(50,))
			a=Dense(60,activation='relu')(i)
			a=Dense(70,activation='relu')(a)
			a=Dense(80,activation='relu')(a)
			a=Dense(100,activation='softmax')(a)
			self.Classifier=Model(inputs=i,outputs=a)
			self.Classifier.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	
	def LoadMerger(self):
		if os.path.exists('./Merger.h5'):
			self.Merger=load_model('./Merger.h5')
		if self.Merger==None:
			i1=Input(shape=(50,))
			i2=Input(shape=(220,))
			a=Concatenate(axis=-1)([i1,i2])
			a=Dense(270,activation='relu')(a)
			a=Dense(500,activation='relu')(a)
			a=Dense(400,activation='relu')(a)
			a=Dense(400,activation='relu')(a)
			a=Dense(300,activation='relu')(a)
			a=Dense(300,activation='relu')(a)
			a=Dense(200,activation='tanh')(a)
			self.Merger=Model(inputs=[i1,i2],outputs=a)
			self.Merger.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	
	def LoadDivision(self):
		if os.path.exists('./Division.h5'):
			self.Division=load_model('./Division.h5')
		i=Input(shape=(200,))
		if self.Division==None:
			h1=[Dense(50,activation='relu')(i) for k in range(100)]
			h2=[Dense(50,activation='tanh')(a) for a in h1]
			h3=[Reshape([1,50])(a) for a in h2]
			o=Concatenate(axis=1)(h3)
			self.Division=Model(inputs=i,outputs=o)
			self.Division.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
	
	def LoadExpecter(self):
		if os.path.exists('./Expecter.h5'):
			self.Expecter=load_model('./Expecter.h5')
		if self.Expecter==None:
			i1=Input(shape=(100,))
			i2=Input(shape=(100,50))
			o=Dot(axes=1)([i1,i2])
			self.Expecter=Model(inputs=[i1,i2],outputs=o)
			self.Expecter.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
		

	def LoadNoliner(self):
		if os.path.exists('./Noliner.h5'):
			self.Noliner=load_model('./Noliner.h5')
		if self.Noliner==None:
			i=Input(shape=(50,))
			a=Dense(40,activation='relu')(i)
			a=Dense(20,activation='relu')(a)
			a=Dense(2,activation='tanh')(a)
			self.Noliner=Model(inputs=i,outputs=a)
			self.Noliner.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
	

	def LoadModel(self):
		self.LoadEncoder()
		self.LoadDecoder()
		self.LoadClassifier()
		self.LoadMerger()
		self.LoadDivision()
		self.LoadExpecter()
		self.LoadNoliner()
		i=Input(shape=(220,))
		ec=self.Encoder(i)
		dc=self.Decoder(ec)
		self.Coder=Model(inputs=i,outputs=dc)
		self.Coder.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
		self.Encoder.Trainable=False
		cls=self.Classifier(ec)
		meg=self.Merger([ec,i])
		div=self.Division(meg)
		exp=self.Expecter([cls,div])
		o=self.Noliner(exp)
		self.Model=Model(inputs=i,outputs=o)
		self.Model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

	def TrainCoder(self):
		for i in range(1000000):
			X,Y=samples(batch=1000,gw=10,gwp=20,env=20)
			self.Coder.train_on_batch(X,X)
			if i%500 == 0:
				self.Encoder.save('./Encoder.h5')
				self.Decoder.save('./Decoder.h5')
				X,Y=samples(batch=1000,gw=10,gwp=20,env=20)
				err,accu=self.Coder.evaluate(X,X)
				print(i,err,accu)
				
	
	def TrainModel(self):
		for i in range(1000000):
			X,Y=samples(batch=100,gw=10,gwp=20,env=20)
			self.Model.train_on_batch(X,Y)
			if j%1000==0:
				self.SaveAllSubModel()
				X,Y=samples(batch=1000,gw=10,gwp=20,env=20)
				err,accu=self.Model.evaluate(X,Y)
				print(i,j,err,accu)

	def SaveAllSubModel(self):
		self.Classifier.save('./Classifier.h5')
		self.Merger.save('./Merger.h5')
		self.Division.save('./Division.h5')
		self.Expecter.save('./Expecter.h5')
		self.Noliner.save('./Noliner.h5')
		

if __name__=='__main__':
	m=TDOA()
	m.LoadModel()
	m.TrainCoder()
