#coding=utf-8
import cv2
import numpy as np

def mean_kernel():
	mean_kernel=np.zeros((3,3),np.float32)
	mean_kernel[0,0]=0
	mean_kernel[0,1]=1
	mean_kernel[0,2]=0
	mean_kernel[1,0]=1
	mean_kernel[1,1]=0.
	mean_kernel[1,2]=1.
	mean_kernel[2,0]=0
	mean_kernel[2,1]=1
	mean_kernel[2,2]=0
	return mean_kernel/4.0

def laplace_kernel():
	meanL=mean_kernel()
	meanL[1,1]=-1
	return meanL

def OpLaplace(img):
	return cv2.filter2D(img,-1,kernel)

def OpIterate(imgU,epoch=100):
	for i in range(epoch):
		imgU=cv2.filter2D(imgU,-1,meanK)-imgL
	return imgU

if __name__=="__main__":
	src=cv2.imread('src.jpg')
	src=np.float32(src)
	print(src.shape)
	meanK=mean_kernel()
	laplaceK=laplace_kernel()
	imgL=cv2.filter2D(src,-1,laplaceK)
	imgGauss=cv2.GaussianBlur(src,(201,201),0)
	imgNoise=np.random.uniform(0,256,src.shape)
	cv2.imwrite('noise.jpg',imgNoise)
	cv2.imwrite('laplace.jpg',imgL)
	cv2.imwrite('gauss.jpg',imgGauss)
	

	for i in range(10):
		imgNoise=OpIterate(imgNoise)
		print('generate image noise_%d.jpg'%i)
		cv2.imwrite('img/noise_%d.jpg'%i,imgNoise)

	for i in range(10):
		imgGauss=OpIterate(imgGauss)
		print('generate image gauss_%d.jpg'%i)
		cv2.imwrite('img/gauss_%d.jpg'%i,imgGauss)
