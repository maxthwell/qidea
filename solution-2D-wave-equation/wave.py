#coding=utf-8
import cv2
import numpy as np
from math import sin,cos

if __name__=='__main__':
	U=np.zeros([200,200,3])
	V=np.zeros([200,200,3])

	r=0.1
	dt=0.01
	
	for i in range(1000):
		A = cv2.Laplacian(U,-1,ksize=5)-r*V
		A[103:105,103:105,int(i/400)%3]+=128*sin(i/10)
		U += V*dt
		V += A*dt
		if i%100==0:
			print(i)
			cv2.imwrite('U/%d.jpg'%i,U)
			cv2.imwrite('V/%d.jpg'%i,V)
	
