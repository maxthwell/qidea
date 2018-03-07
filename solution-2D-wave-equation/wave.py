#coding=utf-8
import cv2
import numpy as np

if __name__=='__main__':
	U=np.random.uniform(0,256,[500,500,3])
	V=np.random.uniform(0,256,[500,500,3])
	w=1.
	dt=1e-3

	for i in range(1000):
		Unew = U + V*dt
		Vnew = V + cv2.Laplacian(U,cv2.CV_32F,ksize = 5)*dt/w**2
		U=Unew
		if i%100==0:
			cv2.imwrite('U/%d.jpg'%i,U)
			cv2.imwrite('V/%d.jpg'%i,V)
	
