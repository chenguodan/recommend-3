import numpy as np
def toone(A):
	for i in range(A.shape[0]):
		A[i] = A[i]/A[i].sum()
	return A

def hdot(A,B,C):
	size = A.shape
	D = np.zeros((size[0],size[1]))
	for i in range(size[0]):
		for j in range(size[1]):
			D[i,j] = A[i,j]*B[i,j]/C[i,j]
	return D
	
	
def updateW1(V1,W1,W2,H,M,b):
	A = W1
	B = np.dot(V1,H.T)+b*np.dot(M,W2)
	C = np.dot(np.dot(W1,H),H.T)+np.empty((W1.shape[0],W1.shape[1]))
	R = hdot(A,B,C)
	R = toone(R)
	return R

def updateW2(V2,W1,W2,H,M,a,b):
	A = W2
	B = a*np.dot(V2,H.T)+b*np.dot(M.T,W1)
	C = a*np.dot(np.dot(W2,H),H.T)+np.empty((W2.shape[0],W2.shape[1]))
	R = hdot(A,B,C)
	R = toone(R)
	return R

def updateH(V1,V2,W1,W2,H,a):
	A = H
	B = np.dot(W1.T,V1)+a*np.dot(W2.T,V2)
	C = np.dot(np.dot(W1.T,W1),H)+a*np.dot(np.dot(W2.T,W2),H)+np.empty((H.shape[0],H.shape[1]))
	R = hdot(A,B,C)
	R = toone(R.T)
	return R.T

def train(V1,V2,M,a,b,K,epoch):
	m1,m2,n = V1.shape[0],V2.shape[0],V1.shape[1]
	W1 = np.random.rand(m1,K)
	W2 = np.random.rand(m2,K)
	H = np.random.rand(K,n)
	for i in range(epoch):
		W1 = updateW1(V1,W1,W2,H,M,b)
		W2 = updateW2(V2,W1,W2,H,M,a,b)
		H = updateH(V1,V2,W1,W2,H,a)
	return H
	
