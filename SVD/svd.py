import numpy as np
import math
#changed
class SVD(object):
	def __init__(self,para):
		self.para = para
		model = dict(p=[],q=[],bu=[],bi=[],mu=[],y=[])
		self.model = model


	def train(self,data):
		train = data['train']
		ulen,vlen = len(data['user']),len(data['video'])
		self.model['p'] = np.random.rand(ulen,self.para['K'])
		self.model['q'] = np.random.rand(vlen,self.para['K'])
		self.model['bu'] = np.random.rand(ulen)
		self.model['bi'] = np.random.rand(vlen)
		self.model['mu'] = np.average(train[:,2])

		d = {}
		for key,value in zip(train[:,0],train[:,1]):
			if key not in d:
				d[key] = [np.array([]),np.array([])]
			d[key][0] = np.append(d[key][0],value) 
		for key in d:
			d[key][1] = np.random.rand(len(d[key][0]),self.para['K'])

		self.model['y'] = d
		for i in xrange(self.para['epoch']):
			err = np.array([])
			for row in train:
				m,n,r = int(row[0]),int(row[1]),row[2]
				#self.model['p'][m],self.model['q'][n],self.model['bu'][m],self.model['bi'][n],e = \
				#svditer(self.model['p'][m],self.model['q'][n],self.model['bu'][m],\
				#self.model['bi'][n],self.model['mu'],r,self.para['gama'],self.para['lambda4'])
				self.model['p'][m],self.model['q'][n],self.model['bu'][m],self.model['bi'][n],self.model['y'][m],e = \
				svdppiter(self.model['p'][m],self.model['q'][n],self.model['bu'][m],\
				self.model['bi'][n],self.model['mu'],self.model['y'][m],r,n,self.para['gama'],self.para['lambda5'],self.para['lambda6'])
				err = np.append(err,e)
			print 'epoch:',i,'/',self.para['epoch'],'cost:',np.dot(err,err)
			
		print 'finished'

	def test(self,data):
		test = data['test']
		test_result = np.array([])
		for row in test:
			m,n = int(row[0]),int(row[1])
			result = self.model['mu'] + self.model['bu'][m] + \
			self.model['bi'][n] + np.dot(self.model['p'][m],self.model['q'][n])
			test_result = np.append(test_result,result)
		error = rmse(test[:,2],test_result)
		return test_result,error

def svditer(p,q,bu,bi,mu,r,gama,lambda4):
	e = r-(mu + bu + bi + np.dot(p,q))
	bu = bu + gama*(e - lambda4*bu)
	bi = bi + gama*(e - lambda4*bi)
	q = q + gama*(e*p - lambda4*q)
	p = p + gama*(e*q - lambda4*p)
	return p,q,bu,bi,e

def svdppiter(p,q,bu,bi,mu,y,r,n,gama,lambda5,lambda6):
	ru =  1/math.sqrt(len(y[0]))
	ruy = ru*y[1].sum(axis=0)
	e = r-(mu + bu + bi + np.dot((p+ruy),q))
	bu = bu + gama*(e - lambda5*bu)
	bi = bi + gama*(e - lambda5*bi)
	q = q + gama*(e*(p+ruy) - lambda6*q)
	p = p + gama*(e*q - lambda6*p)
	for i in xrange(y[1].shape[0]):
		y[1][i] =  y[1][i] + gama*(e*ru*q - lambda6*y[1][i])
	return p,q,bu,bi,y,e
		
def rmse(r1,r2):
	return math.sqrt(np.dot((r1-r2),(r1-r2))/len(r1))
		
		
