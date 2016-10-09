"""
'recommen2'
"""
import numpy as np
import math

class Recom2:
	#parameters: u_v_ranges, dim, user_field
	def __init__(self, sizes, uf):
		self.sizes = sizes
		self.mu = 0.0
		self.bu = np.random.randn(sizes[0])
		self.bv = np.random.randn(sizes[1])
		self.p = np.random.randn(sizes[0], sizes[2])
		self.q = np.random.randn(sizes[1], sizes[2])
		self.uf = uf
		self.y = dict()
		for (k, v) in uf.items():
			self.y[k] = np.random.randn(len(v), sizes[2])
	
	def simu_rate(self, u, v):
		pre = self.mu+self.bu[u]+self.bv[v]+ \
			np.dot(self.p[u], self.q[v])
		return pre+np.dot(self.q[v], self.simu_y(u))
	
	def eval(self, ids, rates, dlen):
		sr = np.zeros(dlen)
		for (u, v), i in zip(ids, range(dlen)):
			sr[i] = self.simu_rate(u, v)
		print "test-rmse: {0}".format(self.cost(rates-sr))
		return sr

	def simu_y(self, u):
		if self.uf.has_key(u):
			return self.y[u].mean(axis = 0)* \
				math.sqrt(len(self.uf[u]))
		else:
			return np.zeros(self.sizes[2])

	def update_yu(self, u, v, e, gama, lmbda1):
		if not self.uf.has_key(u): return
		#by = np.copy(self.y[u])
		ylen = len(self.uf[u])
		for i, v in zip(range(ylen), self.uf[u]):
			self.y[u][i] = self.y[u][i]+gama* \
			(e*self.q[v]/math.sqrt(ylen)-lmbda1*self.y[u][i])
	
	#parameters: epoch, gama, lmbda, lmbda1
	def svd_plus(self, ids, rates, dlen, paras):
		[epoch, gama, lmbda, lmbda1] = paras
		for i in xrange(epoch):
			err = self.update(ids, dlen, rates, \
				gama, lmbda, lmbda1)
			print "cost: {0} in {1}/{2}".format( \
				self.cost(err), i+1, epoch)
		print "train complete"

	def update(self, ids, dlen, rates, gama, lmbda, lmbda1):
		err = []
		for (u, v), r in zip(ids, rates):
			e = r-self.simu_rate(u, v)
			self.bu[u] = self.bu[u]+gama*(e-lmbda*self.bu[u])
			self.bv[v] = self.bv[v]+gama*(e-lmbda*self.bv[v])
			self.q[v] = self.q[v]+gama*(e* \
						(self.p[u]+self.simu_y(u))-lmbda1*self.q[v])
			self.p[u] = self.p[u]+gama*(e*self.q[v]-lmbda1*self.p[u])
			self.update_yu(u, v, e, gama, lmbda1)
			err.append(e)
		return np.asarray(err)
			

	def cost(self, err):
		return math.sqrt(np.dot(err, err)/len(err))

