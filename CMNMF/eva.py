import numpy as np
import train
import scipy as sci

def rmse(result,test_data):
	test_loc = test_data.copy()
	test_loc[test_loc>0] = 1
	delta = train.hdot(test_loc,(test_data-result),np.ones((result.shape[0],result.shape[1])))
	delta = train.hdot(delta,delta,np.ones((delta.shape[0],delta.shape[1])))
	rmse = sci.math.sqrt(delta.sum()/test_loc.sum())
	return rmse
	
