import numpy as np
import train

def test(H,train_data):
	simi = np.dot(H.T,H)
	simi = train.toone(simi)
	result = np.dot(simi,train_data)
	return result
	
	
