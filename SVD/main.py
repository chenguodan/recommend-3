import pre
import svd

data = pre.prepare(0.2)
para = {'gama':0.007,'lambda4':0.02,'lambda5':0.005,'lambda6':0.015,'K':2,'epoch':10}

a = svd.SVD(para)
a.train(data)
result,rmse = a.test(data)

print rmse
