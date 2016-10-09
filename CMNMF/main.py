import numpy as np
import os
import data_prepare
import train
import test
import eva


if os.path.exists('data.npz') == True:
	print 'data_prepare compeleted'
else:
	data_prepare.data_prepare()
	print 'data_prepare compeleted'
        
data = np.load('data.npz')
data_user_video = data['arr_0']
data_video_album = data['arr_1']
data_user_album = np.random.rand(data_user_video.shape[0],data_video_album.shape[1])

a,b,k,ecoph = 1,1,100,50

H = train.train(data_user_album.T,data_user_video.T,data_video_album.T,a,b,k,ecoph)
result = test.test(H,data_user_album)
rmse = eva.rmse(result,data_user_album)
print rmse


