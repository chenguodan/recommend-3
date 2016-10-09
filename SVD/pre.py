import numpy as np

def index(data,d1,d2):
	data[:,0] = [d1[x] for x in data[:,0]]
	data[:,1] = [d2[x] for x in data[:,1]]
	return data

def prepare(test_set):
	data = np.loadtxt('data.txt')
	user_line = data[:,0].astype(int)
	video_line = data[:,1].astype(int)
	user_id = np.unique(user_line)	
	video_id = np.unique(video_line)
	d_user = dict(zip(user_id,np.arange(len(user_id),dtype=int)))
	d_video = dict(zip(video_id,np.arange(len(video_id),dtype=int)))

	d = {}
	for key,value in zip(user_line,video_line):
		if key not in d:
			d[key]  = np.array([])
		d[key] = np.append(d[key],value)

	pair = []
	for key,value in d.iteritems():
		test_set_num = int(len(d[key])*test_set)
		mask = key*np.ones(test_set_num)
		pair += zip(mask,value[0:test_set_num])

	data_train = data.copy()
	data_test = data.copy()

	for user,video in pair:
		user_location = np.where(user_line == user)
		video_location = np.where(video_line == video)
		location = np.intersect1d(user_location,video_location)
		data_train[location,2] = 0  	

	data_test[:,2] = data[:,2] - data_train[:,2]
	data_train = index(data_train,d_user,d_video)
	data_test = index(data_test,d_user,d_video)
	data_train = np.delete(data_train,np.where(data_train[:,2] == 0),0)
	data_test = np.delete(data_test,np.where(data_test[:,2] == 0),0)
	data = dict(train = data_train,test = data_test,user = user_id,video = video_id)
	return data
		
