import numpy as np


def data_prepare():
	data_user_video = np.loadtxt('data.txt')
	data_video_album = np.loadtxt('vid_albumid_mapping.txt')

	user_id = np.unique(data_user_video[:,0])
	video_id = np.unique(data_user_video[:,1])

	video_id_album = np.unique(data_video_album[:,0])
	video_id = np.array(list(set(video_id).union(video_id_album)))
	album_id = np.unique(data_video_album[:,1])
	album_id = album_id[2:]
	user_id.sort()
	video_id.sort()
	album_id.sort()

	user_video = np.zeros((len(user_id),len(video_id)))
        video_album = np.zeros((len(video_id),len(album_id)))
        
 
	for row in data_user_video:
		user_video[np.where(user_id == row[0]),np.where(video_id == row[1])] = row[2]
        for row in data_video_album:
		video_album[np.where(video_id == row[0]),np.where(album_id == row[1])] = 1

	np.savez('data.npz',user_video,video_album,user_id,video_id,album_id)







