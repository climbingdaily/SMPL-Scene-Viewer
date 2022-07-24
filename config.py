# view point type
# 'First': first person view point
# 'Second': second person view point
# 'Third': third person view point, free to control view point
viewpoint_type = 'First' 

# if the file is on remote machine
remote = True
username = 'dyd'
hostname = 'localhost'
port = 911


# vis_double.py
start = 0
end = -1 # all frames
scene_path = '/hdd/dyd/lidarhumanscene/data/0417003/0417003.pcd'
smpl_file_path = '/hdd/dyd/lidarhumanscene/data/0417003/synced_data/two_person_param.pkl'
pred_file_path = None

# vis_pred_smpl.py
# file_path = '/hdd/dyd/lidarhumanscene/data/0604_haiyun/synced_data/second_person/segments.pkl'
file_path = 'C:\\Users\\DAI\\Desktop\\temp\\chenchen001_label.h5py'

# tracking_filter_tools.py
start_frame = 0
end_frame = -1 # all frames
tracking_folder = '/hdd/dyd/lidarhumanscene/data/0417003/segment_by_tracking'
tracking_filter = True