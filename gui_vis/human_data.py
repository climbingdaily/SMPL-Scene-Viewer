
import numpy as np
from util import load_data_remote, generate_views
from vis_smpl_scene import load_vis_data, get_head_global_rots, vertices_to_head

class HUMAN_DATA:
    FOV = 'first'
    FREE_VIEW = False

    def __init__(self, is_remote):
        self.is_remote = is_remote
        self.cameras = {}

    def load(self, filename):
        load_data_class = load_data_remote(self.is_remote)
        humans = load_data_class.load_pkl(filename)
        self.vis_data_list = load_vis_data(humans)
        self.set_cameras(humans)

    def set_cameras(self, humans):
        try:
            self.cameras['first'] = generate_views(humans['first_person']['lidar_traj']
                                [:, 1:4], get_head_global_rots(humans['first_person']['pose']))
            print(f'[Camera]: First person views')
        except:
            print(f'There is no first pose in the data')


        try:
            second_verts = self.vis_data_list['humans']['Second opt_pose']
            second_pose = humans['second_person']['opt_pose']
            self.cameras['second'] = generate_views(vertices_to_head(second_verts) + np.array([0, 0, 0.2]), get_head_global_rots(second_pose))
            print(f'[Camera]: second opt. person views')
        except:
            try:
                second_verts = self.vis_data_list['humans']['Second pose']
                second_pose = humans['second_person']['pose']
                self.cameras['second'] = generate_views(vertices_to_head(second_verts) + np.array([0, 0, 0.2]), get_head_global_rots(second_pose))
                print(f'[Camera]: second person views')
            except:
                print(f'There is no second pose in the data')
    
    def get_cameras(self, FOV):
        return self.cameras[FOV.lower()]
