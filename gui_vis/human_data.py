
import numpy as np
from scipy.spatial.transform import Rotation as R
from util import load_data_remote, generate_views
from vis_smpl_scene import load_vis_data, get_head_global_rots, vertices_to_head

def make_3rd_view(positions, rots, rotz=0, lookdown=32):
    """
    It takes the positions and rotations of the camera, and returns the positions and rotations of the
    camera, but with the camera rotated by `rotz` degrees around the z-axis, and moved to a new position
    
    Args:
      positions: the position of the camera in the world
      rots: the rotation of the camera
      rotz: rotation around the z axis. Defaults to 0
      lookdown: the angle of the camera, in degrees. Defaults to 32
    """
    # lookdown = R.from_rotvec(np.deg2rad(lookdown) * np.array([0, 0, 1])).as_matrix()
    rotz = R.from_rotvec(np.deg2rad(rotz) * np.array([0, 0, 1])).as_matrix()
    move = rotz @ np.array([0.5, -1, 1.2])

    rots = np.zeros_like(rots)
    for i in range(rots.shape[0]):
        rots[i] =  rotz
    views = generate_views(positions + move, rots, dist=0, rad=np.deg2rad(lookdown))
    return views

class HUMAN_DATA:
    FOV = 'first'
    FREE_VIEW = False

    def __init__(self, is_remote):
        self.is_remote = is_remote
        self.cameras = {}

    def load(self, filename):
        load_data_class = load_data_remote(self.is_remote)
        self.humans = load_data_class.load_pkl(filename)
        self.vis_data_list = load_vis_data(self.humans)
        # self.set_cameras()

    def set_cameras(self, offset_center=-0.2):
        humans_verts = self.humans

        try:
            lidar_position = humans_verts['first_person']['lidar_traj'][:, 1:4]
            head_rots = get_head_global_rots(humans_verts['first_person']['pose'])
            self.cameras['First Lidar View'] = generate_views(lidar_position, head_rots, dist=offset_center)
        except Exception as e:
            print(e)
            print(f'No First Lidar View')

        try:
            verts = self.vis_data_list['humans']['First pose']
            root_position = vertices_to_head(verts, 0)
            root_rots = get_head_global_rots(humans_verts['first_person']['pose'], parents=[0])
            self.cameras['First root View'] = generate_views(root_position, root_rots, rad=np.deg2rad(-10), dist=-0.3)
            self.cameras['3rd View +Y'] = make_3rd_view(root_position, root_rots, rotz=0)
            self.cameras['3rd View -X'] = make_3rd_view(root_position, root_rots, rotz=90)
            self.cameras['3rd View -Y'] = make_3rd_view(root_position, root_rots, rotz=180)
            self.cameras['3rd View +X'] = make_3rd_view(root_position, root_rots, rotz=270)

        except Exception as e:
            print(e)
            print(f'No First root View')

        try:
            try:
                second_verts = self.vis_data_list['humans']['Second opt_pose']
                second_pose = humans_verts['second_person']['opt_pose']
            except:
                try:
                    second_verts = self.vis_data_list['humans']['Second pose']
                    second_pose = humans_verts['second_person']['pose']
                except Exception as e:
                    print(e)
                    print(f'There is no second pose in the data')
            position = vertices_to_head(second_verts) + np.array([0, 0, 0.2])
            rotation = get_head_global_rots(second_pose)
            self.cameras['Second View'] = generate_views(position, rotation, dist=offset_center)

            position = vertices_to_head(second_verts, 0)
            rotation = get_head_global_rots(second_pose, parents=[0])
            self.cameras['(p2) 3rd View +Y'] = make_3rd_view(position, rotation, rotz=0)
            self.cameras['(p2) 3rd View -X'] = make_3rd_view(position, rotation, rotz=90)
            self.cameras['(p2) 3rd View -Y'] = make_3rd_view(position, rotation, rotz=180)
            self.cameras['(p2) 3rd View +X'] = make_3rd_view(position, rotation, rotz=270)
        except Exception as e:
            print(e)

        views = list(self.cameras.keys())
        for view in views:
            print(f'[Camera]: {view}')
        return views

    def get_extrinsic(self, FOV):
        return self.cameras[FOV]
