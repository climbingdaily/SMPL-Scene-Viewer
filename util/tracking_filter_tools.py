################################################################################
# File: \tracking_filter_tools.py                                              #
# Created Date: Sunday July 17th 2022                                          #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
################################################################################

from cmath import e
import numpy as np
import os
import shutil
import argparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from util import o3dvis, list_dir_remote, Data_loader

def select_pcds_by_id(folder, ids):
    pcds = os.listdir(folder)
    os.makedirs(folder + '_select', exist_ok=True)
    for pcd_path in pcds:
        if os.path.isdir(os.path.join(folder, pcd_path)):
            continue
        if pcd_path.endswith('.pcd') and int(pcd_path.split('_')[0]) in ids:
            shutil.copyfile(os.path.join(folder, pcd_path), os.path.join(folder + '_select', pcd_path))
            print(f'\r{pcd_path} saved in {folder}_select', end='', flush=True)

ids = [2,303,216,421,733,832,1037,1207,3437,3116,3218,4753,4922,5222,5725] # 0417-01
ids = [9, 152, 181, 193, 293, 379, 451, 674, 962, 1601, 1709, 2319, 2777, 1395, 2839, 92, 187, 275, 310, 1905, 1960] # 0417-03


def nearest_box(pre_trajs, box_list, diff_frames, framerate = 20, dist_thresh=0.3):
    
    dist_XY = [np.linalg.norm(pre_trajs[-1, :2] - box[:2]) for box in box_list]
    diff_Z = [abs(pre_trajs[-1, 2] - box[2]) for box in box_list]
    min_index = dist_XY.index(min(dist_XY))
    min_vel_XY = min(dist_XY) * framerate / (diff_frames + 0.1)
    min_vel_Z = diff_Z[min_index] * framerate / (diff_frames + 0.1)
    # if min_vel_XY < 5 :
    return min_index, min_vel_XY, min_vel_Z

class filter_tracking_by_interactive():
    def __init__(self, tracking_folder, remote=False, framerate=20):
        self.view_initialized = False
        self.vis = o3dvis(window_name='filter_tracking_by_interactive')
        self.checked_ids = {}
        self.real_person_id = []
        self.save_list = []
        self.pre_geometries = []
        self.pre_human_boxes = {}
        self.trajectory = {}
        self.none_human_boxes = []
        self.real_human_boxes = []
        self.reID = {}
        self.tracking_folder = tracking_folder
        self.remote = remote
        self.load_data = Data_loader(remote)
        self.join = '/' if remote else '\\'
        self._raw_select = self.tracking_folder + '_raw_select'
        self._select = self.tracking_folder + '_select'
        self.load_data.mkdir(self._select)
        self.load_data.mkdir(self._raw_select)
        self.vel_xy = 0
        self.vel_z = 0
        self.framerate = framerate

    def copy_human_pcd(self, save_file):
        self.load_data.mkdir(self._raw_select)
        source = self.join.join([self.tracking_folder, save_file])
        target = self.join.join([self._raw_select, save_file])
        self.load_data.cpfile(source, target)

    def copy_save_files(self, reid=False):
        # 在远程设备上新建文件夹
        # ss = [self.join.join([self.tracking_folder, p]) for p in self.save_list]
        # tt = [self.join.join([self._raw_select, p]) for p in self.save_list]
        # [self.load_data.cpfile(f[0], f[1]) for f in zip(ss, tt)]
        pc = o3d.geometry.PointCloud()

        for pcd_path in self.save_list:
            source = self.join.join([self.tracking_folder, pcd_path])
            if self.load_data.isdir(source):
                continue
            if pcd_path.endswith('.pcd'):
                humanid, appendix = pcd_path.split('_')
                target = self.join.join([self._select, pcd_path])

                # if humanid in self.reID:
                #     new_id = self.reID[humanid]
                #     target = self.join.join([os.path.dirname(target), f'{new_id}_{appendix}'])
                pc += self.load_data.load_point_cloud(source)

                self.load_data.cpfile(source, target)
                print(f'\r{pcd_path} saved in {self._select}', end='', flush=True)

        save_all_pcd_path = self.join.join([os.path.dirname(self.tracking_folder), 'all_human.pcd'])
        xyz = np.asarray(pc.points)
        colors = np.asarray(pc.colors)
        self.load_data.write_pcd(save_all_pcd_path, xyz, rgb = colors*255)
            
    def add_box(self, box, color):
        transform = R.from_rotvec(
            box[6] * np.array([0, 0, 1])).as_matrix()
        center = box[:3]
        extend = box[3:6]
        bbox = o3d.geometry.OrientedBoundingBox(center, transform, extend)
        bbox.color = color
        self.vis.add_geometry(bbox, reset_bounding_box = False, waitKey=0)
        return bbox

    def interactive_choose(self, file_path = None, scene_path=None, pre_box=None, cur_box=None, strs='a real human'):
        """
        It takes in a point cloud, and a bounding box, and asks the user if the bounding box is correct. 
        
        If the user says yes, the bounding box turns green. If the user says no, the bounding box turns
        grey. 
        
        The function returns True if the user says yes, and False if the user says no. 
        
        The function also takes in a second bounding box, which is drawn in cyan. 
        
        The function also takes in a string, which is printed to the screen. 
        
        The function also takes in a point cloud, which is drawn in grey. 
        
        :param file_path: the path to the point cloud file
        :param scene_path: the path to the scene point cloud
        :param pre_box: the bounding box of the previous frame
        :param cur_box: the bounding box of the current point cloud
        :param strs: the string to be displayed on the screen, defaults to a real human (optional)
        :return: The return value is a boolean value.
        """
        if scene_path is not None:
            # pts = o3d.io.read_point_cloud(scene_path)
            pts = self.load_data.load_point_cloud(scene_path)
            if not pts.has_colors():
                pts.paint_uniform_color([0.5,0.5,0.5])
            if self.view_initialized:
                self.vis.add_geometry(pts, reset_bounding_box=False)
            else:
                self.view_initialized = True
                self.vis.add_geometry(pts, reset_bounding_box=True)
                self.vis.set_view()

            self.pre_geometries.append(pts)

        if file_path is not None:
            # pts = o3d.io.read_point_cloud(file_path)
            pts = self.load_data.load_point_cloud(file_path)
            
            self.vis.add_geometry(pts, reset_bounding_box=False)
            if cur_box is not None:
                box = self.add_box(cur_box, (1, 0, 0))
            else:
                box = pts.get_oriented_bounding_box()
                box.color = (1, 0, 0)
                self.vis.add_geometry(box, reset_bounding_box = False, waitKey=0)
            self.pre_geometries.append(pts)
            self.pre_geometries.append(box)
            
        if pre_box is not None:
            box2 = self.add_box(pre_box, (0, 0, 1))
            # self.pre_geometries.append(box)
        else:
            box2 = None
            
        while True:
            print(f'Is this {strs}? Press \'Y\'/\'N\' | Vel_xy: {self.vel_xy:.1f}, vel_z: {self.vel_z:.1f}', end='', flush=True)
            state = self.vis.return_press_state()
            if state and file_path is not None:
                box.color = (0,1,0)
                self.vis.vis.update_geometry(box)
            
            elif file_path is not None:
                box.color = (0.5, 0.5, 0)
                self.vis.vis.update_geometry(box)
            
            if box2 is not None:
                box2.color = (0, 0.5, 0.5)
                self.vis.vis.update_geometry(box2)

            return state

    def get_box(self, frameid, humanid, tracking_results):
        """
        :param frameid: the frame number of the video
        :param humanid: the id of the human you want to track
        :param tracking_results: the output of the tracking algorithm, which is a dictionary with the
        following structure:
        :return: The current box position.
        """
        # cur box postion
        frameid = int(frameid)
        ids = tracking_results[frameid]['ids']
        cur_pos = np.where(ids == int(humanid))[0][0]
        cur_box = tracking_results[frameid]['boxes_lidar'][cur_pos]
        return cur_box

    def is_too_far(self, frameid, humanid, tracking_results, framerate = 20):
        """
        > This function calculates the velocity of a human in the ldiar
        
        :param frameid: the current frame number
        :param humanid: the id of the human
        :param tracking_results: the tracking results from the tracking algorithm
        :param framerate: the frame rate of the ldiar, defaults to 20 (optional)
        """
        # cur box postion
        frameid = int(frameid)
        cur_box = self.get_box(frameid, humanid, tracking_results)

        # pre box postion
        pre_framid = int(self.checked_ids[humanid])
        pre_box = self.get_box(pre_framid, humanid, tracking_results)

        dist = np.linalg.norm(pre_box[:2] - cur_box[:2])
        vel = dist * framerate / (frameid - pre_framid)

        return abs(vel), pre_box, cur_box, pre_framid

    def is_false_box(self, cur_box, dist_thresh=0.3):
        """
        Args:
          cur_box: the current box that we're checking
          dist_thresh: the distance threshold for the xy plane. If the distance between the current box and
        the box in the none_human_boxes list is less than this threshold, then the current box is considered
        a false positive.
        
        Returns:
          a boolean value.
        """
        for box in self.none_human_boxes: 
            dist_xy = np.linalg.norm(cur_box[:2] - box[:2])
            dist_z = abs(cur_box[2] - box[2])
            if dist_xy < dist_thresh and dist_z < 1:
                self.none_human_boxes.append(cur_box)
                return True
        return False

    def is_real_box(self, cur_box, dist_thresh=0.3):
        """
        Args:
          cur_box: the current box we're checking
          dist_thresh: the distance threshold for the xy coordinates of the bounding box.
        
        Returns:
          the list of real human boxes.
        """
        for box in self.real_human_boxes:
            dist_xy = np.linalg.norm(cur_box[:2] - box[:2])
            dist_z = abs(cur_box[2] - box[2])
            if dist_xy < dist_thresh and dist_z < 1:
                self.real_human_boxes.append(cur_box)
                return True
        return False

    def choose_new_id(self, cur_box, cur_humanid, cur_framid, framerate=20):
        """
        If the current human's velocity is less than 5 m/s and the duration between the current frame and
        the previous frame is less than 1 second, then the current human is the same person as the previous
        human
        
        :param cur_box: the current box
        :param cur_humanid: the current human id
        :param cur_framid: the current frame id
        :param framerate: the frame rate of the video, defaults to 20 (optional)
        """
        for humanid, box in self.pre_human_boxes.items():
            dist = np.linalg.norm(cur_box[:3] - box['box'][:3])
            duration = abs(int(cur_framid) - int(box['frameid']))
            vel = dist * framerate / (duration + 0.01)
            if vel < 5 and duration<framerate:
                while humanid in self.reID:
                    humanid = self.reID[humanid]
                self.reID[cur_humanid] = humanid

    def is_real_human(self, frameid, humanid, tracking_folder, tracking_results, scene_path, filtered = False):
        """
        It takes in a frameid, humanid, tracking_folder, tracking_results, scene_path, and filtered. 
        
        It returns is_person. 
        
        It does this by: 
        
        1. Creating a file_path variable. 
        2. Creating a scene_folder variable. 
        3. Creating a scene_path variable. 
        4. Creating an is_person variable. 
        5. Looping through pre_geometries. 
        6. Removing geometry. 
        7. If humanid is in checked_ids: 
        8. Creating a vel variable. 
        9. Creating a pre_box variable. 
        10. Creating a cur_box variable. 
        11. Creating a pre_framid variable. 
        12. If vel is greater than 5: 
        13. Printing a string. 
        14. If filtered is true or interactive_choose returns
        
        :param frameid: the current frame id
        :param humanid: the id of the human
        :param tracking_folder: the folder where the tracking results are stored
        :param tracking_results: a dictionary of tracking results, which is a dictionary of frameid and a
        list of humanid and their bounding boxes
        :param scene_path: the path to the scene image
        :param filtered: whether the data is filtered or not, defaults to False (optional)
        :return: a boolean value.
        """

        file_path = self.join.join([tracking_folder, f'{humanid}_{frameid}.pcd'])

        is_person = False

        for geometry in self.pre_geometries:
            self.vis.remove_geometry(geometry, reset_bounding_box=False)

        if humanid in self.checked_ids:
            
            vel, pre_box, cur_box, pre_framid = self.is_too_far(frameid, humanid, tracking_results, self.framerate)

            if vel > 5:
                print(f'Checking Human:{humanid} Cur frame:{frameid} (red) | Pre frame {pre_framid} (blue)')
                if not self.is_false_box(cur_box):
                    if self.is_real_box(cur_box) or filtered \
                        or self.interactive_choose(file_path=file_path, 
                                                scene_path=scene_path, 
                                                pre_box=pre_box, 
                                                cur_box=cur_box, 
                                                strs='a real human'):
                        is_person = True
                        self.real_human_boxes.append(cur_box)
                        # self.choose_new_id(cur_box, humanid, frameid)
                    else:
                        self.none_human_boxes.append(cur_box)

            elif humanid in self.real_person_id:
                is_person = True

            else:
                pass

        else:
            print(f'Checking Human:{humanid} Frame:{frameid}')
            cur_box = self.get_box(frameid, humanid, tracking_results)
            if not self.is_false_box(cur_box):
                if self.is_real_box(cur_box) or filtered or self.interactive_choose(
                                    file_path=file_path, 
                                    scene_path=scene_path, 
                                    strs='a real human'):
                    self.choose_new_id(cur_box, humanid, frameid, self.framerate)
                    is_person = True
                    self.real_human_boxes.append(cur_box)
                else:
                    self.none_human_boxes.append(cur_box)

        return is_person

    def load_existing_tracking_list(self, tracking_folder):
        """
        It loads the existing tracking list.
        
        :param tracking_folder: the folder where the tracking results are stored
        :return: A dictionary of lists.
        """
        if self.remote:
            pcd_paths = list_dir_remote(self.load_data.client, tracking_folder)
        else:
            pcd_paths = os.listdir(tracking_folder)
        tracking_list = {}
        for pcd_path in pcd_paths:
            if not pcd_path.endswith('.pcd'):
                continue
            humanid = pcd_path.split('_')[0]
            frameid = pcd_path.split('_')[1].split('.')[0]
            if frameid in tracking_list:
                tracking_list[frameid].append(humanid)
            else:
                tracking_list[frameid] = [humanid, ]
        return tracking_list

    def raw_tracking_selected_human(self, frameid, humanid, tracking_results, scene_paths, filtered = 0):  
        """
        It takes a frameid, humanid, tracking_results, scene_paths, and filtered as input, and returns
        is_human and humanid.
        
        Args:
          frameid: the frame number
          humanid: the id of the human in the tracking results
          tracking_results: a list of dictionaries, each dictionary contains the bounding box of a human in
        a frame
          scene_paths: list of paths to the scenes
          filtered: 0 or 1. If 1, then the human is filtered out if it's not a real human. If 0, then the
        human is not filtered out. Defaults to 0
        """
        
        cur_box = self.get_box(frameid, humanid, tracking_results)
        if filtered >=0:
            is_human = self.is_real_human(frameid, humanid, 
                                    self.tracking_folder, 
                                    tracking_results, 
                                    scene_paths[int(frameid)], filtered > 0)
        else:
            is_human = False
            box = self.get_box(frameid, humanid, tracking_results)
            # self.none_human_boxes.append(box)

        if is_human:            
            if humanid not in self.real_person_id:
                self.real_person_id.append(humanid)

            self.checked_ids[humanid] = frameid  
            self.trajectory[frameid] = {'box': cur_box, 'humanid': humanid}
            
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            sphere.translate(cur_box[:3])
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color([0.5,0.5,0.5])

            self.vis.add_geometry(sphere, reset_bounding_box = False, waitKey=0)
            self.save_list.append(f'{humanid}_{frameid}.pcd')
            self.pre_human_boxes[humanid] = {'box': cur_box, 'frameid': frameid}
            # self.copy_human_pcd(f'{humanid}_{frameid}.pcd')

        return is_human, humanid

    def filter_human(self, frameid, humanid, tracking_results, scene_paths, filtered):  
        """
        It checks if the human is real or not, and if it is, it saves the human's ID and the frame ID in a
        list
        
        Args:
          frameid: the current frame id
          humanid: the id of the human
          tracking_results: a list of tracking results, each element is a dict,
          scene_paths: a list of paths to the scenes in the dataset
          filtered: a list of human ids that have been filtered out
        
        Returns:
          is_human, humanid
        """

        is_human = self.is_real_human(frameid, humanid, 
                                    self.tracking_folder, 
                                    tracking_results, 
                                    scene_paths[int(frameid)], filtered)
        if is_human:
            if humanid not in self.real_person_id:
                self.real_person_id.append(humanid)

            self.save_list.append(f'{humanid}_{frameid}.pcd')
            self.copy_human_pcd(f'{humanid}_{frameid}.pcd')

        elif humanid in self.real_person_id:
            self.real_person_id.pop(self.real_person_id.index(humanid))

        self.checked_ids[humanid] = frameid  # save previous frameid for humanid 
        
        cur_box = self.get_box(frameid, humanid, tracking_results)

        self.pre_human_boxes[humanid] = {'box': cur_box, 'frameid': frameid}

        return is_human, humanid

    def run(self, start=0, tracking=False):
        """
        For each frame, find the nearest box to the previous frame, and then filter the human in the
        nearest box
        
        Args:
          start: the frame number to start from. Defaults to 0
          tracking: whether to use tracking. Defaults to False
        """

        tracking_list = self.load_existing_tracking_list(self.tracking_folder)
        basename = os.path.dirname(os.path.dirname(self.tracking_folder))

        tracking_file_path = self.load_data.glob(basename + f'{self.join}*tracking.pkl')[0]

        tracking_results = self.load_data.load_pkl(tracking_file_path)

        tracking_traj_path = self.join.join([os.path.dirname(self.tracking_folder), 'tracking_traj.txt'])
        scene_paths = self.load_data.glob(basename + f'{self.join}lidar_data{self.join}*lidar_frames_rot/*.pcd')
        
        for frameid in sorted(tracking_list.keys()):
            if int(frameid) < start:
                continue
            humanids = tracking_list[frameid]
            if tracking:
                # 找到所有的box
                bboxes = [self.get_box(frameid, h, tracking_results) for h in tracking_list[frameid]]

                pre_frameids = [k for k in self.trajectory.keys()]
                if len(pre_frameids) > 0:

                    trajectory = np.array(
                        [self.trajectory[fid]['box'][:3] for fid in pre_frameids]).astype(np.float32)
                    append = np.array([self.trajectory[fid]['humanid']
                                      for fid in pre_frameids]).astype(np.float32)
                    save_traj = np.concatenate(
                        (trajectory.reshape(-1, 3), append.reshape(-1, 1)), axis=1)
                    # self.load_data.write_txt(tracking_traj_path, save_traj)
                    pre_box = self.trajectory[pre_frameids[-1]]['box']
                    # 挑出离上一个最近的
                    # todo: 取最近的四个点中的非离群点算距离
                    gap = int(frameid) - int(pre_frameids[-1])
                    nn_id, self.vel_xy, self.vel_z = nearest_box(trajectory, bboxes, gap, self.framerate)
                    # is_human = True if humanid in self.real_person_id and vel_xy < 10 else False
                    if self.vel_xy > 9.3 or self.vel_z > 2:
                        filtered = -1 
                    elif self.vel_xy < 3 and self.vel_z < 1 and gap < 10:
                        filtered = 1
                    else:
                        filtered = 0
                    is_human, humanid = self.raw_tracking_selected_human(
                        frameid, humanids[nn_id], tracking_results, scene_paths, filtered = filtered)
                else :
                    for humanid in humanids:
                        is_human, humanid = self.raw_tracking_selected_human(
                            frameid, humanid, tracking_results, scene_paths)
                        if is_human:
                            break
            else:
                for humanid in humanids:
                    self.filter_human(frameid, humanid, tracking_results, scene_paths, filtered)
        if tracking:
            pre_frameids = [k for k in self.trajectory.keys()]

            # center pos
            trajectory = np.array(
                [self.trajectory[fid]['box'][:3] for fid in pre_frameids]).astype(np.float32)

            # humanid
            append = np.array([self.trajectory[fid]['humanid']
                                for fid in pre_frameids]).astype(np.float32)

            # frameid
            save_traj = np.concatenate(
                (trajectory.reshape(-1, 3), append.reshape(-1, 1), np.array(pre_frameids).reshape(-1, 1)), axis=1)
            self.load_data.write_txt(tracking_traj_path, save_traj)
        self.copy_save_files()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking", '-T', action='store_true')
    parser.add_argument("--start_frame", '-S', type=int, default=-1)
    parser.add_argument("--framerate", type=int, default=0)
    parser.add_argument("--tracking_file", type=str, default=None)
    parser.add_argument('--tracking_folder', '-f', type=str,
                        default=None, help='A directory')
    args, opts = parser.parse_known_args()

    import config
    start_frame = config.start_frame if args.start_frame == -1 else args.start_frame
    is_remote = True if '--remote' in opts else config.remote
    tracking_filter = True if '--filter' in opts else config.tracking_filter
    tracking_folder = config.tracking_folder if args.tracking_folder is None else args.tracking_folder
    framerate = config.framerate if args.framerate == 0 else args.framerate

    # select_pcds_by_id(args.folder, ids)
    filter = filter_tracking_by_interactive(tracking_folder, remote=is_remote, framerate=framerate)
    filter.run(start=start_frame, tracking=tracking_filter)
