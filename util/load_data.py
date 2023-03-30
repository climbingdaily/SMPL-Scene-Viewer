################################################################################
# File: \load_data.py                                                          #
# Created Date: Monday July 18th 2022                                          #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
################################################################################

from logging import raiseExceptions
import numpy as np
import pickle as pkl
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import shutil
import paramiko
# import config
import cv2

from util import pypcd

view = {
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 68.419929504394531, 39.271018981933594, 11.569537162780762 ],
			"boundingbox_min" : [ -11.513210296630859, -35.915927886962891, -2.4593989849090576 ],
			"field_of_view" : 60.0,
			"front" : [ 0.28886465410454343, -0.85891896928352873, 0.42286571841896009 ],
			"lookat" : [ 0.76326815774101275, 3.2896492351216851, 0.040108816664781548 ],
			"up" : [ -0.12866047345544837, 0.40286011796513765, 0.90617338734004726 ],
			"zoom" : 0.039999999999999994
		}
	],
}

def client_server(username=None, hostname=None, port=None, password=None):
    """
    It creates a client object that connects to the server using the username, hostname, and port number
    that you provide
    
    Args:
      username: The username to log in with.
      hostname: The hostname of the server you want to connect to.
      port: The port number to connect to the SSH server on.
    
    Returns:
      A client object.
    """
    if not username or not hostname or not port:
        # username = config.username, hostname = config.hostname, port = config.port
        # raiseExceptions()
        
        return

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, password = password, compress=True)
    return client

def list_dir_remote(client, folder):
    """
    It takes a client object and a folder name, and returns a list of the files in that folder
    
    Args:
      client: the ssh client
      folder: the folder you want to list
    
    Returns:
      A list of files in the folder
    """
    stdin, stdout, stderr = client.exec_command('ls ' + folder)
    res_list = stdout.readlines()
    return [i.strip() for i in res_list]

def read_pcd_from_server(client, filepath, sftp_client = None):
    """
    It reads a pcd file from a remote server and returns a numpy array
    
    Args:
      client: the ssh client
      filepath: the path to the file on the server
      sftp_client: the sftp client that you use to connect to the server.
    
    Returns:
      a numpy array of the point cloud data.
    """
    if sftp_client is None:
        sftp_client = client.open_sftp()
    remote_file = sftp_client.open(filepath, mode='rb')  # 文件路径

    try:
        pc_pcd = pypcd.PointCloud.from_fileobj(remote_file)
        return read_pcd(pc_pcd)
    except Exception as e:
        print(f"Load point cloud {filepath} error")
    finally:
        remote_file.close()

def read_pcd(pc_pcd):
    # pc_pcd = pypcd.point_cloud_from_path(pcd_file)
    pc = np.zeros((pc_pcd.pc_data.shape[0], 3))
    pc[:, 0] = pc_pcd.pc_data['x']
    pc[:, 1] = pc_pcd.pc_data['y']
    pc[:, 2] = pc_pcd.pc_data['z']
    fields = {}
    count = 3
    if 'rgb' in pc_pcd.fields:
        append = pypcd.decode_rgb_from_pcl(pc_pcd.pc_data['rgb'])/255
        pc = np.concatenate((pc, append), axis=1)
        fields['rgb'] = [count, count+1, count+2]
        count += 3
    if 'normal_x' in pc_pcd.fields and 'normal_y' in pc_pcd.fields and 'normal_z' in pc_pcd.fields:        
        append = pc_pcd.pc_data['normal_x'].reshape(-1, 1)
        pc = np.concatenate((pc, append), axis=1)
        append = pc_pcd.pc_data['normal_y'].reshape(-1, 1)
        pc = np.concatenate((pc, append), axis=1)
        append = pc_pcd.pc_data['normal_z'].reshape(-1, 1)
        pc = np.concatenate((pc, append), axis=1)
        fields['normal'] = [count, count+1, count+2]
        count += 3
    if 'intensity' in pc_pcd.fields:        
        append = pc_pcd.pc_data['intensity'].reshape(-1, 1)
        pc = np.concatenate((pc, append), axis=1)
        fields['intensity'] = [count]
        count += 1
    
    return pc, fields
      
def load_scene(vis, pcd_path=None, scene = None, data_loader=None):
    """
    It loads a point cloud from a file and displays it in the viewer
    
    Args:
      vis: the visualization object
      pcd_path: the path to the point cloud file
      scene: the scene to be rendered.
      data_loader: the class that loads the data.
    
    Returns:
      The scene is being returned.
    """
    from time import time
    
    if data_loader is None:
        data_loader = Data_loader(remote=False)

    if scene is None and pcd_path is not None:
        t1 = time()
        print(f'Loading scene from {pcd_path}')
        scene = data_loader.load_point_cloud(pcd_path)
        t2 = time()
        print(f'====> Scene loading comsumed {t2-t1:.1f} s.')
    else:
        print('No scene data!!!')
        return None

    try:
        vis.set_view(view)
        vis.add_geometry(scene)
    except Exception as e:
        pass
        # print(e)
        
    return scene


class Data_loader(object):
    client = None
    sftp_client = None
    def __init__(self, remote, username=None, hostname=None, port=None, password=None):
        self.remote = remote
        if remote:
            Data_loader.make_client_server(username, hostname, port, password)

    @staticmethod
    def make_client_server(username=None, hostname=None, port=None, password=None):
        Data_loader.client = client_server(username, hostname, port, password)
        Data_loader.sftp_client = Data_loader.client.open_sftp()

    def isdir(self, path):
        if self.remote:
            _, stdout, _ = self.client.exec_command(f'[ -d {path} ] && echo OK') # 远程判断文件是否存在
            if stdout.read().strip() == b'OK':
                return True
            else:
                return False
        else:
            return os.path.isdir(path)

    def mkdir(self, path):
        """
        If the remote directory doesn't exist, create it
        
        Args:
          path: The path to the file or directory to be uploaded or downloaded.
        """
        if self.remote:
            _, stdout, _ = self.client.exec_command(f'[ -d {path} ] && echo OK') # 远程判断文件是否存在
            if stdout.read().strip() != b'OK':
                self.client.exec_command(f'mkdir {path}')
        else:
            os.makedirs(path, exist_ok=True)

    def cpfile(self, source, target):
        """
        If the remote flag is set, then copy the file using the remote client. Otherwise, use the local
        copyfile function
        
        Args:
          source: The source file path
          target: The target host to connect to.
        """
        if self.remote:
            _, stdout, _ = self.client.exec_command(f'cp {source} {target} && echo OK') # 远程判断文件是否存在
            if stdout.read().strip() != b'OK':
                print(f'Copy file {source} to {target} error')
        else:
            shutil.copyfile(source, target)
            
    def exec_command(self, command):
        """
        It takes a command as a string, and returns the output of that command as a string
        
        Args:
          command: The command to execute on the remote host.
        
        Returns:
          The return value is a tuple of three items:
                stdin, stdout, stderr
        """
        return self.client.exec_command(command)
    
    def glob(self, str):
        """
        It takes a string as input, and returns a list of files that match the string
        
        Args:
          str: the string to be searched for
        
        Returns:
          A list of files that match the glob pattern.
        """
        from glob import glob
        if self.remote:
            stdin, stdout, stderr = self.client.exec_command('ls ' + str)
            res_list = stdout.readlines()
            files = [i.strip() for i in res_list]
        else:
            files = glob(str)
        return files

    def list_dir(self, folder):
        """
        If the remote flag is set, then execute the command 'ls' on the remote server, and return the
        result. Otherwise, return the result of the local 'ls' command
        
        Args:
          folder: the folder to list
        
        Returns:
          A list of files in the folder
        """
        if self.remote:
            stdin, stdout, stderr = self.client.exec_command('ls ' + folder)
            res_list = stdout.readlines()
            dirs = [i.strip() for i in res_list]
        else:
            dirs = os.listdir(folder)
        return dirs

    def load_point_cloud(self, file_name, pointcloud = None, position = None):
        """
        > Load point cloud from local or remote server
        
        Args:
          file_name: the name of the file to be loaded
          pointcloud: The point cloud to be visualized.
          position: the position of the point cloud
        
        Returns:
          A pointcloud object.
        """
        if pointcloud is None:
            pointcloud = o3d.geometry.PointCloud()
            
        if file_name.endswith('.txt'):
            pts = np.loadtxt(file_name)
            xyz = [1,2,3] if pts.shape[1] == 9 else [0,1,2]
            pointcloud.points = o3d.utility.Vector3dVector(pts[:, xyz]) 
        elif file_name.endswith('.pcd'):
            if self.remote:
                pcd, fields = read_pcd_from_server(self.client, file_name, self.sftp_client)
            else:
                pcd, fields = read_pcd(pypcd.point_cloud_from_path(file_name)) 

            pointcloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
            if 'normal' in fields:
                pointcloud.normals = o3d.utility.Vector3dVector(pcd[:, fields['normal']])

            if 'rgb' in fields:
                pointcloud.colors = o3d.utility.Vector3dVector(pcd[:, fields['rgb']])

            elif 'intensity' in fields:
                if pcd[:, fields['intensity']].max() > 255:
                    intensity = np.array([155 * np.log2(i/100) / np.log2(864) + 100 if i > 100 else i for i in pcd[:, fields['intensity']]]).squeeze()
                else:
                    intensity = pcd[:, fields['intensity']].squeeze()
                scale = 1 if intensity.max() < 1.1 else 255
                colors = plt.get_cmap('gray')(intensity/scale)[:, :3]
                pointcloud.colors = o3d.utility.Vector3dVector(colors)

            if position is not None:
                
                points = np.asarray(pointcloud.points)
                rule1 = abs(points[:, 0] - position[0]) < 40
                rule2 = abs(points[:, 1] - position[1]) < 40
                rule3 = abs(points[:, 2] - position[2]) < 5
                rule = [a and b and c for a,b,c in zip(rule1, rule2, rule3)]

                pointcloud = pointcloud.select_by_index(np.arange(len(rule1))[rule]) 
        elif  file_name.endswith('.ply'):
            pointcloud = o3d.io.read_triangle_mesh(file_name)
            # pointcloud.compute_vertex_normals()        
        elif  file_name.endswith('.obj'):
            pointcloud = o3d.io.read_triangle_mesh(file_name, True)
            # pointcloud.textures[0] = [o3d.io.read_image(file_name.replace('.obj', '.jpg'))]
            # pointcloud.compute_vertex_normals()
            # pointcloud = o3d.t.geometry.TriangleMesh.from_legacy(pointcloud)
        elif file_name.endswith('.bin'):
            pc = np.fromfile('1329073157315402.bin', dtype=np.float32).reshape(-1, 4)
            pointcloud.points = o3d.utility.Vector3dVector(pc[:, :3])
        else:
            pass
        return pointcloud

    def load_imgs(self, filepath):
        if self.remote:
            with self.sftp_client.open(filepath) as obj:
                img_code = np.frombuffer(obj.read(), dtype=np.uint8)
                img = cv2.imdecode(img_code, cv2.IMREAD_COLOR)[..., [2,1,0]]
                img = o3d.geometry.Image(np.asarray(img, order="C"))
        else:
            img = o3d.io.read_image(filepath)

        return img

    def write_image_to_server(self, image_path, remote_path):
        if self.remote:
            with self.sftp_client.open(remote_path, "wb") as remote_file:
                # Read image data from file
                img = cv2.imread(image_path)
                # Encode image data and upload via SFTP
                _, im_buf_arr = cv2.imencode(".jpg", img)
                im_byte_arr = im_buf_arr.tobytes()
                remote_file.write(im_byte_arr)

    def load_pkl(self, filepath):
        """
        If the remote flag is set to True, then the function will open the filepath using the sftp_client
        object, and load the pickle file. 
        
        If the remote flag is set to False, then the function will open the filepath using the open()
        function, and load the pickle file. 
        
        The function returns the loaded pickle file.
        
        Args:
          filepath: the path to the file you want to load
        
        Returns:
          A list of dictionaries.
        """
        if self.remote:
            with self.sftp_client.open(filepath, mode='rb') as f:
                dets = pkl.load(f)

        else:
            with open(filepath, 'rb') as f:
                dets = pkl.load(f)

        return dets

    def write_pcd(self, filepath, data, rgb=None, intensity=None, mode='w') -> None:
        """
        It takes a point cloud, and writes it to a file on the remote server
        
        Args:
          filepath: The path to the file on the remote server.
          data: numpy array of shape (N, 3)
          rgb: a numpy array of shape (N, 3) value from 0 to 255
          intensity: the intensity of the point cloud, (N, 1) 
          mode: 'w' for write, 'a' for append. Defaults to w
        """

        if rgb is not None and intensity is not None:
            rgb = pypcd.encode_rgb_for_pcl(rgb.astype(np.uint8))
            dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),('rgb', np.float32), ('intensity', np.float32)])
            pc = np.rec.fromarrays([data[:,0], data[:,1], data[:,2], rgb, intensity], dtype=dt)

        elif rgb is not None:
            rgb = pypcd.encode_rgb_for_pcl(rgb.astype(np.uint8))
            dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),('rgb', np.float32)])
            pc = np.rec.fromarrays([data[:,0], data[:,1], data[:,2], rgb], dtype=dt)

        elif intensity is not None:
            dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),('intensity', np.float32)])
            pc = np.rec.fromarrays([data[:,0], data[:,1], data[:,2], intensity], dtype=dt)

        else:
            dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
            pc = np.rec.fromarrays([data[:,0], data[:,1], data[:,2]], dtype=dt)

        pc = pypcd.PointCloud.from_array(pc)
        if self.remote:
            f = self.sftp_client.open(filepath, mode = mode)
        else:
            f = open(filepath, mode = mode)  
        pc.save_pcd_to_fileobj(f, compression='binary')

    def write_txt(self, filepath, data, mode='w') -> None:
        """
        > This function writes a list of lists to a text file on the remote server
        
        Args:
          filepath: The path to the file on the remote server.
          data: a list of lists, where each list is a row of data
          mode: 'w' for write, 'a' for append, 'r' for read, 'rb' for read binary, 'wb' for write binary.
        Defaults to w
        """
        save_data = []
        for line in data:
            ll = ''
            for l in line:
                ll += f"{l:.4f}\t"
            save_data.append(ll + '\n')
        if self.sftp_client is not None:
            with self.sftp_client.open(filepath, mode = mode) as f:
                f.writelines(save_data)
        else:
            with open(filepath, mode = mode) as f:
                f.writelines(save_data)

    def read_poses(self, data_root_path):
        """
        It reads the poses.txt file and returns the poses44 array.
        
        Args:
          data_root_path: the path to the root directory of the dataset
        
        Returns:
          poses44
        """
        if self.remote:
            with self.sftp_client.open(data_root_path + '/poses.txt', mode='r') as f:
                poses = f.readlines()

            ps = []
            for p in poses:
                ps.append(p.strip().split(' '))
            poses34 = np.asarray(ps).astype(np.float32).reshape(-1, 3, 4)
        else:
            poses34 = np.loadtxt(os.path.join(data_root_path, 'poses.txt')).reshape(-1, 3, 4)
        
        poses44 = np.array([np.concatenate((p, np.array([[0,0,0,1]]))) for p in poses34])
        return poses44
