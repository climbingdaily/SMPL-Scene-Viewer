import numpy as np
import pickle as pkl
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from util.o3dvis import o3dvis
import matplotlib.pyplot as plt
import shutil
import pypcd
import paramiko

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

def client_server(username = 'dyd', hostname = "10.24.80.241", port = 911):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, compress=True)
    return client

def list_dir_remote(client, folder):
    stdin, stdout, stderr = client.exec_command('ls ' + folder)
    res_list = stdout.readlines()
    return [i.strip() for i in res_list]

def read_pcd_from_server(client, filepath, sftp_client = None):
    if sftp_client is None:
        sftp_client = client.open_sftp()
    remote_file = sftp_client.open(filepath, mode='rb')  # 文件路径

    try:
        pc_pcd = pypcd.PointCloud.from_fileobj(remote_file)
        pc = np.zeros((pc_pcd.pc_data.shape[0], 3))
        pc[:, 0] = pc_pcd.pc_data['x']
        pc[:, 1] = pc_pcd.pc_data['y']
        pc[:, 2] = pc_pcd.pc_data['z']
        if 'rgb' in pc_pcd.fields:
            append = pypcd.decode_rgb_from_pcl(pc_pcd.pc_data['rgb'])/255
            pc = np.concatenate((pc, append), axis=1)
        if 'normal_x' in pc_pcd.fields:        
            append = pc_pcd.pc_data['normal_x'].reshape(-1, 1)
            pc = np.concatenate((pc, append), axis=1)
        if 'normal_y' in pc_pcd.fields:        
            append = pc_pcd.pc_data['normal_y'].reshape(-1, 1)
            pc = np.concatenate((pc, append), axis=1)
        if 'normal_z' in pc_pcd.fields:        
            append = pc_pcd.pc_data['normal_z'].reshape(-1, 1)
            pc = np.concatenate((pc, append), axis=1)
        if 'intensity' in pc_pcd.fields:        
            append = pc_pcd.pc_data['intensity'].reshape(-1, 1)
            pc = np.concatenate((pc, append), axis=1)
        
        return np.concatenate((pc, append), axis=1)
    except Exception as e:
        print(f"Load point cloud {filepath} error")
    finally:
        remote_file.close()

      
def load_scene(vis, pcd_path=None, scene = None):
    from time import time
    reading_class = load_data_remote(remote=True)
    if pcd_path is not None:
        t1 = time()
        print(f'Loading scene from {pcd_path}')
        scene = reading_class.load_point_cloud(pcd_path)
        t2 = time()
        print(f'====> Scene loading comsumed {t2-t1:.1f} s.')
    vis.set_view(view)
    vis.add_geometry(scene)
    return scene


class load_data_remote(object):
    client = None
    sftp_client = None
    def __init__(self, remote):
        self.remote = remote
        if remote:
            load_data_remote.make_client_server()

    @staticmethod
    def make_client_server():
        load_data_remote.client = client_server()
        load_data_remote.sftp_client = load_data_remote.client.open_sftp()

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
        if self.remote:
            _, stdout, _ = self.client.exec_command(f'[ -d {path} ] && echo OK') # 远程判断文件是否存在
            if stdout.read().strip() != b'OK':
                self.client.exec_command(f'mkdir {path}')
        else:
            os.makedirs(path, exist_ok=True)

    def cpfile(self, source, target):
        if self.remote:
            _, stdout, _ = self.client.exec_command(f'cp {source} {target} && echo OK') # 远程判断文件是否存在
            if stdout.read().strip() != b'OK':
                print(f'Copy file {source} to {target} error')
        else:
            shutil.copyfile(source, target)
            
    def exec_command(self, command):
        return self.client.exec_command(command)
    
    def list_dir(self, folder):
        if self.remote:
            stdin, stdout, stderr = self.client.exec_command('ls ' + folder)
            res_list = stdout.readlines()
            dirs = [i.strip() for i in res_list]
        else:
            dirs = os.listdir(folder)
        return dirs

    def load_point_cloud(self, file_name, pointcloud = None, position = [0, 0, 0]):
        if pointcloud is None:
            pointcloud = o3d.geometry.PointCloud()
            
        if self.remote:
            # client = client_server()
            # files = sorted(list_dir_remote(client, file_path))
            _, stdout, _ = self.client.exec_command(f'[ -f {file_name} ] && echo OK') # 远程判断文件是否存在
            if stdout.read().strip() != b'OK':
                print(f'Load {file_name} error')
                return pointcloud
        elif not os.path.exists(file_name):
            return pointcloud

        if file_name.endswith('.txt'):
            pts = np.loadtxt(file_name)
            pointcloud.points = o3d.utility.Vector3dVector(pts[:, :3]) 
        elif file_name.endswith('.pcd') or file_name.endswith('.ply'):
            if self.remote:
                pcd = read_pcd_from_server(self.client, file_name, self.sftp_client)
                pointcloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
                if pcd.shape[1] == 6:
                    pointcloud.colors = o3d.utility.Vector3dVector(pcd[:, 3:6])  
                elif pcd.shape[1] > 6:
                    pointcloud.colors = o3d.utility.Vector3dVector(pcd[:, 3:6]) 
                    # pointcloud.normals = o3d.utility.Vector3dVector(pcd[:, 6:9]) 
            else:
                pcd = o3d.io.read_point_cloud(file_name)
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                rule1 = abs(points[:, 0] - position[0]) < 40
                rule2 = abs(points[:, 1] - position[1]) < 40
                rule3 = abs(points[:, 2] - position[2]) < 5
                rule = [a and b and c for a,b,c in zip(rule1, rule2, rule3)]
                
                pointcloud.points = o3d.utility.Vector3dVector(points[rule])
                pointcloud.colors = o3d.utility.Vector3dVector(colors[rule])

                # print(len(pcd.poits))
                # pointcloud.paint_uniform_color([0.5, 0.5, 0.5])
            # segment_ransac(pointcloud, return_seg=True)
            
        else:
            pass
        return pointcloud

    def load_pkl(self, filepath):
        if self.remote:
            with self.sftp_client.open(filepath, mode='rb') as f:
                dets = pkl.load(f)

        else:
            with open(filepath, 'rb') as f:
                dets = pkl.load(f)

        return dets

    def write_pcd(self, filepath, data, rgb=None, intensity=None, mode='w') -> None:

        if rgb is not None and intensity is not None:
            rgb = pypcd.encode_rgb_for_pcl(rgb.astype(np.uint8))
            dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),('rgb', np.float32), ('intensity', np.uint8)])
            pc = np.rec.fromarrays([data[:,0], data[:,1], data[:,2], rgb, intensity], dtype=dt)

        elif rgb is not None:
            rgb = pypcd.encode_rgb_for_pcl(rgb.astype(np.uint8))
            dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),('rgb', np.float32)])
            pc = np.rec.fromarrays([data[:,0], data[:,1], data[:,2], rgb], dtype=dt)

        elif intensity is not None:
            dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),('intensity', np.uint8)])
            pc = np.rec.fromarrays([data[:,0], data[:,1], data[:,2], intensity], dtype=dt)

        else:
            dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
            pc = np.rec.fromarrays([data[:,0], data[:,1], data[:,2]], dtype=dt)

        pc = pypcd.PointCloud.from_array(pc)
        with self.sftp_client.open(filepath, mode = mode) as f:
            pc.save_pcd_to_fileobj(f, compression='binary')

    def write_txt(self, filepath, data, mode='w') -> None:
        save_data = []
        for line in data:
            ll = ''
            for l in line:
                ll += f"{l:.4f}\t"
            save_data.append(ll + '\n')
        
        with self.sftp_client.open(filepath, mode = mode) as f:
            f.writelines(save_data)

    def read_poses(self, data_root_path):
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
