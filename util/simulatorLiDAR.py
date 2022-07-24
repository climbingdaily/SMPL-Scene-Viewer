import os
import open3d as o3d
import numpy as np
from glob import glob
import configargparse
from multiprocessing import Pool
import functools
import time

lab_list = [5, 6, 7, 8, 39, 40, 41, 42]             # floor_height = -4.9 m
haiyun_list = [24,25,26,27]                         # floor_height = -5.7 m
haiyun_list_2 = [28,29,30,31,32,33,34,35,36,37,38]    # floor_height = -6.5 m

done_list = [7, 8, 24, 25, 29, 30, 31, 32, 34, 35, 36, 38, 39, 42]

def hidden_point_removal(pcd, camera = [0, 0, 0]):
    # diameter = np.linalg.norm(
    # np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

    # print("Define parameters used for hidden_point_removal")
    
    # camera = [view_point[0], view_point[0], diameter]
    # camera = view_point
    dist = np.linalg.norm(pcd.get_center())
    # radius = diameter * 100
    radius = dist * 1000

    # print("Get all points that are visible from given view point")
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    # print("Visualize result")
    pcd = pcd.select_by_index(pt_map)
    return pcd


def select_points_on_the_scan_line(points, view_point=None, scans=64, line_num=1024, fov_up=16.2, fov_down=-16.2, precision=1.1):
    
    fov_up = np.deg2rad(fov_up)
    fov_down = np.deg2rad(fov_down)
    fov = abs(fov_down) + abs(fov_up)

    ratio = fov/(scans - 1)   # 64bins 的竖直分辨率
    hoz_ratio = 2 * np.pi / (line_num - 1)    # 64bins 的水平分辨率
    # precision * np.random.randn() 

    # print(points.shape[0])
    
    if view_point is not None:
        points -= view_point
    depth = np.linalg.norm(points, 2, axis=1)
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    yaw = np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    
    # pc_ds = []

    saved_box = { s:{} for s in np.arange(scans)}

    #### 筛选fov范围内的点
    for idx in range(0, points.shape[0]):
        rule1 =  pitch[idx] >= fov_down
        rule2 =  pitch[idx] <= fov_up
        rule3 = abs(pitch[idx] % ratio) < ratio * 0.4
        rule4 = abs(yaw[idx] % hoz_ratio) < hoz_ratio * 0.4
        if rule1 and rule2:
            scanid = np.rint((pitch[idx] + 1e-4) / ratio) + scans // 2
            pointid = np.rint((yaw[idx] + 1e-4) // hoz_ratio)
            if scanid < 0 or scanid >= scans:
                continue
            if pointid > 0 and scan_x[idx] < 0:
                pointid += 1024 // 2
            elif pointid < 0 and scan_y[idx] < 0:
                pointid += 1024 // 2
            
            z = np.sin(scanid * ratio + fov_down)
            xy = abs(np.cos(scanid * ratio + fov_down))
            y = xy * np.sin(pointid * hoz_ratio)
            x = xy * np.cos(pointid * hoz_ratio)

            # 找到根指定激光射线夹角最小的点
            cos_delta_theta = np.dot(points[idx], np.array([x, y, z])) / depth[idx]
            delta_theta = np.arccos(abs(cos_delta_theta))
            if pointid in saved_box[scanid]:
                if delta_theta < saved_box[scanid][pointid]['delta_theta']:
                    saved_box[scanid][pointid].update({'points': points[idx], 'delta_theta': delta_theta})
            else:
                saved_box[scanid][pointid] = {'points': points[idx], 'delta_theta': delta_theta}

    save_points  =[]
    for key, value in saved_box.items():
        if len(value) > 0:
            for k, v in value.items():
                save_points.append(v['points']) 

    # pc_ds = np.array(pc_ds)
    save_points = np.array(save_points)


    #####
    # print(f'\r{save_points.shape}', end=' ', flush=True)
    pc=o3d.open3d.geometry.PointCloud()
    pc.points= o3d.open3d.utility.Vector3dVector(save_points)
    pc.paint_uniform_color([0.5, 0.5, 0.5])
    # pc.estimate_normals()

    return pc

def translate(points, y_dist=5, z_height=3.1):
    """
    params: 
        向human向Y轴拉近 @y_dist, 向上平移 @z_height
    returns: 
        points
    """
    points.translate(np.array([0, -y_dist, z_height]))
    return points

def sample_data(file_path, out_root, shorter_dist, move_z, rot):

    # save data path
    filename, _ = os.path.splitext(os.path.basename(file_path))
    save_path = os.path.join(out_root, filename + '.pcd')
    if os.path.exists(save_path):
        return

    print(f'\rProcess {file_path}', end='\t', flush=True)

    time1 = time.time()

    point_clouds = o3d.open3d.io.read_triangle_mesh(file_path)

    time2 = time.time()

    # print(f'Read {(time2- time1):.3f} s.')

    if len(point_clouds.triangles) > 0:
        point_clouds.compute_vertex_normals()
        point_clouds = point_clouds.sample_points_poisson_disk(100000)
    else:
        point_clouds = o3d.io.read_point_cloud(file_path)
        

    # point_clouds
    view_point = point_clouds.get_center()
    view_point[0] += 0
    view_point[1] += -6.0
    view_point[2] += 0

    # process data
    time3 = time.time()
    # print(f'CPU {(time3- time2):.3f} s.')

    try:
        point_clouds.translate(np.array([0, -shorter_dist, move_z])) # human向Y轴拉近 @shorter_dist, 向上平移 @move_z
        # point_clouds.rotate(rot) #这个仅围绕中心点旋转
        point_clouds.points = o3d.utility.Vector3dVector(np.asarray(point_clouds.points) @ rot.T)
        point_clouds = hidden_point_removal(point_clouds)
        point_clouds = select_points_on_the_scan_line(np.asarray(point_clouds.points))
        o3d.io.write_point_cloud(save_path, point_clouds)
    except:
        print(f'cannot sample data from {save_path} !!!')

    time4 = time.time()
    # print(f'{(time4- time3):.3f} s.')
    

def simulatorLiDAR(root, out_root=None, shorter_dist=0, move_z = 0, rot = np.eye(3), threads = 1):
    
    if out_root is None:
        out_root = root.replace('pose_rot', 'sampled_ouster')
    os.makedirs(out_root, exist_ok=True)        

    filelist = sorted(glob(root+'/*.ply'))

    time1 = time.time()

    if threads ==  1:
        for index in range(0, len(filelist)):
            sample_data(filelist[index], out_root, shorter_dist, move_z)
    
    elif threads > 1:
        with Pool(threads) as p:
            p.map(functools.partial(sample_data, out_root=out_root,
                  shorter_dist=shorter_dist, move_z=move_z, rot = rot), filelist)
    else:
        print(f'Input threads: {threads} error')

    time2 = time.time()

    print(f'\n {root} processed. Consumed {(time2- time1):.2f} s.')

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument("--file_path", '-F', type=str,
                        default='/hdd/dyd/lidarcap/labels/3d/pose_rot')
    parser.add_argument("--threads", '-T', type=int,
                        default='16')
    args = parser.parse_args()

    for folder in sorted(os.listdir(args.file_path), key=lambda x: int(x)):
        process_folder = os.path.join(args.file_path, folder)
        
        # # 模拟生成线扫的激光雷达
        if int(folder) in lab_list:
            simulatorLiDAR(process_folder, shorter_dist = 5, move_z=4.9-1.85, rot = np.eye(3), threads = 16)

        elif int(folder) in haiyun_list:
            simulatorLiDAR(process_folder, shorter_dist = 10, move_z=5.7-1.85, rot = np.eye(3), threads = 16)

        elif int(folder) in haiyun_list_2:
            simulatorLiDAR(process_folder, shorter_dist = 5, move_z=6.5-1.85, rot = np.eye(3), threads = 16)

        else:
            print(f'No {process_folder}')
