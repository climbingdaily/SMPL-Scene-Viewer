import os
import sys
import json
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.append('.')
sys.path.append('..')

from util import read_json_file

def save_json_file(file_name, save_dict):
    """
    Saves a dictionary into json file
    Args:
        file_name:
        save_dict:
    Returns:
    """
    with open(file_name + '_bak.json', 'w') as fp:
        try:
            json.dump(save_dict, fp, indent=4)
            with open(file_name, 'w') as fp:
                json.dump(save_dict, fp, indent=4)
        except Exception as e:
            print(f'{file_name} {e}')
            sys.exit(0)
            
    os.remove(file_name + '_bak.json')

def calibration(points2d, points3d, intrinsic, dist):
    """
    The function takes in 2D and 3D points, intrinsic and distortion parameters, and returns the
    extrinsic matrix and projection error.
    
    Args:
      points2d: A numpy array of shape (N, 2) containing the 2D coordinates of N corresponding points in
    the image.
      points3d: A numpy array of shape (N, 3) containing the 3D coordinates of N points in the world
    coordinate system.
      intrinsic: The intrinsic matrix of the camera, which contains information about its focal length,
    principal point, and skew.
      dist: The distortion coefficients of the camera. These coefficients are used to correct for lens
    distortion in the image.
    
    Returns:
      the extrinsic matrix and the projection error.
    """
    _, r, t = cv2.solvePnP(points3d, points2d, intrinsic, dist)
    R=cv2.Rodrigues(r)[0]
    
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t.reshape(3,)

    camera_points = world_to_camera(points3d, extrinsic)
    pixel_points  = camera_to_pixel(camera_points[camera_points[:, 2] > 0], 
                                    intrinsic, dist)

    proj_err = np.linalg.norm((points2d - pixel_points), axis = -1)

    return extrinsic, proj_err

def world_to_camera(X, extrinsic_matrix):
    n = X.shape[0]
    X = np.concatenate((X, np.ones((n, 1))), axis=-1).T
    X = np.dot(extrinsic_matrix, X).T
    return X[..., :3]


def camera_to_world(X, extrinsic_matrix):
    matrix = np.linalg.inv(extrinsic_matrix)
    n = X.shape[0]
    X = np.concatenate((X, np.ones((n, 1))), axis=-1).T
    X = np.dot(matrix, X).T
    return X[..., :3]

def camera_to_pixel(X, intrinsic_matrix, distortion_coefficients=np.zeros(5)):
    # focal length
    f = np.array([intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]])
    # center principal point
    c = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]])
    k = np.array([distortion_coefficients[0],
                 distortion_coefficients[1], distortion_coefficients[4]])
    p = np.array([distortion_coefficients[2], distortion_coefficients[3]])
    XX = X[..., :2] / X[..., 2:]
    r2 = np.sum(XX[..., :2]**2, axis=-1, keepdims=True)

    radial = 1 + np.sum(k * np.concatenate((r2, r2**2, r2**3),
                        axis=-1), axis=-1, keepdims=True)

    tan = 2 * np.sum(p * XX[..., ::-1], axis=-1, keepdims=True)
    XXX = XX * (radial + tan) + r2 * p[..., ::-1]
    return f * XXX + c

def load_pcd(filename, transform=None):
    # data_loader = Data_loader(False)
    # points3d = data_loader.load_point_cloud(filename)

    points3d = o3d.io.read_point_cloud(filename)
    points3d.estimate_normals()
    colors = (np.array([0.5,0.5,0.5]) + np.array(points3d.normals) / 2) * 255
    # colors = np.asarray(points3d.colors) * 255
    if transform is not None:
        points3d.transform(transform)

    return colors, np.asarray(points3d.points)

def color_point_cloud(img_path, pcd_path, T, K, dist):
    """
    This function takes an image and a point cloud, projects the point cloud onto the image, and assigns
    colors to the points based on the corresponding pixels in the image.
    
    Args:
      img_path: The file path to the image used for coloring the point cloud.
      pcd_path: The path to the point cloud file.
      T: T is the transformation matrix that transforms points from world coordinates to camera
    coordinates.
      K: K is the camera intrinsic matrix, which contains information about the focal length, principal
    point, and skew coefficient of the camera. It is used to convert 3D points in camera coordinates to
    2D pixel coordinates.
      dist: The parameter "dist" is not used in the given code snippet. It is not clear what it
    represents without further context.
    
    Returns:
      the original point cloud with colors assigned to each point based on the corresponding pixel color
    in the input image. It also saves two point clouds as PCD files, one with colors assigned to all
    points and one with colors assigned only to the visible points in the image.
    """
    img  = cv2.imread(img_path)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pcd  = o3d.io.read_point_cloud(pcd_path)
    h, w = img.shape[:2]

    camera_points = world_to_camera(np.asarray(pcd.points), T)
    rule_a = camera_points[:, 2] > 0
    pixel_points  = camera_to_pixel(camera_points[rule_a], K)
    pixel_points  = np.round(pixel_points).astype(np.int32)

    rule1 = pixel_points[:, 0] >= 0
    rule2 = pixel_points[:, 0] < img.shape[1]
    rule3 = pixel_points[:, 1] >= 0
    rule4 = pixel_points[:, 1] < img.shape[0]
    rule5 = np.asarray([a and b and c and d for a, b, c, d in zip(rule1, rule2, rule3, rule4)])

    colors = []
    for i in range(pixel_points.shape[0]):
        x, y = pixel_points[i].astype(int)
        if x < 0 or x >= w or y < 0 or y >= h: 
            colors.append([0, 0, 0])
        else:
            colors.append(img[y, x])

    all_colors         = np.asarray([0,0,0] * len(pcd.points)).reshape(-1, 3)
    all_colors[rule_a] = np.array(colors)
    pcd.colors         = o3d.utility.Vector3dVector(all_colors / 255.0)

    color_pcd = pcd.select_by_index(rule_a.nonzero()[0]).select_by_index(rule5.nonzero()[0])

    save_path = f"{os.path.splitext(pcd_path)[0]}_color_all.pcd"
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=False)
    print(f"PCD is save to {save_path}")

    save_path = f"{os.path.splitext(pcd_path)[0]}_color.pcd"
    o3d.io.write_point_cloud(save_path, color_pcd, write_ascii=False)
    print(f"Color PCD is save to {save_path}")
    return pcd

def filter_points(points, min_angle_deg=20, min_distance=0.2, max_distance=80):
    """
    Filter a 3D point cloud based on the angle between the points and the XY plane of the camera coordinate system,
    as well as the minimum and maximum distance from the camera.

    Args:
    - points: a numpy array of shape (N, 3) containing the 3D points in camera coordinates to be filtered
    - min_angle_deg: the minimum angle between a point and the XY plane of the camera coordinate system, in degrees
    - min_distance: the minimum distance between a point and the camera origin, in meters
    - max_distance: the maximum distance between a point and the camera origin, in meters

    Returns:
    - A filtered 3D point cloud as a numpy array of shape (N', 3)
    """

    # Calculate the distance and angle between each point and the XY plane
    points = points[points[:, 2] > 0]
    distance = np.linalg.norm(points, axis=1)
    xy_norm = np.linalg.norm(points[:, :2], axis=1)
    angle = np.arccos(xy_norm / distance) * 180 / np.pi

    # Filter points based on the angle, minimum distance, and maximum distance criteria
    mask = (angle > min_angle_deg) & (distance > min_distance) & (distance < max_distance)
    filtered_points = points[mask]

    return filtered_points

def plot_points_on_img(img_path, points3d, extrinsic, intrinsic, dist, colors=None, max_depth=15):
    """
    This function takes in an image, 3D points, camera extrinsic and intrinsic parameters, and projects
    the 3D points onto the image and saves the resulting image.
    
    Args:
      img_path: The file path of the image on which the points will be plotted.
      points3d: A numpy array of shape (N, 3) representing N 3D points in world coordinates.
      extrinsic: The extrinsic matrix represents the position and orientation of the camera in the world
    coordinate system. It is a 3x4 matrix that combines the rotation and translation of the camera.
      intrinsic: The intrinsic matrix of the camera, which contains information about the focal length,
    principal point, and skew. It is used to convert 3D points in camera coordinates to 2D pixel
    coordinates.
      dist: The distortion coefficients of the camera used to capture the image.
      colors: An optional array of colors for each point in the 3D space. If provided, the colors will
    be used to color the points in the image. If not provided, a default color map will be used based on
    the depth of each point.
      max_depth: The maximum depth value for the points to be plotted. Points with depth values greater
    than this will not be plotted. Defaults to 15
    """

    img = cv2.imread(img_path)

    camera_points = world_to_camera(points3d, extrinsic)
    if colors is not None:
        colors = colors[camera_points[:, 2] > 0]
    camera_points = filter_points(camera_points)
    pixel_points  = camera_to_pixel(camera_points, intrinsic, dist)
    pixel_points  = np.round(pixel_points).astype(np.int32)

    rule1 = pixel_points[:, 0] >= 0
    rule2 = pixel_points[:, 0] < img.shape[1]
    rule3 = pixel_points[:, 1] >= 0
    rule4 = pixel_points[:, 1] < img.shape[0]
    rule  = [a and b and c and d for a, b, c, d in zip(rule1, rule2, rule3, rule4)]

    camera_points = camera_points[rule]
    pixel_points  = pixel_points[rule]
    depth         = np.linalg.norm(camera_points, axis=1)
    
    if colors is not None:
        colors = colors[rule]
    else:
        colors = plt.get_cmap('hsv')(depth / max_depth)[:, :3] * 255

    for d, color, (x, y) in zip(depth, colors, pixel_points):
        if d > 0.5:
            cv2.circle(img, (x, y), 2, color=color, thickness=-1)

    save_img_path = f"{os.path.splitext(img_path)[0]}_proj.jpg"
    cv2.imwrite(save_img_path, img)
    print(f"Image saved to {save_img_path}")

    img_title = 'Points overlay'

    cv2.imshow(img_title, img)
    while cv2.getWindowProperty(img_title, cv2.WND_PROP_VISIBLE) > 0:
        cv2.imshow(img_title, img)
        key = cv2.waitKey(100)
        if key == 27:  # Press 'Esc'
            break


def plot_pixel_when_click(img_path, 
                          scale=5, 
                          size=50, 
                          cam_info_path=None):
    """
    This function allows the user to click on a pixel in an image and saves the coordinates of the
    clicked pixels in a JSON file.
    
    Args:
      img_path: The path to the image file that will be displayed and where the user can click to plot
    pixels.
      scale: The scale parameter is used to determine the zoom level of the area around the clicked
    pixel. It multiplies the size of the area around the pixel that is displayed in the "Zoom" window.
    Defaults to 5
      size: The size parameter determines the size of the zoomed-in area around the clicked pixel.
    Defaults to 50
      cam_info_path: The path to the camera information file in JSON format. If not provided, it
    defaults to 'sloper4d_cam.json' in the current working directory.
    """
    img = cv2.imread(img_path)

    h, w, c = img.shape[:3]
    
    if cam_info_path is None:
        cam_info_path = os.path.abspath('sloper4d_cam.json')

    if not os.path.exists(cam_info_path):
        save_json_file(cam_info_path, {})

    cam_info = read_json_file(cam_info_path)

    print("Image size:", w, h, c)

    # origin  = img.copy()
    if '2d_points' not in cam_info:
        cam_info['2d_points'] = []

    list_xy = cam_info['2d_points']
    for index, (a, b) in enumerate(list_xy):
        cv2.circle(img, (a, b), 2, (255, 0, 0), thickness=-1)
        cv2.putText(img, f'{index}: {a:.0f} {b:.0f}', (a, b), 
                    cv2.FONT_HERSHEY_PLAIN,
                    2.0, (0, 255, 0), thickness=1)

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
            x = round(x, 2)
            y = round(y, 2)
            list_xy.append([x, y])
            print(f'Clicked pixel coordinates: {x:.1f} {y:.1f}')

            a, b = x, y
            cv2.circle(img, (a, b), 2, (255, 0, 0), thickness=-1)
            cv2.putText(img, f'{len(list_xy)-1}: {a:.0f} {b:.0f}', (a, b), 
                        cv2.FONT_HERSHEY_PLAIN,
                        2.0, (0, 255, 0), thickness=1)
            cv2.imshow('image', img)

        elif event == cv2.EVENT_MOUSEMOVE:
            ymin  = max(0, y-size)
            ymax  = min(h, y+size)
            xmin  = max(0, x-size)
            xmax  = min(w, x+size)
            xsize = (xmax - xmin) * scale
            ysize = (ymax - ymin) * scale

            area = img[ymin:ymax, xmin:xmax].copy()
            area = cv2.resize(area, dsize=(xsize, ysize),
                            interpolation=cv2.INTER_LINEAR)
            cv2.circle(area, ((x-xmin) * scale, (y-ymin) * scale), 
                       3, (255, 0, 0), thickness=-1)
            cv2.putText(area, f'{x:.0f} {y:.0f}', 
                        ((x-xmin) * scale, (y-ymin) * scale), 
                        cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
            cv2.imshow('Zoom', area)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_EVENT_LBUTTONDOWN)

    while cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) > 0:
        if img is not None: # 检查img是否被赋值
            cv2.imshow('image', img)
        key = cv2.waitKey(100)
        if key == 27:  # 按下Esc键
            break

    imgdir = os.path.dirname(img_path)
    cam_info['2d_points'] = list_xy
    cam_info['ex_img'] = os.path.join(imgdir, 'picked.jpg')
    save_json_file(cam_info_path, cam_info)

    cv2.imwrite(os.path.join(imgdir, 'picked.jpg'), img)

DEFAULT_INTRINSIC = [[599.628, 0.000, 971.613], 
                     [0.000, 599.466, 540.258], 
                     [0.000, 0.000, 1.000]]
DEFAULT_DIST = [0.003, -0.003, -0.001, 0.004, 0.000]

def load_cam_info(filename):
    with open(filename, 'r') as f:
        cam_info  = json.load(f)
        # assert '2d_points' in cam_info and '3d_points' in cam_info,  f"2D pixels or 3D points not exist in {filename}"
        
        points2d  = np.array(cam_info['2d_points'], dtype=np.float32)  if '2d_points' in cam_info else None
        points3d  = np.array(cam_info['3d_points'], dtype=np.float32)  if '3d_points' in cam_info else None

        extrinsic = cam_info['extrinsics'] if 'extrinsics' in cam_info else None
        intrinsic = cam_info['intrinsics'] if 'intrinsics' in cam_info else DEFAULT_INTRINSIC
        dist      = cam_info['dist'] if 'dist' in cam_info else DEFAULT_DIST

        img_path  = cam_info['ex_img'] if 'ex_img' in cam_info else None

        extrinsic = np.array(extrinsic, dtype=np.float32)
        intrinsic = np.array(intrinsic, dtype=np.float32)
        dist      = np.array(dist, dtype=np.float32)
        # dist      = np.array([0,0,0,0,0], dtype=np.float32)
    
    return points2d, points3d, extrinsic, intrinsic, dist, img_path

def cali_ex(cam_info_path):
    points2d, points3d, _, intrinsic, dist, img_path = load_cam_info(cam_info_path)

    extrinsic, proj_err = calibration(points2d, points3d, 
                                intrinsic.reshape(3,3), dist)

    camera_position  = - extrinsic[:3, :3].T @ extrinsic[:3, 3]
    camera_direction = extrinsic[:3, :3].T @ np.array([[1,0,0],[0,-1,0],[0,0,-1]])

    print('========================')
    print(f'camera intrinsic:    \n {intrinsic.reshape(3,3)}')
    print(f'camera distortion:   \n {dist}')
    print(f'camera extrinsic:    \n {extrinsic}')
    print(f'Re-projection error: \n {proj_err}')
    print(f'camera position:     \n {camera_position}')
    print(f'camera orientation:  \n {camera_direction}')
    print('========================')
    
    with open(cam_info_path, 'r') as f:
        cam_info  = json.load(f)
    cam_info['extrinsics'] = extrinsic.tolist()
    cam_info['intrinsics'] = intrinsic.tolist()
    cam_info['dist'] = dist.tolist()
    save_json_file(cam_info_path, cam_info)

    if img_path is not None:
        plot_points_on_img(img_path, points3d, extrinsic, intrinsic, dist)

    return extrinsic, intrinsic, dist

def waymo_cam_to_extrinsic(cam):
    """
    It takes a camera matrix and returns the extrinsic matrix
    
    Args:
      cam: the camera matrix
    
    Returns:
      The extrinsic matrix
    """
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.array([[0,-1,0],[0,0,-1],[1,0,0]]) @ cam[:3, :3].T
    extrinsic[:3, 3] = -(extrinsic[:3, :3] @ cam[:3, 3])
    return extrinsic

if __name__ == '__main__':  
    import configargparse
    parser = configargparse.ArgumentParser()
                        
    parser.add_argument("--cam_info", "-C", type=str, default=None)
    parser.add_argument("--img_path", "-I", type=str, default=None)
    parser.add_argument("--pc_path",  "-P", type=str, default=None)
    parser.add_argument("--waymo", action='store_true')

    args = parser.parse_args()
    
    if args.img_path is not None and args.cam_info is not None and args.pc_path is not None:
        _, _, extrinsic, intrinsic, dist, _ = load_cam_info(args.cam_info)
        cam_info = read_json_file(args.cam_info)
        if args.pc_path.endswith('.txt'):
            points = np.loadtxt(args.pc_path)
        elif args.pc_path.endswith('.pcd'):
            _, points = load_pcd(args.pc_path)
        else:
            raise NotImplementedError
        
        if args.waymo:
            extrinsic = waymo_cam_to_extrinsic(extrinsic)
            dist = [0,0,0,0,0]

        plot_points_on_img(args.img_path, 
                           points, extrinsic, intrinsic, dist)
 
        color_point_cloud(args.img_path, 
                          args.pc_path, extrinsic, intrinsic, dist)

    elif args.img_path is not None:
        # only image_path
        plot_pixel_when_click(args.img_path, cam_info_path=args.cam_info)
        
    elif args.cam_info is not None:
        # only cam_info
        cali_ex(args.cam_info)
