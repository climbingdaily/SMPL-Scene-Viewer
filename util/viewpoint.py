import numpy as np
from pkg_resources import ExtractionError
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from .tool_func import filterTraj

def make_cloud_in_vis_center(point_cloud):
    center = point_cloud.get_center()
    yaw = np.arctan2(center[1], center[0])

    # rot the points, make them on the X-axis
    rot = R.from_rotvec(np.array([0, 0, -yaw])).as_matrix()

    # put points in 5 meters distance
    trans_x = 5 - (rot @ center)[0]

    rt = np.concatenate((rot.T, np.array([[trans_x, 0, -center[-1]]]))).T
    rt = np.concatenate((rt, np.array([[0, 0, 0, 1]])))

    point_cloud.transform(rt)
    # point_cloud.traslate(rt)

    return rt, center

def generate_views(position, direction, filter=True, rad=np.deg2rad(15)):
    assert len(position) == len(direction)

    mocap_init = np.array([[-1, 0, 0, ], [0, 0, 1], [0, 1, 0]])
    base_view = {
        "trajectory" : 
        [
            {
                "field_of_view" : 90.0,
                "front" : [ 0, -np.cos(rad), np.sin(rad)],
                "lookat" : [ 0, 1, 1.7],
                "up" : [ 0, np.sin(rad), np.cos(rad)],
                "zoom" : 0.0065
            }
        ],
    }

    if direction[0].shape[0] == 4:
        func = R.from_quat
    else:
        func = R.from_matrix

    init_direction = func(direction[0]).as_matrix()

    if filter:
       position = filterTraj(position)

    view_list = []
    extrinsic_list = []
    for t, r in zip(position, direction):
        view = deepcopy(base_view)
        # rot = func(r).as_euler('xyz', degrees=False)
        rot = func(r).as_matrix() @ init_direction.T

        view['trajectory'][0]['lookat'] = t.tolist()
        view['trajectory'][0]['up'] = rot @ np.array(view['trajectory'][0]['up'])
        view['trajectory'][0]['front'] = rot @ np.array(view['trajectory'][0]['front'])
        view_list.append(view)
        
        front = view['trajectory'][0]['front']
        up = view['trajectory'][0]['up']
        origin = view['trajectory'][0]['lookat']

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = np.stack([-np.cross(front, up), -up, -front])
        extrinsic[:3, 3] = - (extrinsic[:3, :3] @ origin)
        extrinsic_list.append(extrinsic)
    
    return view_list, extrinsic_list
