################################################################################
# File: \creat_mesh.py                                                         #
# Created Date: Friday August 19th 2022                                        #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
################################################################################

import numpy as np
import open3d as o3d
from copy import deepcopy

def creat_chessboard(lenght = 2, size_x = 52, size_y = 52, white = [1,1,1], black = [0.3, 0.3, 0.3]):
    """
    It creates a chessboard of a given size, with a given number of squares, and with a given color for
    the white and black squares
    
    Args:
      lenght: the length of each square. Defaults to 1
      size_x: The width of the chessboard. Defaults to 10
      size_y: The height of the chessboard. Defaults to 10
      white: The color of the white squares
      black: The color of the black squares.
    
    Returns:
      A list of triangle meshes.
    """

    square_box = o3d.geometry.TriangleMesh.create_box(
        lenght, lenght, 0.01, create_uv_map=True, map_texture_to_each_face=True)

    boards = o3d.geometry.TriangleMesh()

    num_x = size_x//lenght
    num_y = size_y//lenght

    for i in range(num_x):
        for j in range(num_y):
            g = deepcopy(square_box)
            g.translate((i * lenght, j * lenght, 0))
            color = black if (i + j) % 2 == 0 else white
            g.paint_uniform_color(color)
            boards += g
    boards.compute_triangle_normals()
    boards.translate((-size_x/2, -size_y/2, -0.01))
    return [boards]

def creat_plane(lenght = 6, size_x = 24, size_y = 24, material = 'Tiles074'):
    import open3d.visualization as vis

    ground_plane = o3d.geometry.TriangleMesh.create_box(
        lenght, lenght, 0.01, create_uv_map=True, map_texture_to_each_face=True)
    ground_plane.compute_triangle_normals()
    # rotate_180 = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi, 0, 0))
    # ground_plane.rotate(rotate_180)
    ground_plane.translate((-lenght/2, -lenght/2, -0.01))
    ground_plane.paint_uniform_color((1, 1, 1))
    ground_plane = o3d.t.geometry.TriangleMesh.from_legacy(ground_plane)

    # Material to make ground plane more interesting - a rough piece of glass
    mat_ground = vis.Material("defaultLit")
    mat_ground.scalar_properties['roughness'] = 0.1
    mat_ground.scalar_properties['reflectance'] = 0.72
    mat_ground.scalar_properties['transmission'] = 0.6
    mat_ground.scalar_properties['thickness'] = 0.3
    mat_ground.scalar_properties['absorption_distance'] = 0.1
    mat_ground.vector_properties['absorption_color'] = np.array(
        [0.82, 0.98, 0.972, 1.0])
    mat_ground.texture_maps['albedo'] = o3d.t.io.read_image(
        f"demo_scene_assets/{material}_Color.jpg")
    mat_ground.texture_maps['roughness'] = o3d.t.io.read_image(
        f"demo_scene_assets/{material}_Roughness.png")
    mat_ground.texture_maps['normal'] = o3d.t.io.read_image(
        f"demo_scene_assets/{material}_NormaDX.jpg")
    ground_plane.material = mat_ground

    planes = []
    num_x = size_x//lenght
    num_y = size_y//lenght
    for i in range(-(num_x//2), (num_x+1)//2):
        for j in range(-(num_y//2), (num_y+1)//2):

            g = ground_plane.clone()
            g.material = mat_ground

            g.translate((i * lenght + (num_x+1) % 2 * lenght/2, j * lenght + (num_y+1) % 2 * lenght/2, 0))
            planes.append(g)
    return planes

def add_material(geometry, material = 'Tiles074'):
    import open3d.visualization as vis
    geometry = o3d.t.geometry.TriangleMesh.from_legacy(geometry)
    # Material to make ground plane more interesting - a rough piece of glass
    mat_ground = vis.Material("defaultLit")
    mat_ground.scalar_properties['roughness'] = 0.1
    mat_ground.scalar_properties['reflectance'] = 0.72
    mat_ground.scalar_properties['transmission'] = 0.6
    mat_ground.scalar_properties['thickness'] = 0.3
    mat_ground.scalar_properties['absorption_distance'] = 0.1
    mat_ground.vector_properties['absorption_color'] = np.array(
        [0.82, 0.98, 0.972, 1.0])
    mat_ground.texture_maps['albedo'] = o3d.t.io.read_image(
        f"demo_scene_assets/{material}_Color.jpg")
    mat_ground.texture_maps['roughness'] = o3d.t.io.read_image(
        f"demo_scene_assets/{material}_Roughness.png")
    mat_ground.texture_maps['normal'] = o3d.t.io.read_image(
        f"demo_scene_assets/{material}_NormaDX.jpg")
    geometry.material = mat_ground
    return geometry

def create_ground(
    center=[0, 0, 0], xdir=[1, 0, 0], ydir=[0, 1, 0], # 位置
    step=1, xrange=10, yrange=10, # 尺寸
    white=[1., 1., 1.], black=[0.,0.,0.], # 颜色
    two_sides=True
    ):
    """
    borrow from https://github.com/zju3dv/EasyMocap/blob/master/easymocap/visualize/geometry.py

    It creates a checkerboard ground plane
    
    Args:
      center: the center of the ground
      xdir: the direction of the x axis
      ydir: the direction of the y axis of the ground
      step: the size of each square. Defaults to 1
      xrange: the number of squares in the x direction. Defaults to 10
      yrange: the number of squares in the y direction. Defaults to 10
      white: the color of the white squares
      black: the color of the black squares
      two_sides: whether to create a ground with two sides. Defaults to True
    
    Returns:
      A dictionary with keys 'vertices', 'faces', 'colors', and 'name'.
    """

    if isinstance(center, list):
        center = np.array(center)
        xdir = np.array(xdir)
        ydir = np.array(ydir)
    print('[Vis Info] {}, x: {}, y: {}'.format(center, xdir, ydir))
    xdir = xdir * step
    ydir = ydir * step
    vertls, trils, colls = [],[],[]
    cnt = 0
    min_x = -xrange if two_sides else 0
    min_y = -yrange if two_sides else 0
    for i in range(min_x, xrange):
        for j in range(min_y, yrange):
            point0 = center + i*xdir + j*ydir
            point1 = center + (i+1)*xdir + j*ydir
            point2 = center + (i+1)*xdir + (j+1)*ydir
            point3 = center + (i)*xdir + (j+1)*ydir
            if (i%2==0 and j%2==0) or (i%2==1 and j%2==1):
                col = white
            else:
                col = black
            vert = np.stack([point0, point1, point2, point3])
            col = np.stack([col for _ in range(vert.shape[0])])
            tri = np.array([[2, 3, 0], [0, 1, 2]]) + vert.shape[0] * cnt
            cnt += 1
            vertls.append(vert)
            trils.append(tri)
            colls.append(col)
    vertls = np.vstack(vertls)
    trils = np.vstack(trils)
    colls = np.vstack(colls)

    ground = o3d.geometry.TriangleMesh()
    
    ground.vertices = o3d.utility.Vector3dVector(vertls)
    ground.vertex_colors = o3d.utility.Vector3dVector(colls)
    ground.triangles = o3d.utility.Vector3iVector(trils)
    return [ground]
    # return {'vertices': vertls, 'faces': trils, 'colors': colls, 'name': 'ground'}

if __name__ == "__main__":
    creat_chessboard()
    create_ground()
    creat_plane()