import open3d as o3d
import numpy as np
from os.path import dirname, join, basename, split, splitext

def generate_mesh(file_path, depth=12, radius=0.1, vis=False):
    
    print('============')
    print(f'Generating mesh from the {file_path}, it will cost a little bit long time...')

    import matplotlib.pyplot as plt
    pcd = o3d.io.read_point_cloud(file_path)

    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=20))

    if vis:
        o3d.visualization.draw_geometries([pcd],
                                    zoom=0.664,
                                    front= [-0.4761, -0.4698, -0.7434],
                                    lookat=[1.8900, 3.2596, 0.9284],
                                    up=[0.2304, -0.8825, 0.4101])
    print('run Poisson surface reconstruction')
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth)
            
    print(mesh)

    if vis:
        o3d.visualization.draw_geometries([mesh],
                                    zoom=0.664,
                                    front=[-0.4761, -0.4698, -0.7434],
                                    lookat=[1.8900, 3.2596, 0.9284],
                                    up=[0.2304, -0.8825, 0.4101])
        print('visualize densities')

    densities = np.asarray(densities)
    # density_colors = plt.get_cmap('plasma')(
    #     (densities - densities.min()) / (densities.max() - densities.min()))
    # density_colors = density_colors[:, :3]

    # density_mesh = o3d.geometry.TriangleMesh()
    # density_mesh.vertices = mesh.vertices
    # density_mesh.triangles = mesh.triangles
    # density_mesh.triangle_normals = mesh.triangle_normals
    # density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)

    # if vis:
    #     o3d.visualization.draw_geometries([density_mesh],
    #                                 zoom=0.664,
    #                                 front=[-0.4761, -0.4698, -0.7434],
    #                                 lookat=[1.8900, 3.2596, 0.9284],
    #                                 up=[0.2304, -0.8825, 0.4101])

    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, 0.12)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.merge_close_vertices(0.05)
    
    iters = 5
    print(f'filter with Laplacian with {iters} iterations')
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=iters)
    mesh.compute_vertex_normals()

    print(mesh)
    save_path = splitext(file_path)[0] + '.ply'
    o3d.io.write_triangle_mesh(save_path, mesh)

    print(f'Mesh saved at {save_path}')

    if vis:
        o3d.visualization.draw_geometries([mesh],
                                        zoom=0.664,
                                        front=[-0.4761, -0.4698, -0.7434],
                                        lookat=[1.8900, 3.2596, 0.9284],
                                        up=[0.2304, -0.8825, 0.4101])