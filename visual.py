from argparse import ArgumentParser
import open3d as o3d
import numpy as np
import os

# from https://github.com/intel-isl/Open3D/blob/master/examples/Python/Advanced/load_save_viewpoint.py
def fun():
    path_xyz = "/home/pfjiang/fsdownload/2023-04-24-10-09-10_level_230438341-原料/4.xyz"
    points = np.loadtxt(path_xyz)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # 设置点云显示的颜色（0-1）
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.array([1, 1, 1.0])
    vis.add_geometry(point_cloud)

    vis.run()
    vis.destroy_window()
    # point_cloud.paint_uniform_color([43/255, 76/255, 14/255])
    # o3d.visualization.draw_geometries([point_cloud])

def eval():
    path = "./outputs1/patchmatchnet011_l3.ply"
    path1 = "./outputs_tanks/Panther.ply"
    path2 = "./outputs_blend/blend001_l3.ply"
    # "/data/Points/stl/stl001_total.ply"
    # "./output/scan1/fused.ply"
    # "./output_source/scan4/fused.ply"

    pcd = o3d.io.read_point_cloud(path)
    print(f'contains {len(pcd.points) / 1e6:.2f} M points')
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.array([1, 1, 1.0])
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    eval()
    #fun()
