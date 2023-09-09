# Partially borrowed from Nacho's lidar odometry (KISS-ICP)

from abc import ABC
import copy
from functools import partial
import os
from typing import Callable, List
import time
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion

import numpy as np
import open3d as o3d

YELLOW = np.array([1, 0.706, 0])
RED = np.array([128, 0, 0]) / 255.0
BLACK = np.array([0, 0, 0]) / 255.0
GOLDEN = np.array([1.0, 0.843, 0.0])

# colors used for visualization (better to add this to the visualizer)
color_table = [[230. / 255., 0., 0.],  # red
               [60. / 255., 180. / 255., 75. / 255.],  # green
               [0., 0., 255. / 255.],  # blue
               [255. / 255., 0, 255. / 255.],
               [255. / 255., 165. / 255., 0.],
               [128. / 255., 0, 128. / 255.],
               [0., 255. / 255., 255. / 255.],
               [210. / 255., 245. / 255., 60. / 255.],
               [250. / 255., 190. / 255., 190. / 255.],
               [0., 128. / 255., 128. / 255.]
               ]

def text_3d(text, pos, direction=None, degree=90.0, font='/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', font_size=20):
    """
    Generate a 3D text point cloud used for visualization.
    Author: Jiahui Huang, heiwang1997 at github
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (1., 0., 0.)

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    
    pcd.scale(0.2, np.asarray([[0,0,0]]).T)        
    
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


class StubVisualizer(ABC):
    def __init__(self):
        pass

    def update(self, frame, target, pose):
        pass


class OptVisualizer(StubVisualizer):
    # Public Interaface ----------------------------------------------------------------------------
    def __init__(self, frame_axis_len = 0.1, pause_time_s = 1e-2):
        # Initialize GUI controls
        self.block_vis = False
        self.play_crun = True
        self.reset_bounding_box = True
        self.skip = False

        # Create data
        self.scan = o3d.geometry.PointCloud()
        self.gt_scan = o3d.geometry.PointCloud()
        self.frame = o3d.geometry.TriangleMesh()
        self.mesh = o3d.geometry.TriangleMesh()
        self.cano_mesh = o3d.geometry.TriangleMesh()
        self.cano_frame =  o3d.geometry.TriangleMesh()
        self.txt = o3d.geometry.PointCloud()
        
        self.pause_time_s = pause_time_s # pause x second after one visualization
        self.cano_tran = np.zeros(3)
        self.frame_axis_len = frame_axis_len

        # Initialize visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self._register_key_callbacks()
        self._initialize_visualizer()

        # Visualization options
        self.render_mesh = True
        self.render_frame = True
        self.render_gt = True

        self.vis_cano = False

        self.global_view = False
        self.view_control = self.vis.get_view_control()
        self.camera_params = self.view_control.convert_to_pinhole_camera_parameters()

    def update_view(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def pause_view(self):
        # check if there will be a pause
        while self.block_vis:
            self.update_view()
            if self.play_crun:
                break    

    def clean_vis(self):
        self.skip = False
        self.vis.remove_geometry(self.mesh) 
        self.vis.remove_geometry(self.scan) 
        self.vis.remove_geometry(self.gt_scan)
        self.vis.remove_geometry(self.frame) 
        self.vis.remove_geometry(self.cano_mesh) 
        self.vis.remove_geometry(self.cano_frame) 
        self.vis.remove_geometry(self.txt) 
        self.reset_bounding_box = True

    def update(self, scan, pose, mesh = None):
        self._update_geometries(scan, pose, mesh)
        self.update_view()
        self.pause_view()

    def update_mesh(self, mesh):
        self._update_mesh(mesh)
        self.update_view()
        self.pause_view()

    def update_mesh_pose(self, cano_mesh, transform, iter):
        tran_mesh = copy.deepcopy(cano_mesh)
        tran_mesh.transform(transform)
        cano_mesh_vis = copy.deepcopy(cano_mesh)
        cano_mesh_vis.translate(self.cano_tran)

        self._update_mesh_cano(tran_mesh, cano_mesh_vis, transform, iter)

        self.update_view()
        self.pause_view()

    
    def add_scan(self, scan):
        self.scan = scan
        self.vis.add_geometry(self.scan, self.reset_bounding_box) 

        self.cano_tran = scan.get_axis_aligned_bounding_box().get_center()
        self.cano_tran[0] += (2 * self.frame_axis_len) # shifted a bit for visualization

        self.txt_tran = np.copy(self.cano_tran)
        self.txt_tran[0] -= (3.5 * self.frame_axis_len)
        
        if self.vis_cano:
            self.cano_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.frame_axis_len, origin=self.cano_tran)
            self.vis.add_geometry(self.cano_frame, self.reset_bounding_box)  

        self.txt = text_3d(str(0), self.txt_tran)   
        self.vis.add_geometry(self.txt, self.reset_bounding_box)  

        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False
        
        self.update_view()

        self.pause_view()

    def add_gt_scan(self, gt_scan):
        if self.render_gt:
            self.gt_scan = gt_scan
        else:
            self.gt_scan = o3d.geometry.PointCloud()
        
        self.vis.add_geometry(self.gt_scan, self.reset_bounding_box) 

        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False
        
        self.update_view()

        self.pause_view()
    
    def destroy_window(self):
        self.vis.destroy_window()
    
    def stop(self):
        self.block_vis = True
        if self.play_crun:
            self.play_crun = not self.play_crun
        while self.block_vis:
            self.vis.poll_events()
            self.vis.update_renderer()
            if self.play_crun:
                break
        return self.skip

    # Private Interaface ---------------------------------------------------------------------------
    def _initialize_visualizer(self):
        w_name = self.__class__.__name__
        self.vis.create_window(window_name=w_name, width=1920, height=1080)
        self.vis.add_geometry(self.scan)
        self.vis.add_geometry(self.frame)
        self.vis.add_geometry(self.mesh)
        self._set_white_background(self.vis)
        self.vis.get_render_option().point_size = 10 # change default point cloud visualization size here
        self.vis.get_render_option().light_on = True
        self.vis.get_render_option().mesh_show_wireframe = False
        self.vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color
        print(100 * "*")
        print(f"{w_name} initialized. Press [SPACE] to pause/start, [N] to skip, [V] to switch back to the default viewpoint, [M] to toggle the mesh, [F] to toggle the pose frame, [ESC] to exit.")

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.vis.register_key_callback(ord(str(key)), partial(callback))

    # TODO add (hide canonical frame)
    # TODO add (hide ground truth mesh)
    def _register_key_callbacks(self):
        self._register_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["V"], self._toggle_view)
        self._register_key_callback(["F"], self._toggle_frame)
        self._register_key_callback(["M"], self._toggle_mesh)
        self._register_key_callback(["C"], self._toggle_cano)
        self._register_key_callback(["N"], self._skip)
        self._register_key_callback(["G"], self._toggle_gt)
        # self._register_key_callback(["B"], self._set_black_background)
        # self._register_key_callback(["W"], self._set_white_background)

    def _set_black_background(self, vis):
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]

    def _set_white_background(self, vis):
        vis.get_render_option().background_color = [1.0, 1.0, 1.0]

    def _quit(self, vis):
        print("Destroying Visualizer")
        vis.destroy_window()
        os._exit(0)

    def _next_frame(self, vis):
        self.block_vis = not self.block_vis

    def _skip(self, vis):
        self.skip = True
        self.play_crun = not self.play_crun

    def _start_stop(self, vis):
        self.play_crun = not self.play_crun

    def _toggle_frame(self, vis):
        self.render_frame = not self.render_frame
        return False

    def _toggle_mesh(self, vis):
        self.render_mesh = not self.render_mesh
        return False
    
    def _toggle_gt(self, vis):
        self.render_gt = not self.render_gt
        return False
    
    def _toggle_cano(self, vis):
        self.vis_cano = not self.vis_cano
        return False
    
    def _update_geometries(self, scan, pose, mesh = None):
        # Scan (toggled by "F")
        if self.render_frame:
            self.scan.points = o3d.utility.Vector3dVector(scan.points)
            self.scan.paint_uniform_color(GOLDEN)
        else:
            self.scan.points = o3d.utility.Vector3dVector()

        # Always visualize the coordinate frame
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.frame_axis_len, origin=np.zeros(3))
        self.frame = self.frame.transform(pose)
    
        # Mesh Map (toggled by "M")
        if self.render_mesh:
            if mesh is not None:
                self.vis.remove_geometry(self.mesh, self.reset_bounding_box)
                self.mesh = mesh
                self.vis.add_geometry(self.mesh, self.reset_bounding_box)
        else:
            self.vis.remove_geometry(self.mesh, self.reset_bounding_box) 

        self.vis.update_geometry(self.scan)
        self.vis.add_geometry(self.frame, self.reset_bounding_box)            

        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False

    def _update_mesh(self, mesh):
        if mesh is not None:
            self.vis.remove_geometry(self.mesh, self.reset_bounding_box) 
            self.mesh = mesh
            self.vis.add_geometry(self.mesh, self.reset_bounding_box)
        
        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False

    def _update_mesh_cano(self, mesh, cano_mesh, transform, iter):
        self.vis.remove_geometry(self.mesh, self.reset_bounding_box)

        if self.vis_cano:
            self.vis.remove_geometry(self.cano_mesh, self.reset_bounding_box)
        if self.render_mesh:
            self.mesh = mesh
            
            self.vis.add_geometry(self.mesh, self.reset_bounding_box)
            if self.vis_cano:
                self.cano_mesh = cano_mesh
                self.vis.add_geometry(self.cano_mesh, self.reset_bounding_box)

        self.vis.remove_geometry(self.frame, self.reset_bounding_box)
        if self.render_frame:
            self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.frame_axis_len, origin=np.zeros(3))
            self.frame = self.frame.transform(transform)
            self.vis.add_geometry(self.frame, self.reset_bounding_box)

        self.vis.remove_geometry(self.txt, self.reset_bounding_box)
        self.txt = text_3d(str(iter), self.txt_tran)   
        self.vis.add_geometry(self.txt, self.reset_bounding_box)  
        
        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False


    def _toggle_view(self, vis):
        self.global_view = not self.global_view
        vis.update_renderer()
        vis.reset_view_point(True)
        current_camera = self.view_control.convert_to_pinhole_camera_parameters()
        if self.camera_params and not self.global_view:
            self.view_control.convert_from_pinhole_camera_parameters(self.camera_params)
        self.camera_params = current_camera
