import os
import json
import open3d as o3d
import numpy as np
import cv2

# this dataloader is adapted from https://github.com/PRBonn/shape_completion_toolkit/blob/master/dataloader.py

class ShapeCompletionDataset():

    def __init__(self,
                 data_source=None,
                 split='train',
                 return_pcd = True,
                 return_rgbd = True,
                 ):

        assert return_pcd or return_rgbd, "return_pcd and return_rgbd are set to False. Set at least one to True"

        self.data_source = data_source
        self.split = split
        self.return_pcd = return_pcd
        self.return_rgbd = return_rgbd
        
        self.fruit_list = self.get_file_paths()

    def get_file_paths(self):
        fruit_list = {}
        for fid in os.listdir(os.path.join(self.data_source, self.split)):
            fruit_list[fid] = {
                'path': os.path.join(self.data_source, self.split, fid),
            }
        return fruit_list

    def get_gt(self, fid):        
        return o3d.io.read_point_cloud(os.path.join(self.fruit_list[fid]['path'],'gt/pcd/fruit.ply'))

    def get_rgbd(self, fid):
        fid_root = self.fruit_list[fid]['path']

        intrinsic_path = os.path.join(fid_root,'input/intrinsic.json')
        intrinsic = self.load_K(intrinsic_path)
        
        rgbd_data = {
            'intrinsic':intrinsic,
            'pcd': o3d.geometry.PointCloud(),
            'frames':{}
            }

        erosion_shape = cv2.MORPH_RECT # MORPH_RECT, MORPH_CROSS
        erosion_size = 5
        erosion_element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                (erosion_size, erosion_size))

        frames = os.listdir(os.path.join(fid_root,'input/masks/'))
        for frameid in frames:
            
            pose_path = os.path.join(fid_root,'input/poses/',frameid.replace('png','txt'))
            pose = np.loadtxt(pose_path)

            rgb_path = os.path.join(fid_root,'input/color/',frameid)
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            
            depth_path = os.path.join(fid_root,'input/depth/',frameid.replace('png','npy'))
            depth = np.load(depth_path)

            # filter the depth image, 1.5cm sigma, in 3 neighborhood 
            depth = cv2.bilateralFilter(depth,3,15,15) 
        
            # erosion for depth image
            depth = cv2.erode(depth, erosion_element)

            mask_path = os.path.join(fid_root,'input/masks/',frameid)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            frame_key = frameid.replace('.png','')

            # TODO: for greenhouse data
            # if self.split == 'test':
            #     rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
            #     mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            #     depth = np.rot90(depth, k=3)

            rgbd_data['frames'][frame_key] = {
                'rgb': rgb,
                'depth': depth,
                'mask': mask,
                'pose': pose,
                'fname': frame_key
            }

            if self.return_pcd:
                frame_pcd = self.rgbd_to_pcd(rgb, depth, mask, pose, intrinsic)
                rgbd_data['pcd'] += frame_pcd


        return rgbd_data

    @staticmethod
    def load_K(path):
        with open(path,'r') as f:
            data = json.load(f)['intrinsic_matrix']
        k = np.reshape(data, (3, 3), order='F') 
        return k

    @staticmethod
    def rgbd_to_pcd(rgb, depth, mask, pose, K):

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),
                                                                  o3d.geometry.Image(depth*mask),
                                                                  depth_scale = 1,
                                                                  depth_trunc=1.0,
                                                                  convert_rgb_to_intensity=False)

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(height=rgb.shape[0],
                                 width=rgb.shape[1],
                                 fx=K[0,0],
                                 fy=K[1,1],
                                 cx=K[0,2],
                                 cy=K[1,2],
                                 )

        extrinsic = np.linalg.inv(pose)
        
        frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)
        return frame_pcd


    def __len__(self):
        return len(self.fruit_list)

    def __getitem__(self, idx):
        
        keys = list(self.fruit_list.keys())
        fid = keys[idx]
        
        item = {}
        
        if self.split != 'test':
            gt_pcd = self.get_gt(fid)
            item['groundtruth_pcd'] = gt_pcd
        
        input_data = self.get_rgbd(fid)
        if self.return_pcd:
            item['rgbd_pcd'] = input_data['pcd']
        if self.return_rgbd:
            item['rgbd_intrinsic'] = input_data['intrinsic']
            item['rgbd_frames'] = input_data['frames']
        
        item['fid'] = fid # Yue_added

        return item
    