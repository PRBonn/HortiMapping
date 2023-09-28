import os
import click
import copy
from os.path import join, dirname, abspath
import numpy as np
from numpy.linalg import inv, det
import open3d as o3d
from tqdm import tqdm
from skimage import io
from scipy.spatial.transform import Rotation
from PIL import Image
import torch
import yaml

from wild_completion.utils import clean_mesh, get_pose_init, get_render_data, set_random_seed, axis_angle_to_rotation_matrix
from wild_completion.mesher import MeshExtractor
from wild_completion.optimizer import Optimizer

from deepsdf.deep_sdf.workspace import config_decoder, load_latent_vectors

from wild_completion.opt_visualizer import OptVisualizer, color_table

@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'configs/wild_pepper.yaml'))

def main(config):

    set_random_seed(42)

    cfg = yaml.safe_load(open(config))

    dev = cfg['device']
    dtype = torch.float32

    # model folder
    DeepSDF_DIR = cfg['deepsdf_dir']
    checkpoint = "latest"

    # load deep sdf decoder and init latent code
    decoder = config_decoder(DeepSDF_DIR, checkpoint)
    decoder.cuda()
    latents_train = load_latent_vectors(DeepSDF_DIR, checkpoint).to(dev)
    init_latent = torch.mean(latents_train, 0) # the mean latent code for training data
    # init_latent = torch.zeros_like(init_latent) # or use the zero code initializaition
    code_len = init_latent.shape[0]
    print("DeepSDF model loaded")
    print("Init average latent code:")
    print(init_latent)

    # data folder
    data_base = cfg['data_dir']
    submap_folder = os.path.join(data_base, "submaps")
    complete_submap_folder = submap_folder+"_complete"
    if not os.path.exists(complete_submap_folder):
        os.makedirs(complete_submap_folder)
    clean_submap_folder = submap_folder+"_clean"
    if not os.path.exists(clean_submap_folder):
        os.makedirs(clean_submap_folder)
    pose_folder = submap_folder+"_pose"
    if not os.path.exists(pose_folder):
        os.makedirs(pose_folder)

    estimate_scale = cfg['opt']['scale_on']
    object_radius_max_m = float(cfg['vis']['object_radius_max_m'])
    mc_res_mm = float(cfg['vis']['mc_res_mm'])
    voxels_dim = int(2*object_radius_max_m*1e3/mc_res_mm)

    cam_info_path = cfg["cam_info_path"]
    with open(cam_info_path, "r") as stream:
        cam_param = yaml.safe_load(stream)
        K = np.array(cam_param['intrinsics'])
        extrinsics = np.array(cam_param['extrinsics'])
        img_size = cam_param['img_size']
    print("intrinsic matrix:")
    print(K)
    invK = inv(K)
    K_torch = torch.tensor(K, device=dev, dtype=dtype)

    print("Image size:", img_size)

    # load images and poses
    submap_id_imgs = {}
    depth_imgs = {}
    rgb_imgs = {} # only for visualization, not used actually
    cam_poses = {}
    frame_count = 0
    # provide the rgb (only for vis), depth, submap_id (instance seg) and cam_pose file in pairs
    for fname in tqdm(sorted(os.listdir(data_base))):
        if 'id' not in fname:
            continue
        
        if (frame_count < cfg["begin_frame"] or frame_count > cfg["end_frame"] or \
            frame_count % cfg["every_frame"] != 0): 
            frame_count += 1
            continue

        submap_id_img_path = os.path.join(data_base, fname)
        # load images
        submap_id_img = io.imread(submap_id_img_path)
        depth_img = io.imread(submap_id_img_path.replace("submap_id.png", "depth.tiff")).astype(float)
        rgb_img = io.imread(submap_id_img_path.replace("submap_id.png", "color.png")).astype(float)
        # load pose
        pose_file_path = submap_id_img_path.replace("submap_id.png", "pose.txt")
        if os.path.isfile(pose_file_path):
                pose_data = [float(x) for x in open(pose_file_path, 'r').read().split()]
                T_wc = np.eye(4)
                for row in range(4):
                    for col in range(4):
                        T_wc[row, col] = pose_data[row * 4 + col]
        # assign to dictionary
        frame_id = fname.split("_")[0]
        submap_id_imgs[frame_id] = submap_id_img
        depth_imgs[frame_id] = depth_img
        rgb_imgs[frame_id] = rgb_img
        cam_poses[frame_id] = T_wc

        # print(frame_id, "loaded")
        frame_count += 1
    
    # initialization
    mesh_extractor = MeshExtractor(decoder, code_len=code_len, voxels_dim=voxels_dim, cube_radius=object_radius_max_m) # mc res: 0.2/40 ~ 5mm
    if cfg['vis']['vis_on']:
        vis = OptVisualizer(pause_time_s=cfg['vis']['vis_pause_s'])
    else:
        vis = None
    opt = Optimizer(cfg, decoder, mesh_extractor, vis)

    # For each sweetpepper submap
    for submap_name in tqdm(sorted(os.listdir(submap_folder))):
        
        submap_cat = (submap_name.split("_")[1]).split(".")[0]
        submap_id = int(submap_name.split("_")[0])

        if submap_id > 1 and submap_id < cfg["begin_submap"]:
            continue

        print("Submap:", submap_id)

        # load the submap mesh
        submap_path = os.path.join(submap_folder, submap_name)
        cur_mesh = o3d.io.read_triangle_mesh(submap_path)
        cur_mesh.compute_vertex_normals()

        if submap_cat == "Background":
            bg_pcd = cur_mesh.sample_points_uniformly(number_of_points=500000)
            bg_pcd = bg_pcd.voxel_down_sample(voxel_size=0.005)
            continue

        # get rendering data for the submap
        render_data = get_render_data(submap_id, submap_id_imgs, depth_imgs, cam_poses, img_size, invK, cfg)
        if render_data["count"] == 0:
            print("No valid match, skip this submap")
            continue

        # show one of the matched frames, for visualization only
        if cfg['vis']['vis_on']:
            mid_frame_idx = int(len(render_data["frame_id"])/2)
            mid_frame_id = render_data["frame_id"][mid_frame_idx]
            cur_pix_fg = render_data["pix_fg"][mid_frame_idx]
            cur_pix_bg = render_data["pix_bg"][mid_frame_idx]
            cur_T_wc_torch = render_data["T_wc"][mid_frame_idx]
            
            mid_frame_depth_img = depth_imgs[mid_frame_id]
            mid_frame_rgb_img = rgb_imgs[mid_frame_id]
            mid_submap_id_img = submap_id_imgs[mid_frame_id]
            cur_fruit_mask = (mid_submap_id_img==submap_id)

            cur_rgb_img_clone = np.copy(mid_frame_rgb_img)
            cur_rgb_img_clone[~cur_fruit_mask] *= 0.3 # for visualization only (highlight masked part)
            # cur_rgb_img_clone[mid_frame_depth_img==0] *= 0.8 # for visualization only (highlight the part with valid depth)
            # visualize the fg and bg samples
            if cfg['vis']['show_pix_sample']:
                cur_rgb_img_clone[cur_pix_fg[:,1], cur_pix_fg[:,0]] = np.array([0,0,255]) #fg samples
                cur_rgb_img_clone[cur_pix_bg[:,1], cur_pix_bg[:,0]] = np.array([255,0,0]) #bg samples
            cur_rgb_img_clone = cur_rgb_img_clone.astype(np.uint8)
            cur_rgb_img_show = Image.fromarray(cur_rgb_img_clone)
            if cfg['vis']['rot_img']:
                cur_rgb_img_show = cur_rgb_img_show.rotate(-90, expand=True)
            cur_rgb_img_show.show()
            vis.clean_vis()

        # Process the submap
        # clean the submap and sample point cloud from it
        cur_pcd_world = clean_mesh(cur_mesh, cfg['opt']['recon']['n_pts'], cfg['opt']['recon']['cluster_dist_m'])

        # get the initial guess of pose
        cur_center, init_rot_y_rad, cur_bbx_size, valid_flag = get_pose_init(cur_pcd_world, bg_pcd)

        if not valid_flag:
            continue

        T_wo_torch = torch.eye(4, device=dev, dtype=dtype)
        # we would anyway give a translation initial guess according to the object bbx center
        T_wo_torch[:3,3] = torch.tensor(cur_center, device=dev, dtype=dtype) 
        if not cfg['opt']['pose_init']['rot_on']: # no rotation initial guess
            init_rot_y_rad = 0.
        axis_angle_init = torch.tensor([0, init_rot_y_rad, 0], device=dev, dtype=dtype)
        object_radius_m = object_radius_max_m*0.8
        if cfg['opt']['pose_init']['scale_on']:
            scale_init = max(cur_bbx_size / (2*object_radius_m), 0.5) # also apply the scale inital guess
        else: # no scale initial guess
            scale_init = 1.
        print("Init scale", scale_init)
        T_wo_torch[:3, :3] = axis_angle_to_rotation_matrix(axis_angle_init) * scale_init 
        T_ow_torch = torch.inverse(T_wo_torch)

        mean_color = np.mean(np.array(cur_pcd_world.colors), axis=0) # use avaerge color of the point cloud
        cur_color = color_table[submap_id%10] # use random color

        cur_pcd_w = copy.deepcopy(cur_pcd_world)
        points_w_torch = torch.tensor(np.array(cur_pcd_w.points), device=dev, dtype=dtype)

        if cfg['vis']['vis_on']:
            vis.add_scan(cur_pcd_world)
            skip_flag = vis.stop()
            if skip_flag:
                vis.clean_vis()
                continue
        
        latent = init_latent.clone().detach()
        # conduct the shape and pose joint optimization of the pepper
        latent, T_ow_torch, _ = opt.shape_pose_joint_opt(latent, T_ow_torch, render_data, points_w_torch, object_radius_max_m, cur_color)

        T_ow_cur = T_ow_torch.cpu().detach().numpy()
        T_wo = inv(T_ow_cur)

        final_scale = det(T_wo[:3,:3])**(1/3)
        rot_mat = T_wo[:3,:3]/final_scale
        rot = Rotation.from_matrix(rot_mat)
        euler = rot.as_euler('zyx', degrees=True)
        yaw, pitch, roll = euler[0], euler[1], euler[2]
        # print(yaw, pitch, roll)

        if final_scale < cfg['opt']['outlier']['scale_min'] or final_scale > cfg['opt']['outlier']['scale_max']:
            print("The final scale %f is a outlier, not valid" %final_scale)
            continue
        if abs(pitch) > cfg['opt']['outlier']['rot_max_deg']:
            print("The final pitch rotation %f is a outlier, not valid" %pitch)
            continue
        if abs(roll) > cfg['opt']['outlier']['rot_max_deg']:
            print("The final roll rotation %f is a outlier, not valid" %roll)
            continue    

        # reconstruction with completion
        complete_mesh_o3d = mesh_extractor.complete_mesh(latent, T_wo, mean_color)
        complete_mesh_path = os.path.join(complete_submap_folder, submap_name)
        o3d.io.write_triangle_mesh(complete_mesh_path, complete_mesh_o3d)
        print("save the complete mesh to %s\n" % (complete_mesh_path))

        clean_pcd_path = os.path.join(clean_submap_folder, submap_name)
        o3d.io.write_point_cloud(clean_pcd_path, cur_pcd_world)
        print("save the clean point cloud to %s\n" % (clean_pcd_path))

        pose_out_path = os.path.join(pose_folder, submap_name.replace("ply", "npy"))
        np.save(pose_out_path, T_wo)
        print("save the submap pose file to %s\n" % (pose_out_path))

        if cfg['vis']['vis_on']:
            vis.stop()
            vis.clean_vis()
        
if __name__ == "__main__":
    main()

