import os
from os.path import join, dirname, abspath
import click
import copy
from datetime import datetime
import numpy as np
from numpy.linalg import inv
import open3d as o3d
from tqdm import tqdm
from PIL import Image
import torch
import cv2
import json
import yaml
import wandb

from metrics_3d.precision_recall import PrecisionRecall
from metrics_3d.chamfer_distance import ChamferDistance

from wild_completion.utils import get_render_data, get_time, clean_pcd, setup_wandb, set_random_seed
from wild_completion.mesher import MeshExtractor
from wild_completion.optimizer import Optimizer

from deepsdf.deep_sdf.workspace import config_decoder, load_latent_vectors

from opt_visualizer import OptVisualizer, color_table


@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'configs/lab_pepper.yaml'))

def main(config):

    set_random_seed(42)

    cfg = yaml.safe_load(open(config))    

    dev = cfg['device']
    dtype = torch.float32

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

    object_radius_max_m = float(cfg['vis']['object_radius_max_m'])
    mc_res_mm = float(cfg['vis']['mc_res_mm'])
    voxels_dim = int(2*object_radius_max_m*1e3/mc_res_mm)

    if cfg['baseline_name'] == 'DeepSDF':
        deepsdf_baseline = True
    else:
        deepsdf_baseline = False

    # initialization
    mesh_extractor = MeshExtractor(decoder, code_len=code_len, voxels_dim=voxels_dim, cube_radius=object_radius_max_m) # mc res: 0.2/40 ~ 5mm
    if cfg['vis']['vis_on']:
        vis = OptVisualizer(object_radius_max_m * 1.2, pause_time_s=cfg['vis']['vis_pause_s'])
    else:
        vis = None
    opt = Optimizer(cfg, decoder, mesh_extractor, vis)

    split_f = open(cfg['split'],'r')
    split = json.load(split_f)
    test_split=split['test']
    if cfg['fruit_id']!="none": # overwrite
        test_split = [cfg['fruit_id']]
    print(test_split)

    # metrics
    t_array = []  # record the optimization consuming time
    iter_array = [] # record the optimization iteration number
    cd_metric = ChamferDistance()
    pr_metric = PrecisionRecall(min_t=0.001, max_t=0.01, num=100)

    # better to calculate together
    for fruit_id in test_split:
        print("For fruit", fruit_id)

        input_base=os.path.join(cfg['data_dir'], fruit_id)
        rgbd_base=os.path.join(input_base, "realsense")
        rgb_folder=os.path.join(rgbd_base, "color")
        depth_folder=os.path.join(rgbd_base, "depth")
        mask_folder=os.path.join(rgbd_base, "masks")
        map_folder=os.path.join(rgbd_base, "scene")
        tf_folder=os.path.join(input_base, "tf")
        tf_file=os.path.join(tf_folder, "tf_allposes.npz")
        bbx_file=os.path.join(tf_folder, "bounding_box.npz")
        tfs=np.load(tf_file, allow_pickle=True)['arr_0']
        bbx=np.load(bbx_file, allow_pickle=True)['arr_0']
        min_bound, max_bound=bbx[0,:], bbx[1,:]
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound,max_bound)
        valid_frame_count=tfs.shape[0]
        print("Valid frame count:", valid_frame_count)

        mask_files = sorted(os.listdir(mask_folder)) # may have some mask imgs missing (so we use mask_files as the basis)
        mask_file_count = len(mask_files)
        sample_mask_file_idx = np.linspace(0, mask_file_count-1, min(mask_file_count,cfg["frame_per_fruit"])).astype(np.int32)

        map_pcd_file = os.path.join(map_folder, "integrated.ply")
        map_pcd = o3d.io.read_point_cloud(map_pcd_file)
        
        gt_mesh_folder=os.path.join(input_base, "laser")
        gt_pcd_file=os.path.join(gt_mesh_folder, "fruit.ply")
        gt_pcd=o3d.io.read_point_cloud(gt_pcd_file)
        gt_pcd.paint_uniform_color(np.ones(3)*0.8)
        gt_point_count=len(gt_pcd.points)
        
        intrinsic_json_path=os.path.join(rgbd_base,"intrinsic.json")
        with open(intrinsic_json_path) as json_file:
            cam_param = json.load(json_file)
            K_mat=np.array(cam_param["intrinsic_matrix"]).reshape(3,3).transpose()
            height=cam_param["height"]
            width=cam_param["width"]
            depth_scale=cam_param["depth_scale"]
            img_size=[height, width]
        print("Intrinsic matrix:")
        print(K_mat)
        invK = inv(K_mat)
        K_torch = torch.tensor(K_mat, device=dev, dtype=dtype)
        print("Image size:", img_size)
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.set_intrinsics(
            height=height,
            width=width,
            fx=K_mat[0,0],
            fy=K_mat[1,1],
            cx=K_mat[0,2],
            cy=K_mat[1,2],
        )
        cur_submap_id = 1 # only one target
        T_wm = tfs[0]
        T_mw = inv(T_wm)
        map_pcd = map_pcd.transform(T_wm)
        map_pcd = map_pcd.crop(bbox)

        original_point_count=len(map_pcd.points)
        down_point_count=cfg['opt']['recon']['n_pts']
        map_pcd = map_pcd.random_down_sample(sampling_ratio=min(down_point_count/original_point_count, 1.0))
        map_pcd = clean_pcd(map_pcd, cfg['opt']['recon']['cluster_dist_m'])
        bbox = map_pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()

        submap_id_imgs = {}
        depth_imgs = {}
        rgb_imgs = {}
        cam_poses = {} # T_wc

        frame_count = 0
        for idx in tqdm(sample_mask_file_idx):
            
            frame_count += 1
            mask_file_name = mask_files[idx]
            img_id_str = mask_file_name.split('.')[0]
            img_id = int(img_id_str)
            # print("Frame:", img_id)

            rgb_fname=os.path.join(rgb_folder, mask_file_name) # 00001.png
            depth_fname=os.path.join(depth_folder, mask_file_name.replace("png","npy")) # 00001.npy
            mask_fname=os.path.join(mask_folder, mask_file_name) # 00001.png
            
            bgr_img=cv2.imread(rgb_fname)
            rgb_img=cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            
            depth_img=np.load(depth_fname)
            depth_img_m = np.copy(depth_img)
            depth_img_m /= depth_scale

            mask_img=cv2.imread(mask_fname,cv2.IMREAD_GRAYSCALE)/255

            T_wc = tfs[img_id-1]
            T_cw = inv(T_wc)

            submap_id_imgs[img_id_str]=mask_img
            depth_imgs[img_id_str]=depth_img_m
            rgb_imgs[img_id_str]=rgb_img
            cam_poses[img_id_str]=T_wc

        render_data = get_render_data(cur_submap_id, submap_id_imgs, depth_imgs, cam_poses, img_size, invK, cfg, max_bbx_size=1000)

        # show one of the matched frames, for visualization only
        if cfg['vis']['vis_on']:
            mid_idx = int(render_data["count"]/2)
            frame_id = render_data["frame_id"][mid_idx]
            cur_pix_fg = render_data["pix_fg"][mid_idx]
            cur_pix_bg = render_data["pix_bg"][mid_idx]
            mask_img = submap_id_imgs[frame_id]   
            cur_fruit_mask = (mask_img==cur_submap_id)
            rgb_img = rgb_imgs[frame_id].astype(float)
            depth_img = depth_imgs[frame_id]
            rgb_img[~cur_fruit_mask] *= 0.4 # for visualization only (highlight masked part)
            rgb_img[depth_img==0] *= 0.7 # for visualization only (highlight the part with valid depth)
            # visualize the fg and bg samples
            if cfg['vis']['show_pix_sample']:
                rgb_img[cur_pix_fg[:,1], cur_pix_fg[:,0]] = np.array([0,0,255]) #fg samples
                rgb_img[cur_pix_bg[:,1], cur_pix_bg[:,0]] = np.array([255,0,0]) #bg samples 
            rgb_img = rgb_img.astype(np.uint8)
            rgb_img_show = Image.fromarray(rgb_img)
            rgb_img_show.show()
        
        if cfg['vis']['vis_on']:
            vis.add_scan(map_pcd)
            vis.add_gt_scan(gt_pcd)
            skip_flag = vis.stop()
            if skip_flag:
                vis.clean_vis()
                continue

        mean_color = np.mean(np.array(map_pcd.colors), axis=0) # use avaerge color of the point cloud
        cur_color = color_table[0] # use random color

        cur_pcd_w = copy.deepcopy(map_pcd)
        points_w_torch = torch.tensor(np.array(cur_pcd_w.points), device=dev, dtype=dtype)

        T_wo_torch = torch.eye(4, device=dev, dtype=dtype)
        # we would anyway give a translation initial guess according to the object bbx center
        T_wo_torch[:3,3] = torch.tensor(center, device=dev, dtype=dtype) 
        T_ow_torch = torch.inverse(T_wo_torch)
            
        latent = init_latent.clone().detach()
        t0 = get_time()
        # conduct the shape and pose joint optimization of the pepper
        if deepsdf_baseline:
            latent, _, iter_count = opt.shape_opt_deepsdf(latent, T_ow_torch, points_w_torch, mean_color)
        else: # ours
            latent, T_ow_torch, iter_count = opt.shape_pose_joint_opt(latent, T_ow_torch, render_data, points_w_torch, object_radius_max_m, cur_color)
        t1 = get_time()
        t_array.append(t1-t0)
        iter_array.append(iter_count)

        T_ow_cur = T_ow_torch.cpu().detach().numpy()
        T_wo = inv(T_ow_cur)

        # reconstruction with completion
        complete_mesh_o3d = mesh_extractor.complete_mesh(latent, T_wo, mean_color)
        complete_pcd = complete_mesh_o3d.sample_points_uniformly(gt_point_count)

        if cfg['vis']['vis_on']:
            vis.stop()
            vis.clean_vis()

        # define metrics
        cd_metric.update(gt_pcd,complete_pcd)
        pr_metric.update(gt_pcd,complete_pcd)

    pr_all, re_all, f1_all = pr_metric.compute_at_all_thresholds()
    pr, re, f1, thre = pr_metric.compute_at_threshold(0.005)
    cd = cd_metric.compute()
    t = np.mean(np.asarray(t_array)) # unit: s
    iter = np.mean(np.asarray(iter_array))
    count = len(t_array)
    
    precision = []
    recall = []
    fscore = []
    legend = []
    precision.append(pr_all)
    recall.append(re_all)
    fscore.append(f1_all)
    legend.append('Ours')

    if cfg['fruit_id']=="none":
        print("Results on the whole test set")
    else:
        print("Results on", cfg['fruit_id'])
    print("CD        [mm]:", cd*1e3)
    print("F-score    [%]:", f1)
    print("Precision  [%]:", pr)
    print("Recall:    [%]:", re)
    print("threshold [mm]:", thre)
    print("timing     [s]:", t)
    print("iteration     :", iter)
    print("calculated over %i frames" % count)

    if cfg['vis']['wandb_log_on']:
        setup_wandb()
        wandb.init(project="HOMA", config=cfg, dir=cfg['data_dir']) # your own worksapce
        wandb.run.name = cfg['run_name']+ datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
        wandb_log_content = {'CD[mm]': cd*1e3, 'F-score[%]': f1, 'Precision[%]': pr, 'Recall[%]': re, 'threshold[mm]': thre, 'timing[s]': t, 'iteration':iter, 'frames': count} 
        wandb.log(wandb_log_content)


if __name__ == "__main__":
    main()



