import os
from os.path import join, dirname, abspath
import click
import copy
from datetime import datetime
import numpy as np
from numpy.linalg import inv, det, norm
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

from wild_completion.utils import get_render_data, get_time, clean_pcd, setup_wandb,set_random_seed,get_deg_between_vectors
from wild_completion.mesher import MeshExtractor
from wild_completion.optimizer import Optimizer

from deepsdf.deep_sdf.workspace import config_decoder, load_latent_vectors

from opt_visualizer import OptVisualizer, color_table


@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'configs/cka_pepper.yaml'))

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

    # initialization
    mesh_extractor = MeshExtractor(decoder, code_len=code_len, voxels_dim=voxels_dim, cube_radius=object_radius_max_m) # mc res: 0.2/40 ~ 5mm
    if cfg['vis']['vis_on']:
        vis = OptVisualizer(object_radius_max_m * 1.2, pause_time_s=cfg['vis']['vis_pause_s'])
    else:
        vis = None
    opt = Optimizer(cfg, decoder, mesh_extractor, vis)

    # metrics
    cd_metric = ChamferDistance()
    pr_metric = PrecisionRecall(min_t=0.001, max_t=0.01, num=100)
    t_array = []  # record the optimization consuming time
    iter_array = [] # record the optimization iteration number
    tran_error_array = []
    rot_error_array = []

    for data_dir in cfg['data_dir']:

        input_base=os.path.join(data_dir, "before")
        ros_tf_file=os.path.join(input_base, "rostf_poses_no_jump.npz")
        ros_tfs=np.load(ros_tf_file, allow_pickle=True)['arr_0']

        rgbd_base=os.path.join(input_base, "realsense")
        rgb_folder=os.path.join(rgbd_base, "color")
        depth_folder=os.path.join(rgbd_base, "depth")
        mask_folder=os.path.join(rgbd_base, "masks")
        submap_id_folder=os.path.join(rgbd_base, "submap_ids")

        # load intrinsic
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
        T_cw = np.array([[0,0,-1,0],[-1,0,0,0],[0,1,0,0],[0,0,0,1]]) # T_cw, extrinsic initial guess
        T_wc = inv(T_cw)

        gt_measure_base=os.path.join(data_dir, "fruits_measured")
        if cfg['useable_only']:
            gt_info_json=os.path.join(gt_measure_base, "info_usable.json")
        else:
            gt_info_json=os.path.join(gt_measure_base, "info.json")

        with open(gt_info_json) as json_file:
            gt_fruits_info = json.load(json_file)
            gt_fruits_list = list(gt_fruits_info.keys())

        for fruit_id in gt_fruits_list:
        
            fruit_info = gt_fruits_info[fruit_id]
            cur_submap_id = fruit_info["submap_id"]
            begin_frame = fruit_info["begin_frame"]
            end_frame = fruit_info["end_frame"]

            print("For fruit", fruit_id, " (Submap ",cur_submap_id, ")")

            fruit_measure_base = os.path.join(gt_measure_base, fruit_id)
            tf_folder=os.path.join(fruit_measure_base, "tf")
            tf_cam_file=os.path.join(tf_folder, "tf_allposes.npz") # to each camera frame
            tfs_cam=np.load(tf_cam_file, allow_pickle=True)['arr_0']
            tf_meta_file=os.path.join(tf_folder, "tf.npz") # to the metashape reconstruction of the before sequence's frame
            tf_meta=np.load(tf_meta_file, allow_pickle=True)['arr_0']
            valid_tf_frame_count=tfs_cam.shape[0]

            fruit_result_base = os.path.join(fruit_measure_base, "result_"+cfg["run_name"])
            if not os.path.exists(fruit_result_base):
                os.makedirs(fruit_result_base)

            # load gt cloud
            gt_mesh_folder=os.path.join(fruit_measure_base, "laser")
            gt_pcd_file=os.path.join(gt_mesh_folder, "fruit_clean.ply") # with no stick and pins
            gt_pcd=o3d.io.read_point_cloud(gt_pcd_file)
            gt_pcd = gt_pcd.voxel_down_sample(voxel_size=1e-3)
            gt_point_count=len(gt_pcd.points)

            rgb_files = sorted(os.listdir(rgb_folder))

            sample_frame_idx = np.linspace(begin_frame, end_frame-1, min(end_frame-begin_frame+1,cfg["frame_per_fruit"])).astype(np.int32)

            frame_count = 0
            for img_id in tqdm(sample_frame_idx):
                print("Frame:", img_id)

                frame_count += 1
                rgb_file_name = rgb_files[img_id]
                img_id_str = rgb_file_name.split('.')[0]
                
                rgb_fname=os.path.join(rgb_folder, rgb_file_name) # 00001.png
                depth_fname=os.path.join(depth_folder, img_id_str+".npy") # 00001.npy
                submap_id_fname=os.path.join(submap_id_folder, img_id_str+"_submap_id.png") # 00001_submap_id.png

                if not os.path.exists(submap_id_fname):
                    print("No such submap id file for this frame")
                    continue
                
                bgr_img=cv2.imread(rgb_fname)
                rgb_img=cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                
                depth_img=np.load(depth_fname)
                depth_img_m=depth_img / depth_scale

                submap_id_img=cv2.imread(submap_id_fname,cv2.IMREAD_GRAYSCALE)
                submap_id_img[submap_id_img!=cur_submap_id] = 0

                # masked 
                depth_img_masked=np.copy(depth_img)
                depth_img_masked[submap_id_img==0] = 0.

                rgb_img_o3d = o3d.geometry.Image(rgb_img)
                depth_img_o3d = o3d.geometry.Image(depth_img_masked)
                rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, depth_img_o3d, \
                            depth_scale=depth_scale, depth_trunc=1.0, convert_rgb_to_intensity=False)

                T_gc = tfs_cam[img_id] # T_gc
                T_cg = inv(T_gc)
                T_wg = T_wc @ T_cg

                rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, intrinsic_o3d, \
                            T_cw, project_valid_depth_only=True)
                original_point_count=len(rgbd_pcd.points)
                down_point_count=cfg['opt']['recon']['n_pts']
                if original_point_count < 0.2 * down_point_count: 
                    print("Too few 3d points, skip")
                    continue
                rgbd_pcd = rgbd_pcd.random_down_sample(sampling_ratio=min(down_point_count/original_point_count, 1.0))
                rgbd_pcd = clean_pcd(rgbd_pcd, cfg['opt']['recon']['cluster_dist_m'])
                bbox = rgbd_pcd.get_axis_aligned_bounding_box()
                center = bbox.get_center()

                submap_id_imgs = {img_id_str: submap_id_img}
                depth_imgs = {img_id_str: depth_img_m}
                cam_poses = {img_id_str: T_wc} # T_wc

                render_data = get_render_data(cur_submap_id, submap_id_imgs, depth_imgs, cam_poses, img_size, invK, cfg, max_bbx_size=400)

                # show one of the matched frames, for visualization only
                if cfg['vis']['vis_on']:
                    cur_pix_fg = render_data["pix_fg"][0]
                    cur_pix_bg = render_data["pix_bg"][0]    
                    cur_fruit_mask = (submap_id_img==cur_submap_id)
                    cur_rgb_img_clone = np.copy(rgb_img).astype(float)
                    cur_rgb_img_clone[~cur_fruit_mask] *= 0.4 # for visualization only (highlight masked part)
                    cur_rgb_img_clone[depth_img==0] *= 0.8 # for visualization only (highlight the part with valid depth)
                    # visualize the fg and bg samples
                    if cfg['vis']['show_pix_sample']:
                        cur_rgb_img_clone[cur_pix_fg[:,1], cur_pix_fg[:,0]] = np.array([0,0,255]) #fg samples
                        cur_rgb_img_clone[cur_pix_bg[:,1], cur_pix_bg[:,0]] = np.array([255,0,0]) #bg samples 
                    cur_rgb_img_clone = cur_rgb_img_clone.astype(np.uint8)
                    cur_rgb_img_show = Image.fromarray(cur_rgb_img_clone)
                    if cfg['vis']['rot_img']:
                        cur_rgb_img_show = cur_rgb_img_show.rotate(-90, expand=True)
                    cur_rgb_img_show.show()
                
                gt_pcd_clone = copy.deepcopy(gt_pcd)
                gt_pcd_w = gt_pcd_clone.transform(T_wg) # to the so-called world frame
                gt_pcd_w.paint_uniform_color(np.ones(3)*0.8)
                
                if cfg['vis']['vis_on']:
                    vis.add_scan(rgbd_pcd)
                    skip_flag = vis.stop()
                    if skip_flag:
                        vis.clean_vis()
                        continue

                mean_color = np.mean(np.array(rgbd_pcd.colors), axis=0) # use avaerge color of the point cloud
                cur_color = color_table[0] # use random color

                cur_pcd_w = copy.deepcopy(rgbd_pcd)
                points_w_torch = torch.tensor(np.array(cur_pcd_w.points), device=dev, dtype=dtype)

                T_wo_torch = torch.eye(4, device=dev, dtype=dtype)
                # we would anyway give a translation initial guess according to the object bbx center
                T_wo_torch[:3,3] = torch.tensor(center, device=dev, dtype=dtype) 
                T_ow_torch = torch.inverse(T_wo_torch)
                    
                latent = init_latent.clone().detach()
                t0 = get_time()
                # conduct the shape and pose joint optimization of the pepper
                if cfg['baseline_name'] == 'DeepSDF':
                    latent, _, iter_count = opt.shape_opt_deepsdf(latent, T_ow_torch, points_w_torch, cur_color)
                else: # our method
                    latent, T_ow_torch, iter_count = opt.shape_pose_joint_opt(latent, T_ow_torch, render_data, points_w_torch, object_radius_max_m, mean_color)
                t1 = get_time()
                t_array.append(t1-t0)
                iter_array.append(iter_count)

                T_ow_cur = T_ow_torch.cpu().detach().numpy()
                T_wo = inv(T_ow_cur)

                # reconstruction with completion
                complete_mesh_o3d = mesh_extractor.complete_mesh(latent, T_wo, mean_color)
                complete_pcd = complete_mesh_o3d.sample_points_uniformly(gt_point_count)

                complete_mesh_path = os.path.join(fruit_result_base, "complete_mesh.ply")
                o3d.io.write_triangle_mesh(complete_mesh_path, complete_mesh_o3d)

                if cfg['vis']['vis_on']:
                    vis.add_gt_scan(gt_pcd_w)
                    vis.stop()
                    vis.clean_vis()

                # pose metrics
                final_scale = det(T_wo[:3,:3])**(1/3)
                T_wo_descale = T_wo
                T_wo_descale[:3,:3] /= final_scale

                gt_pcd_file = os.path.join(fruit_result_base, "gt_pcd.ply")
                o3d.io.write_point_cloud(gt_pcd_file, gt_pcd_w)

                estimated_pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                estimated_pose_frame.transform(T_wo_descale)
                estimated_pose_frame_file = os.path.join(fruit_result_base, "estimated_pose.ply")
                o3d.io.write_triangle_mesh(estimated_pose_frame_file, estimated_pose_frame)

                gt_pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                gt_pose_frame.transform(T_wg)
                gt_pose_frame_file = os.path.join(fruit_result_base, "gt_pose.ply")
                o3d.io.write_triangle_mesh(gt_pose_frame_file, gt_pose_frame)
                
                translation_error_vector = T_wg[:3,3] - T_wo[:3,3]
                tran_error = norm(translation_error_vector)*1e3 # in mm
                # print("E_tran (mm):")
                # print(tran_error)
                tran_error_array.append(tran_error)

                rot_error = get_deg_between_vectors(T_wo_descale[:3,2], T_wg[:3,2])
                # print("E_rot (deg):")
                # print(rot_error)
                rot_error_array.append(rot_error)

                # define metrics
                cd_metric.update(gt_pcd_w,complete_pcd)
                pr_metric.update(gt_pcd_w,complete_pcd)


    pr_all, re_all, f1_all = pr_metric.compute_at_all_thresholds()
    pr, re, f1, thre = pr_metric.compute_at_threshold(0.005)
    cd = cd_metric.compute()
    t = np.mean(np.asarray(t_array)) # unit: s
    iter = np.mean(np.asarray(iter_array))
    count = len(t_array)
    tran_error_array = np.asarray(tran_error_array)
    tran_error = np.mean(tran_error_array)
    tran_std = np.std(tran_error_array)
    rot_error_array = np.asarray(rot_error_array)
    rot_error = np.mean(rot_error_array)
    rot_std = np.std(rot_error_array)
    
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
    print("TransError[mm]:", tran_error)
    print("TransStd  [mm]:", tran_std)
    print("RotError [deg]:", rot_error)
    print("RotStd   [deg]:", rot_std)
    print("threshold [mm]:", thre)
    print("timing     [s]:", t)
    print("iteration     :", iter)
    print("calculated over %i frames" % count)

    if cfg['vis']['wandb_log_on']:
        setup_wandb()
        wandb.init(project="HOMA", config=cfg, dir=data_dir) # your own worksapce
        wandb.run.name = cfg['run_name']+ datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
        wandb_log_content = {'CD[mm]': cd*1e3, 'F-score[%]': f1, 'Precision[%]': pr, 'Recall[%]': re, 'threshold[mm]': thre, 'Error_trans[mm]': tran_error, 'Error_rot[deg]': rot_error, 'timing[s]': t, 'iteration':iter, 'frames': count} 
        wandb.log(wandb_log_content)

if __name__ == "__main__":
    main()




