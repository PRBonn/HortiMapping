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

from wild_completion.utils import get_render_data, get_time, clean_pcd, setup_wandb, set_random_seed, clean_mesh, get_pose_init, get_deg_between_vectors, axis_angle_to_rotation_matrix
from wild_completion.mesher import MeshExtractor
from wild_completion.optimizer import Optimizer

from deepsdf.deep_sdf.workspace import config_decoder, load_latent_vectors

from wild_completion.opt_visualizer import OptVisualizer, color_table


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

    if cfg['baseline_name'] == 'DeepSDF':
        deepsdf_baseline = True
    else:
        deepsdf_baseline = False

    object_radius_max_m = float(cfg['vis']['object_radius_max_m'])
    mc_res_mm = float(cfg['vis']['mc_res_mm'])
    voxels_dim = int(2*object_radius_max_m*1e3/mc_res_mm)

    # initialization
    mesh_extractor = MeshExtractor(decoder, code_len=code_len, voxels_dim=voxels_dim, cube_radius=object_radius_max_m) # mc res: 0.2/40 ~ 5mm
    if cfg['vis']['vis_on']:
        vis = OptVisualizer(object_radius_max_m * 1.2)
    else:
        vis = None
    opt = Optimizer(cfg, decoder, mesh_extractor, vis)

    # metrics
    cd_metric = ChamferDistance()
    pr_metric = PrecisionRecall(min_t=0.001, max_t=0.01, num=100)
    tran_error_array = []
    rot_error_array = []
    t_array = []  # record the optimization consuming time
    iter_array = [] # record the optimization iteration number


    # dirty fix, add it to some config file (TODO)
    T_bc = np.array([[0.,-1.,0.,1.85999882],
                     [0.,0.,1.,-0.23719681],
                     [-1.,0.,0.,2.02642561],
                     [0.,0.,0.,1.]])

    for data_dir in cfg['data_dir']:
        
        print("Process", data_dir)

        input_base=os.path.join(data_dir, "before")
        ros_tf_file=os.path.join(input_base, "rostf_poses_no_jump.npz")
        ros_tfs=np.load(ros_tf_file, allow_pickle=True)['arr_0'] # T_bw
        
        cam_tf_file=os.path.join(input_base, "rostf_poses_metashape_aligned.npz") 
        cam_tfs=np.load(cam_tf_file, allow_pickle=True)['arr_0'] # T_wc
        
        submap_folder=os.path.join(input_base, "submaps")

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

        gt_measure_base=os.path.join(data_dir, "fruits_measured")
        if cfg['useable_only']:
            gt_info_json=os.path.join(gt_measure_base, "info_usable.json")
        else:
            gt_info_json=os.path.join(gt_measure_base, "info.json")
        with open(gt_info_json) as json_file:
            gt_fruits_info = json.load(json_file)
            gt_fruits_list = list(gt_fruits_info.keys())

        metashape_base = os.path.join(input_base, "metashape")
        metashape_pose_file = os.path.join(metashape_base, "scaled_poses.npz")
        metashape_poses = np.load(metashape_pose_file, allow_pickle=True)['arr_0'] 
        # print(metashape_poses)

        T_wm = (inv(ros_tfs[0]) @ T_bc) @ inv(metashape_poses[0]) # from metashape to world
        T_mw = inv(T_wm) # from world to metashape

        # load bg map
        bg_map_path = os.path.join(submap_folder, "00001_Background.ply")
        bg_mesh = o3d.io.read_triangle_mesh(bg_map_path)
        bg_mesh.compute_vertex_normals()
        bg_pcd = bg_mesh.sample_points_uniformly(number_of_points=500000)
        bg_pcd = bg_pcd.voxel_down_sample(voxel_size=0.01)

        # for each gt fruit
        for fruit_id in gt_fruits_list:
        
            fruit_info = gt_fruits_info[fruit_id]
            cur_submap_id = fruit_info["submap_id"]
            begin_frame = fruit_info["begin_frame"]
            end_frame = fruit_info["end_frame"]

            print("For fruit", fruit_id, " (Submap ",cur_submap_id, ")")

            fruit_measure_base = os.path.join(gt_measure_base, fruit_id)
            tf_folder=os.path.join(fruit_measure_base, "tf")
            bbx_file=os.path.join(tf_folder, "bounding_box.npz")
            bbx=np.load(bbx_file, allow_pickle=True)['arr_0']  
            min_bound, max_bound=bbx[0,:], bbx[1,:]
            bbox_g = o3d.geometry.AxisAlignedBoundingBox(min_bound,max_bound)
            # print(bbx)

            tf_cam_file=os.path.join(tf_folder, "tf_allposes.npz")
            tfs_cam=np.load(tf_cam_file, allow_pickle=True)['arr_0'] # from each camera frame to the gt fruit frame (T_gc)

            tf_meta_file=os.path.join(tf_folder, "tf.npz") # to the metashape reconstruction of the before sequence's frame
            T_mg=np.load(tf_meta_file, allow_pickle=True)['arr_0']
            T_wg = T_wm @ T_mg

            fruit_result_base = os.path.join(fruit_measure_base, "result_"+cfg["run_name"])
            if not os.path.exists(fruit_result_base):
                os.makedirs(fruit_result_base)

            # load gt cloud
            gt_mesh_folder=os.path.join(fruit_measure_base, "laser")
            gt_pcd_file=os.path.join(gt_mesh_folder, "fruit_clean.ply")
            gt_pcd=o3d.io.read_point_cloud(gt_pcd_file)
            gt_pcd=gt_pcd.voxel_down_sample(voxel_size=1e-3)
            gt_point_count=len(gt_pcd.points)
            
            # load offline photometric reconstruction map for the fruit (used as the upper bound of performance)
            meta_recon_file=os.path.join(fruit_measure_base, "reconstruction.ply")
            meta_recon_pcd=o3d.io.read_point_cloud(meta_recon_file)
            meta_recon_pcd=meta_recon_pcd.transform(inv(T_mg)) # to gt fruit frame
            meta_recon_pcd=meta_recon_pcd.crop(bbox_g) # crop it
            meta_recon_pcd=meta_recon_pcd.transform(T_mg) # back to metashape frame
            meta_recon_pcd=meta_recon_pcd.transform(T_wm) # to world frame

            if not cfg['use_homa']: # offline photometric map would be used
                submap_pcd_world = meta_recon_pcd
                original_point_count=len(submap_pcd_world.points)
                down_point_count=cfg['opt']['recon']['n_pts']
                submap_pcd_world = submap_pcd_world.random_down_sample(sampling_ratio=min(down_point_count/original_point_count, 1.0))
                submap_pcd_world = clean_pcd(submap_pcd_world, cfg['opt']['recon']['cluster_dist_m'])
                bbox = submap_pcd_world.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                T_wo_torch = torch.eye(4, device=dev, dtype=dtype)
                T_wo_torch[:3,3] = torch.tensor(center, device=dev, dtype=dtype) # we would anyway give a translation initial guess according to the object bbx center
                T_ow_torch = torch.inverse(T_wo_torch)
            
            else: # we will load the corresponding submap and process
                submap_filename = ("%05i"%cur_submap_id) + "_Sweetpepper.ply"
                submap_path = os.path.join(submap_folder, submap_filename)
                submap_mesh = o3d.io.read_triangle_mesh(submap_path)
                submap_mesh.compute_vertex_normals()

                # Process the submap
                # clean the submap and sample point cloud from it
                submap_pcd_world = clean_mesh(submap_mesh, cfg['opt']['recon']['n_pts'], cfg['opt']['recon']['cluster_dist_m'])

                # get the initial guess of pose
                cur_center, init_rot_y_rad, cur_bbx_size, valid_flag = get_pose_init(submap_pcd_world, bg_pcd)

                if not valid_flag:
                    continue

                T_wo_torch = torch.eye(4, device=dev, dtype=dtype)
                # we would anyway give a translation initial guess according to the object bbx center
                T_wo_torch[:3,3] = torch.tensor(cur_center, device=dev, dtype=dtype) 
                if not cfg['opt']['pose_init']['rot_on'] or deepsdf_baseline: # no rotation initial guess
                    init_rot_y_rad = 0.
                axis_angle_init = torch.tensor([0, init_rot_y_rad, 0], device=dev, dtype=dtype)
                object_radius_m = object_radius_max_m*0.8
                if not cfg['opt']['pose_init']['scale_on'] or deepsdf_baseline:
                    scale_init = 1. # no scale initial guess
                else: 
                    scale_init = max(cur_bbx_size / (2*object_radius_m), 0.5) # also apply the scale inital guess
                print("Init scale", scale_init)
                T_wo_torch[:3,:3] = axis_angle_to_rotation_matrix(axis_angle_init) * scale_init 
                T_ow_torch = torch.inverse(T_wo_torch)

            rgb_files = sorted(os.listdir(rgb_folder))

            sample_frame_idx = np.linspace(begin_frame, end_frame-1, min(end_frame-begin_frame+1,cfg["frame_per_fruit"])).astype(np.int32)

            submap_id_imgs = {}
            depth_imgs = {}
            rgb_imgs = {}
            cam_poses = {} # T_wc

            frame_count = 0
            for img_id in tqdm(sample_frame_idx):
                # print("Frame:", img_id)

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

                # T_wb = inv(ros_tfs[img_id])
                # T_wc = T_wb @ T_bc

                T_wc = cam_tfs[img_id]
            
                submap_id_imgs[img_id_str]=submap_id_img
                depth_imgs[img_id_str]=depth_img_m
                rgb_imgs[img_id_str]=rgb_img
                cam_poses[img_id_str]=T_wc

            render_data = get_render_data(cur_submap_id, submap_id_imgs, depth_imgs, cam_poses, img_size, invK, cfg, max_bbx_size=400)

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
                if cfg['vis']['rot_img']:
                    rgb_img_show = rgb_img_show.rotate(-90, expand=True)
                rgb_img_show.show()

            if cfg['vis']['vis_on']:
                vis.add_scan(submap_pcd_world)
                # vis.add_gt_scan(gt_pcd)
                skip_flag = vis.stop()
                if skip_flag:
                    vis.clean_vis()
                    continue

            mean_color = np.mean(np.array(submap_pcd_world.colors), axis=0) # use avaerge color of the point cloud
            cur_color = color_table[int(fruit_id)%10] # use random color
            # cur_color = mean_color

            cur_pcd_w = copy.deepcopy(submap_pcd_world)
            points_w_torch = torch.tensor(np.array(cur_pcd_w.points), device=dev, dtype=dtype)
                
            latent = init_latent.clone().detach()
            t0 = get_time()
            # conduct the shape and pose joint optimization of the pepper
            if deepsdf_baseline: 
                latent, _, iter_count = opt.shape_opt_deepsdf(latent, T_ow_torch, points_w_torch, cur_color)
            else: # our method
                latent, T_ow_torch, iter_count = opt.shape_pose_joint_opt(latent, T_ow_torch, render_data, points_w_torch, object_radius_max_m, cur_color)
            
            t1 = get_time()
            t_array.append(t1-t0)
            iter_array.append(iter_count)

            T_ow_cur = T_ow_torch.cpu().detach().numpy()
            T_wo = inv(T_ow_cur)

            # reconstruction with completion
            complete_mesh_o3d = mesh_extractor.complete_mesh(latent, T_wo, mean_color) # in world frame
            complete_pcd = complete_mesh_o3d.sample_points_uniformly(gt_point_count)

            complete_mesh_path = os.path.join(fruit_result_base, "complete_mesh.ply")
            o3d.io.write_triangle_mesh(complete_mesh_path, complete_mesh_o3d)
            # print("save the complete mesh to %s\n" % (complete_mesh_path))

            gt_pcd_clone = copy.deepcopy(gt_pcd)
            gt_pcd_w = gt_pcd_clone.transform(T_wg) 
            gt_pcd_w.paint_uniform_color(np.ones(3)*0.8)

            # define metrics
            cd_metric.update(gt_pcd_w,complete_pcd)
            pr_metric.update(gt_pcd_w,complete_pcd)

            # pose metrics
            final_scale = det(T_wo[:3,:3])**(1/3)
            T_wo_descale = T_wo

            # print(T_wo)
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

            # print(det(T_wo_descale[:3,:3])**(1/3))
            
            translation_error_vector = T_wg[:3,3] - T_wo_descale[:3,3]
            tran_error = norm(translation_error_vector)*1e3 # in mm
            print("E_tran (mm):")
            print(tran_error)
            tran_error_array.append(tran_error)

            # rot_error_test = np.transpose(T_wo_descale[:3,:3]) @ T_wg[:3,:3]
            # rot_error_test = rot_error_test[2,2]
            #rot_error_test = np.degrees(np.arccos(rot_error_test))

            rot_error = get_deg_between_vectors(T_wo_descale[:3,2], T_wg[:3,2])
            print("E_rot (deg):")
            print(rot_error)

            #print(rot_error_test)

            rot_error_array.append(rot_error)
            
            if cfg['vis']['vis_on']:
                vis.add_gt_scan(gt_pcd_w)
                vis.stop()
                vis.clean_vis()


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
