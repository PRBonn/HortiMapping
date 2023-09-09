#
# part of this file is modified from https://github.com/JingwenWang95/DSP-SLAM
#

import numpy as np
from numpy.linalg import inv
from tqdm import tqdm
import torch
import math
import time

from wild_completion.loss import compute_sdf_loss, compute_render_loss
from wild_completion.utils import get_time, exp_se3, exp_sim3, get_robust_res, rotation_matrix_to_axis_angle


class Optimizer(object):
    def __init__(self, cfg, decoder, mesher, vis=None):
        self.dev = cfg['device']
        self.dtype = torch.float32
        self.opt_cfg = cfg['opt']
        self.decoder = decoder
        self.mesher = mesher
        self.vis = vis
        self.vis_pause_time = cfg['vis']['vis_pause_s']

    # jointly optimize shape code and pose
    def shape_pose_joint_opt(self, latent, T_ow_torch, render_data, points_w_torch, cube_radius, cur_color):
        
        # key parameters for the optimization
        iter_count_max = self.opt_cfg['converge']['max_iter']
        epsilon_g = float(self.opt_cfg['converge']['epsilon_g'])
        epsilon_c = float(self.opt_cfg['converge']['epsilon_c'])
        epsilon_t = float(self.opt_cfg['converge']['epsilon_t'])
        epsilon_r = float(self.opt_cfg['converge']['epsilon_r'])
        epsilon_s = float(self.opt_cfg['converge']['epsilon_s'])
        max_render_frame = self.opt_cfg['render']['n_frame']
        num_depth_samples = self.opt_cfg['render']['n_sample_on_ray']
        occ_cutoff_m = float(self.opt_cfg['render']['occ_cutoff_m'])
        log_sdf_occ = self.opt_cfg['render']['log_sdf_occ']
        occlusion_aware = self.opt_cfg['render']['occlusion_on']
        w_recon = float(self.opt_cfg['weight']['w_recon'])
        w_depth = float(self.opt_cfg['weight']['w_depth'])
        w_mask = float(self.opt_cfg['weight']['w_mask'])
        w_codereg = float(self.opt_cfg['weight']['w_codereg']) # larger would cause the pose more stable, but smaller would lead to more variance on the shape # 2.5e-3
        lm_on = self.opt_cfg['lm']['lm_on']
        lm_eye = self.opt_cfg['lm']['lm_eye']
        lm_lambda_0 = float(self.opt_cfg['lm']['lm_lambda_0'])  # very useful actually # make it also changeable (maybe not)
        t_recon = float(self.opt_cfg['recon']['robust_th_m'])
        t_depth = float(self.opt_cfg['render']['robust_th_m'])
        robust_iter = self.opt_cfg['robust_iter'] # begin to apply robust loss from this iteration (begin from 0)
        s_damp = float(self.opt_cfg['lm']['s_damp'])
        estimate_scale = self.opt_cfg['scale_on']
        

        if estimate_scale:
            pose_dim = 7 # 7DOF (Sim3)
        else:
            pose_dim = 6 # 6DOF (Se3)

        code_len = latent.shape[0]
        est_count = pose_dim + code_len # 7+c

        # init mesh
        T_ow = T_ow_torch.cpu().detach().numpy()
        T_wo = inv(T_ow)
        cur_scale = torch.det(T_ow_torch[:3, :3]) ** (-1 / 3)

        if self.vis is not None:
            cur_mesh = self.mesher.complete_mesh(latent, np.eye(4), cur_color)
            self.vis.update_mesh_pose(cur_mesh, T_wo, 0)

            time.sleep(self.vis_pause_time)
            

        # get render data
        # only use part of the matched frames for rendering
        render_frame_count = len(render_data["T_wc"])
        sample_frame_ind = np.linspace(0, render_frame_count-1, min(max_render_frame,render_frame_count)).astype(np.int32)
        render_T_wc_torch = render_data["T_wc"]
        render_rays_fg = render_data["rays_fg"]
        render_rays_bg = render_data["rays_bg"]
        render_depth_fg = render_data["depth_fg"]
        render_depth_bg = render_data["depth_bg"]

        iter_count=0
        
        
        for i in range(iter_count_max):
            # LM optimization

            t1 = get_time()

            # time are mainly spent here

            # -------------------------------------------------------------------------------------
            # I. rendering term
            res_render_depth = torch.empty(0,1,1, device=self.dev, dtype=self.dtype)
            J_render_depth = torch.empty(0,1,est_count, device=self.dev, dtype=self.dtype)
            res_render_mask = torch.empty(0,1,1, device=self.dev, dtype=self.dtype)
            J_render_mask = torch.empty(0,1,est_count, device=self.dev, dtype=self.dtype)

            for idx in sample_frame_ind: # for every image that we would conduct the rendering
                T_wc_torch = render_T_wc_torch[idx]
                T_oc_torch = T_ow_torch @ T_wc_torch
                T_co_torch = torch.inverse(T_oc_torch)

                depth_range = cube_radius * cur_scale
                # would be calculate for each frame for each iteration
                # be careful about the number here, also add to the config
                depth_min, depth_max = T_co_torch[2, 3] - 1.0 * depth_range, T_co_torch[2, 3] + 0.8 * depth_range 
                sampled_depth_along_rays = torch.linspace(depth_min, depth_max, num_depth_samples, device=self.dev, dtype=self.dtype)

                ray_fgbg = torch.cat((render_rays_fg[idx], render_rays_bg[idx]), 0)
                
                # better to limit the loss for bg (we want the model to not shrink too small, better to be larger)
                rend_result = compute_render_loss(self.decoder, latent, ray_fgbg, render_depth_fg[idx], 
                    render_depth_bg[idx], T_oc_torch, sampled_depth_along_rays, estimate_scale, log_sdf_occ,
                    occ_cutoff_m, depth_range, occlusion_aware) 

                if rend_result is not None:
                    cur_res_depth, cur_jac_depth_tow, cur_jac_depth_code, cur_res_mask, cur_jac_mask_tow, cur_jac_mask_code = rend_result
                    # depth
                    cur_J_depth = torch.cat([cur_jac_depth_tow, cur_jac_depth_code], dim=-1) # N, 1, 7+c 
                    res_render_depth = torch.cat([res_render_depth, cur_res_depth], dim=0) 
                    J_render_depth = torch.cat([J_render_depth, cur_J_depth], dim=0)
                    # mask
                    cur_J_mask = torch.cat([cur_jac_mask_tow, cur_jac_mask_code], dim=-1) # N, 1, 7+c 
                    res_render_mask = torch.cat([res_render_mask, cur_res_mask], dim=0) 
                    J_render_mask = torch.cat([J_render_mask, cur_J_mask], dim=0)
                else:
                    print("This frame is not valid")
                    continue
                
            depth_obs_count = res_render_depth.shape[0] 
            mask_obs_count = res_render_mask.shape[0]
            J_render_depth_t = J_render_depth.transpose(1, 2)  # N, 7+c, 1 
            J_render_mask_t = J_render_mask.transpose(1, 2)  # N, 7+c, 1 

            t2 = get_time()

            if i >= robust_iter:
                robust_res_render_depth, robust_w = get_robust_res(res_render_depth, t_depth)
            else:
                robust_res_render_depth = res_render_depth
                robust_w = torch.ones_like(res_render_depth)

            # better to visualize with wandb
            H_render_depth = w_depth * (robust_w*torch.bmm(J_render_depth_t, J_render_depth)).sum(0).squeeze() / depth_obs_count
            b_render_depth = -w_depth * (robust_w*torch.bmm(J_render_depth_t, res_render_depth)).sum(0).squeeze() / depth_obs_count
            # print("H_render_depth:")
            # print(H_render_depth)

            H_render_mask = w_mask * torch.bmm(J_render_mask_t, J_render_mask).sum(0).squeeze() / mask_obs_count
            b_render_mask = -w_mask * torch.bmm(J_render_mask_t, res_render_mask).sum(0).squeeze() / mask_obs_count
            # print("H_render_mask:")
            # print(H_render_mask)

            t3 = get_time()
                            
            # -------------------------------------------------------------------------------------
            # II. sdf reconstruction term
            # transform point cloud
            cur_points_o = (points_w_torch[..., None, :] * T_ow_torch[:3, :3]).sum(-1) + T_ow_torch[:3, 3] # N, 3 

            recon_result = compute_sdf_loss(self.decoder, latent, cur_points_o, estimate_scale) # res: N, 1, 1
            if recon_result is not None:
                res_recon, jac_recon_tow, jac_recon_code = recon_result
            else:
                print("This submap is not valid")
                break

            recon_obs_count = jac_recon_tow.shape[0]
            J_recon = torch.cat([jac_recon_tow, jac_recon_code], dim=-1) # N, 1, 7+c 
            J_recon_t = J_recon.transpose(1, 2)  # N, 7+c, 1 

            t4 = get_time()

            if i >= robust_iter:
                robust_res_recon, robust_w = get_robust_res(res_recon, t_recon) # time consuming, disable this part
            else:
                robust_res_recon = res_recon
                robust_w = torch.ones_like(res_recon)

            H_recon = w_recon * (robust_w*torch.bmm(J_recon_t, J_recon)).sum(0).squeeze() / recon_obs_count
            b_recon = -w_recon * (robust_w*torch.bmm(J_recon_t, res_recon)).sum(0).squeeze() / recon_obs_count
            
            # print("H_recon:")
            # print(H_recon)

            t5 = get_time()
            
            # -------------------------------------------------------------------------------------
            # III. code regularization term

            H_codereg = torch.zeros_like(H_recon)
            H_codereg[pose_dim:est_count, pose_dim:est_count] = w_codereg * torch.eye(code_len, device = self.dev)
            b_codereg = torch.zeros_like(b_recon)
            b_codereg[pose_dim:est_count] = -w_codereg*latent

            # print("H_regul:")
            # print(H_codereg)
            
            # -------------------------------------------------------------------------------------
            # All together
            H = torch.zeros_like(H_recon) # 7+c,7+c
            H += H_render_depth
            H += H_render_mask
            H += H_recon 
            H += H_codereg

            # add scale damping
            if estimate_scale:
                H[pose_dim-1, pose_dim-1] += s_damp  # add a damping for scale

            if lm_on: # levenberg-marquardt
                if lm_eye:
                    lm_lambda = lm_lambda_0 * torch.max(torch.diag(H))
                    H += lm_lambda * torch.eye(est_count, device = self.dev) # use identity
                else:
                    H += lm_lambda_0 * torch.diag(torch.diag(H)) # use JtJ

            b = torch.zeros_like(b_recon) # 7+c,1
            b += b_render_depth
            b += b_render_mask
            b += b_recon
            b += b_codereg

            # optimize
            delta_x = torch.mv(torch.inverse(H), b)
            delta_p = delta_x[:pose_dim] # pose part
            delta_c = delta_x[pose_dim:est_count] # code part

            if estimate_scale:
                delta_T = exp_sim3(delta_p) # we still keep the last row as [0 0 0 1]
            else:
                delta_T = exp_se3(delta_p) # to transformation

            T_ow_torch = torch.mm(delta_T, T_ow_torch) # transfomration update 
            latent += delta_c # code update

            cur_scale = torch.det(T_ow_torch[:3, :3]) ** (-1 / 3)
            delta_scale = torch.det(delta_T[:3, :3]) ** (1 / 3)
            delta_tran = torch.norm(delta_T[0:3,3]) * cur_scale
            delta_rot = torch.norm(rotation_matrix_to_axis_angle(delta_T[0:3,0:3]*cur_scale))*180.0/math.pi  # to degree
            
            loss_recon_l1 = torch.mean(torch.abs(robust_res_recon)).item()
            loss_depth_l1 = torch.mean(torch.abs(robust_res_render_depth)).item()
            loss_mask_l1 = torch.mean(torch.abs(res_render_mask)).item()

            cur_T_wo = inv(T_ow_torch.cpu().detach().numpy())

            print("Current scale:", cur_scale.item())
            print(i , ", Recon loss:", "{:.5}".format(loss_recon_l1), ", Depth render loss:", "{:.5}".format(loss_depth_l1), ", Mask render loss:", "{:.5}".format(loss_mask_l1))

            t6 = get_time()

            print("Render time (s):", "{:.3}".format(t3-t1), ", Recon time (s):", "{:.3}".format(t5-t3), ", Optim time (s):", "{:.3}".format(t6-t5))

            if self.vis is not None:
                cur_mesh = self.mesher.complete_mesh(latent, np.eye(4), cur_color) # here it's still in the object coordinate system
                self.vis.update_mesh_pose(cur_mesh, cur_T_wo, i+1)
                time.sleep(self.vis_pause_time)

            iter_count = i+1

            # judge convergence
            if (torch.max(torch.abs(b)) < epsilon_g and i > 1):
                print('**** Convergence in gradient  ****')
                break
            if (torch.max(torch.abs(delta_c/(latent+1e-12))) < epsilon_c  and i > 1): 
                print('**** Convergence in Shape Latent Code ****')
                break
            if (delta_tran < epsilon_t and delta_rot < epsilon_r and delta_scale < epsilon_s and i > 1): 
                print('**** Convergence in Pose Parameters ****')
                break
            if (i == iter_count_max-1): 
                print('**** Convergence in Maximum Iteration Numbers ****')

            # TODO: add the rotation constraint, if strange rotation -> then reject this submap

        # print("Final shape code:")
        # print(latent)

        if self.vis is not None:
            self.vis.stop()

        return latent, T_ow_torch, iter_count
    

    # only optimize shape using deep-sdf (no pose optimization)
    def shape_opt_deepsdf(self, latent, T_ow_torch, points_w_torch, cur_color):
        
        # key parameters for the optimization
        iter_count_max = self.opt_cfg['converge']['max_iter']
        epsilon_g = float(self.opt_cfg['converge']['epsilon_g'])
        epsilon_c = float(self.opt_cfg['converge']['epsilon_c'])
        max_render_frame = self.opt_cfg['render']['n_frame']
        w_recon = float(self.opt_cfg['weight']['w_recon'])
        w_codereg = float(self.opt_cfg['weight']['w_codereg']) # larger would cause the pose more stable, but smaller would lead to more variance on the shape # 2.5e-3
        lm_on = self.opt_cfg['lm']['lm_on']
        lm_eye = self.opt_cfg['lm']['lm_eye']
        lm_lambda_0 = float(self.opt_cfg['lm']['lm_lambda_0'])  # very useful actually # make it also changeable (maybe not)
        t_recon = float(self.opt_cfg['recon']['robust_th_m'])
        robust_iter = self.opt_cfg['robust_iter'] # begin to apply robust loss from this iteration (begin from 0)
        estimate_scale = self.opt_cfg['scale_on']
        
        pose_dim = 0
        
        code_len = latent.shape[0]
        est_count = pose_dim + code_len # 7+c

        # init mesh
        T_ow = T_ow_torch.cpu().detach().numpy()
        T_wo = inv(T_ow)

        if self.vis is not None:
            cur_mesh = self.mesher.complete_mesh(latent, np.eye(4), cur_color)
            self.vis.update_mesh_pose(cur_mesh, T_wo, 0)
     
        iter_count=0
        
        for i in range(iter_count_max):
            # LM optimization

            # -------------------------------------------------------------------------------------
            # II. sdf reconstruction term
            # transform point cloud
            cur_points_o = (points_w_torch[..., None, :] * T_ow_torch[:3, :3]).sum(-1) + T_ow_torch[:3, 3] # N, 3 

            recon_result = compute_sdf_loss(self.decoder, latent, cur_points_o, estimate_scale) # res: N, 1, 1
            if recon_result is not None:
                res_recon, _, jac_recon_code = recon_result
            else:
                print("This submap is not valid")
                break

            recon_obs_count = jac_recon_code.shape[0]
            J_recon = jac_recon_code # N, 1, c
            J_recon_t = J_recon.transpose(1, 2)  # N, c, 1 

            if i >= robust_iter:
                robust_res_recon, robust_w = get_robust_res(res_recon, t_recon)
            else:
                robust_res_recon = res_recon
                robust_w = torch.ones_like(res_recon)

            H_recon = w_recon * (robust_w*torch.bmm(J_recon_t, J_recon)).sum(0).squeeze() / recon_obs_count
            b_recon = -w_recon * (robust_w*torch.bmm(J_recon_t, res_recon)).sum(0).squeeze() / recon_obs_count
            
            # print("H_recon:")
            # print(H_recon)

            # -------------------------------------------------------------------------------------
            # III. code regularization term

            H_codereg = torch.zeros_like(H_recon)
            H_codereg[pose_dim:est_count, pose_dim:est_count] = w_codereg * torch.eye(code_len, device = self.dev)
            b_codereg = torch.zeros_like(b_recon)
            b_codereg[pose_dim:est_count] = -w_codereg*latent

            # print("H_regul:")
            # print(H_codereg)
            
            # -------------------------------------------------------------------------------------
            # All together
            H = torch.zeros_like(H_recon) # 7+c,7+c
            H += H_recon 
            H += H_codereg

            if lm_on: # levenberg-marquardt
                if lm_eye:
                    lm_lambda = lm_lambda_0 * torch.max(torch.diag(H))
                    H += lm_lambda * torch.eye(est_count, device = self.dev) # use identity
                else:
                    H += lm_lambda_0 * torch.diag(torch.diag(H)) # use JtJ

            b = torch.zeros_like(b_recon) # 7+c,1
            b += b_recon
            b += b_codereg

            # optimize
            delta_x = torch.mv(torch.inverse(H), b)
            # delta_p = delta_x[:pose_dim] # pose part
            delta_c = delta_x[pose_dim:est_count] # code part

            latent += delta_c # code update

            loss_recon_l1 = torch.mean(torch.abs(robust_res_recon)).item()

            print(i , ", Recon:", "{:.5}".format(loss_recon_l1))

            cur_T_wo = inv(T_ow_torch.cpu().detach().numpy())

            if self.vis is not None:
                cur_mesh = self.mesher.complete_mesh(latent, np.eye(4), cur_color) # here it's still in the object coordinate system
                self.vis.update_mesh_pose(cur_mesh, cur_T_wo, i+1)

            iter_count = i+1

            # judge convergence
            if (torch.max(torch.abs(b)) < epsilon_g and i > 1):
                print('**** Convergence in gradient  ****')
                break
            if (torch.max(torch.abs(delta_c/(latent+1e-12))) < epsilon_c  and i > 1): 
                print('**** Convergence in Shape Latent Code ****')
                break
            if (i == iter_count_max-1): 
                print('**** Convergence in Maximum Iteration Numbers ****')

        return latent, T_ow_torch, iter_count