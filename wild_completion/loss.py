#
# part of this file is modified from https://github.com/JingwenWang95/DSP-SLAM
#

import torch
from wild_completion.utils import *

def compute_render_loss(decoder, latent_vector, ray_directions, depth_obs_fg, depth_obs_bg, 
    t_obj_cam, sampled_ray_depth, scale_on = False, log_occ_on = False, occupancy_th=0.01, 
    object_bbx_radius = 0.1, occlusion_on = True,
    occlusion_th = 0.03, min_valid_sample = 100, min_grad_thre=1e-6):
    """
    get (mask and depth) rendering loss and jacobian
    :param decoder: DeepSDF decoder
    :param latent_vector: shape code
    :param ray_directions: (N, 3) under camera coordinate
    :param depth_obs_fg: (Nf,) observed depth values for foreground pixels
    :param depth_obs_bg: (Nb,) observed depth values for background pixels, Nf+Nb = N
    :param t_obj_cam: c2o transformation (4, 4) in Sim(3)
    :param sampled_ray_depth: (M,) linspace between d_min and d_max
    :param occupancy_th: cut-off threshold for converting SDF to occupancy
    :return: Jacobian wrt pose (K, 1, 6), Jacobian wrt shape code (K, 1, code_len), error residuals (K, 1, 1)
    Note that K is the number of points that have non-zero Jacobians out of a total of N * M points
    """

    depth_obs = torch.cat((depth_obs_fg, depth_obs_bg), 0)
    depth_obs_fg_count = depth_obs_fg.shape[0]

    # (n_rays, num_samples_per_ray, 3) = (n_rays, 1, 3) * (num_samples_per_ray, 1)
    sampled_points_cam = ray_directions[..., None, :] * sampled_ray_depth[:, None]
    # (n_rays, num_samples_per_ray, 3)
    sampled_points_obj = \
        (sampled_points_cam[..., None, :] * t_obj_cam.cuda()[:3, :3]).sum(-1) + t_obj_cam.cuda()[:3, 3] # transform to object frame
    n_rays = sampled_points_obj.shape[0]
    n_depths = sampled_ray_depth.shape[0]

    # (num_rays, num_samples_per_ray) # this part can also be deleted
    valid_indices = torch.where(torch.norm(sampled_points_obj, dim=-1) < object_bbx_radius) # close to the object
    # (n_valid, 3) flattened = (n_rays, n_depth_sample, 3)[x, y, :]
    query_points_obj = sampled_points_obj[valid_indices[0], valid_indices[1], :]

    # if too few query points, return immediately
    if query_points_obj.shape[0] < min_valid_sample:
        return None

    # flattened
    with torch.no_grad():
        sdf_values = decode_sdf(decoder, latent_vector, query_points_obj).squeeze() # here we don't need the grad

    if sdf_values is None:
        raise Exception("no valid query points?")

    # Full dimension (n_rays, n_samples_per_ray)
    occ_values = torch.full((n_rays, n_depths), 0.).cuda()
    valid_indices_x, valid_indices_y = valid_indices  # (indices on x, y dimension)
    
    if log_occ_on:
        logistic_gaussian_ratio = 0.55
        sigma_sigmoid = occupancy_th/3*logistic_gaussian_ratio
        occ_values[valid_indices_x, valid_indices_y] = sdf_to_occupancy_log(sdf_values, sigma_sigmoid)
        # the ratio between the sigma of a fitting of gaussian distribution using derivative of logistic distribution  
    else:
        occ_values[valid_indices_x, valid_indices_y] = sdf_to_occupancy(sdf_values, occupancy_th)

    with_grad = (sdf_values > -occupancy_th) & (sdf_values < occupancy_th)
    with_grad_indices_x = valid_indices_x[with_grad]
    with_grad_indices_y = valid_indices_y[with_grad]

    # point-wise values, i.e. multiple points might belong to one ray (m, n_samples_per_ray)
    occ_values_with_grad = occ_values[with_grad_indices_x, :] # with dulplication
    k = occ_values_with_grad.shape[0]  # number of sample points with grad
    d_min = sampled_ray_depth[0]
    d_max = sampled_ray_depth[-1]
    delta_d = (d_max - d_min) / (n_depths - 1)

    # d_term_bg = term_depth_ratio * d_max
    d_term_bg = d_max + delta_d # this need to be fulfilled to make the derivation work

    # Render function (with dulplication)
    acc_trans = torch.cumprod(1 - occ_values_with_grad, dim=-1)
    acc_trans_augment = torch.cat(
        (torch.ones(k, 1).cuda(), acc_trans),
        dim=-1)
    o = torch.cat(
        (occ_values_with_grad, torch.ones(k, 1).cuda()),
        dim=-1)
    d = torch.cat(
        (sampled_ray_depth, torch.tensor([d_term_bg]).cuda()),
        dim=-1)
    term_prob = (o * acc_trans_augment)
    term_prob_no_end = term_prob[:,:-1]
    occ_ray = torch.sum(term_prob_no_end, dim=-1)

    # rendered depth values (k,), for the k rays
    d_u = torch.sum(d * term_prob, dim=-1) # original
    # d_u = torch.sum(sampled_ray_depth * term_prob_no_end, dim=-1)  # do not use the termination depth
    # var_u = torch.sum(term_prob * (d[None, :] - d_u[:, None]) ** 2, dim=-1)

    # Get Jacobian of depth residual wrt occupancy probability de_do
    o_k = occ_values[with_grad_indices_x, with_grad_indices_y]
    dm_do = acc_trans[:,-1] / (1. - o_k) # mask derivative
    l = torch.arange(n_depths).cuda()
    l = l[None, :].repeat(k, 1)
    acc_trans[l < with_grad_indices_y[:, None]] = 0. # only keep after
   
    de_do = acc_trans.sum(dim=-1) * delta_d / (1. - o_k)
    # de_do = (acc_trans.sum(dim=-1) * delta_d - d_term_bg * acc_trans[:,-1]) / (1. - o_k)

    # Remove points with zero gradients, and get de_ds = de_do * do_ds
    non_zero_grad = (de_do > min_grad_thre) # this is actually disabled
    de_do = de_do[non_zero_grad]
    dm_do = dm_do[non_zero_grad]
    d_u = d_u[non_zero_grad] # m, rendered depth
    occ_ray = occ_ray[non_zero_grad]
    with_grad_indices_x = with_grad_indices_x[non_zero_grad] # for each sample point
    with_grad_indices_y = with_grad_indices_y[non_zero_grad]
    o_k = o_k[non_zero_grad]

    if log_occ_on:
        do_ds = -o_k * (1 -o_k) / sigma_sigmoid # don't forget the negative sign
    else:
        do_ds = -1. / (2 * occupancy_th)

    # de_ds = (de_do * delta_d * do_ds).view(-1, 1, 1) # for each point sample
    de_ds = (de_do * do_ds).view(-1, 1, 1)
    dm_ds = (dm_do * do_ds).view(-1, 1, 1) # for each point sample

    # get residuals
    # occlusion aware  
    # outside the mask and the measured depth is smaller than the render one
    if occlusion_on:
        # depth measurements for with grad samples (with dulplication)
        depth_obs_non_zero_grad = depth_obs[with_grad_indices_x]  # (m,)
        possible_occlusion_mask = (with_grad_indices_x >= depth_obs_fg_count) & (depth_obs_non_zero_grad < d_u - occlusion_th) & (depth_obs_non_zero_grad > 0.) # m, bool
        no_occlusion_mask = ~possible_occlusion_mask # we do not need the supervision in the potential occlusion part

        with_grad_indices_x = with_grad_indices_x[no_occlusion_mask]
        with_grad_indices_y = with_grad_indices_y[no_occlusion_mask]

        # # first fg, then bg 
        depth_obs[depth_obs_fg_count:] = d_term_bg # bg depth to d_term
        depth_obs_non_zero_grad = depth_obs[with_grad_indices_x]  # (m,)
        
        d_u = d_u[no_occlusion_mask]
        de_ds = de_ds[no_occlusion_mask, ...]
        
        dm_ds = dm_ds[no_occlusion_mask, ...]
        occ_ray = occ_ray[no_occlusion_mask]
    else: 
        depth_obs[depth_obs_fg_count:] = d_term_bg # bg depth to d_term
        depth_obs_non_zero_grad = depth_obs[with_grad_indices_x]


    res_d = depth_obs_non_zero_grad - d_u  # (m2,) # measured - rendered

    # those free-space rays would not be counted because there's no gradient for them

    # from sample point to the ray
    with_grad_indices_x_fg = with_grad_indices_x[with_grad_indices_x<depth_obs_fg_count]
    group_id_fg = torch.unique(with_grad_indices_x_fg)
    valid_ray_count_fg = group_id_fg.shape[0]
    
    group_id, groups_idx, groups_count = torch.unique(with_grad_indices_x, return_inverse=True, return_counts = True)
    valid_ray_count = group_id.shape[0]
    valid_ray_count_bg = valid_ray_count-valid_ray_count_fg

    occ_ray = torch.zeros(size=(valid_ray_count,)).cuda().scatter_add_(0, groups_idx, occ_ray) / groups_count
    res_d_ray = torch.zeros(size=(valid_ray_count,)).cuda().scatter_add_(0, groups_idx, res_d) / groups_count
    res_d_ray = res_d_ray.view(-1, 1, 1)

    mask_fg_ray = torch.ones(size=(valid_ray_count_fg,)).cuda()
    mask_bg_ray = torch.zeros(size=(valid_ray_count_bg,)).cuda()
    mask_ray = torch.cat((mask_fg_ray, mask_bg_ray))
    res_m_ray = occ_ray - mask_ray
    res_m_ray = res_m_ray.view(-1, 1, 1) # may change this from L1 to BCE loss

    # print("Valid FG:", valid_ray_count_fg, ",Valid BG:", valid_ray_count_bg)

    # print(occ_ray)
    # print(mask_ray)
    # print(res_mask_ray)
    
    # from sample point to the ray
    pts_with_grad_obj = sampled_points_obj[with_grad_indices_x, with_grad_indices_y]
    _, ds_di = get_batch_sdf_jacobian(decoder, latent_vector, pts_with_grad_obj) # here we get the grad for a subset of points
    
    dm_di = dm_ds * ds_di
    de_di = de_ds * ds_di  # (m2, 1, code_len + 3) 
    
    dm_dxo = dm_di[..., -3:]  # (m2, 1, 3)
    de_dxo = de_di[..., -3:]  # (m2, 1, 3)

    # Jacobian for pose and code
    if scale_on:
        dxo_dtow = get_points_to_pose_jacobian_sim3(pts_with_grad_obj)
        pose_para_count = 7
    else:
        dxo_dtow = get_points_to_pose_jacobian_se3(pts_with_grad_obj)
        pose_para_count = 6
    
    jac_d_tow = torch.bmm(de_dxo, dxo_dtow) # (m2, 1, 6 or 7)
    jac_d_code = de_di[..., :-3]  # (m2, 1, code_len)
    code_len = jac_d_code.shape[-1]

    jac_m_tow = torch.bmm(dm_dxo, dxo_dtow) # (m2, 1, 6 or 7)
    jac_m_code = dm_di[..., :-3]  # (m2, 1, code_len)

    groups_idx_dup = groups_idx.repeat_interleave(pose_para_count).view(-1,1,pose_para_count)
    jac_m_tow_ray = torch.zeros(size=(valid_ray_count,1,pose_para_count)).cuda().scatter_add_(0, groups_idx_dup, jac_m_tow)
    jac_d_tow_ray = torch.zeros(size=(valid_ray_count,1,pose_para_count)).cuda().scatter_add_(0, groups_idx_dup, jac_d_tow)

    groups_idx_dup = groups_idx.repeat_interleave(code_len).view(-1,1,code_len)
    jac_m_code_ray = torch.zeros(size=(valid_ray_count,1,code_len)).cuda().scatter_add_(0, groups_idx_dup, jac_m_code)
    jac_d_code_ray = torch.zeros(size=(valid_ray_count,1,code_len)).cuda().scatter_add_(0, groups_idx_dup, jac_d_code)
    
    return res_d_ray, jac_d_tow_ray, jac_d_code_ray, res_m_ray, jac_m_tow_ray, jac_m_code_ray

def compute_sdf_loss(decoder, latent_vector, pts_surface_obj, scale_on = False):
    """
    get sdf consistency loss and jacobian
    :param decoder: DeepSDF decoder
    :param latent_vector: shape code, torch tensor
    :param pts_surface_obj: surface points under object coordinate (N, 3), torch tensor
    :return: res_sdf: error residuals (N, 1, 1), 
             jac_recon_tow:Jacobian wrt pose (N, 1, 6), 
             jac_recon_code: Jacobian wrt shape code (N, 1, C)
    """
    # (n_sample_surface, 3)
    res_sdf, de_di = get_batch_sdf_jacobian(decoder, latent_vector, pts_surface_obj) # (N, 1, 1)
    # SDF term Jacobian
    de_dxo = de_di[..., -3:] # the last 3 are for xyz, [N, 1, 3]
    # Jacobian for pose (se3 6dof)
    if scale_on:
        dxo_dtoc = get_points_to_pose_jacobian_sim3(pts_surface_obj) # [N, 3, 7]
    else:        
        dxo_dtoc = get_points_to_pose_jacobian_se3(pts_surface_obj) # [N, 3, 6]
    
    jac_recon_tow = torch.bmm(de_dxo, dxo_dtoc) # batch matrix multipy # [N, 1, 6 or 7]
    # Jacobian for code
    jac_recon_code = de_di[..., :-3] # the ones before last 3 are for shape code, [N, 1, C]

    return res_sdf, jac_recon_tow, jac_recon_code