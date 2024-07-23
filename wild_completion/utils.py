#
# part of this file is modified from https://github.com/JingwenWang95/DSP-SLAM
#


import torch
from torch.autograd import grad
import time
import numpy as np
import os
import getpass
import json
import copy
from addict import Dict
import plyfile
import math
import open3d as o3d
import skimage.measure as measure
from collections import Counter
from deepsdf.deep_sdf.workspace import config_decoder, load_latent_vectors


def get_rays(sampled_pixels, invK):
    """
    This function computes the ray directions given sampled pixel
    and camera intrinsics
    :param sampled_pixels: (N, 2), order is [u, v]
    :param invK: (3, 3)
    :return: ray directions (N, 3) under camera frame
    """
    n = sampled_pixels.shape[0]
    # (n, 3) = (n, 2) (n, 1)
    u_hom = np.concatenate([sampled_pixels, np.ones((n, 1))], axis=-1)
    # (n, 3) = (n, 1, 3) * (3, 3)
    directions = (u_hom[:, None, :] * invK).sum(-1)

    return directions.astype(np.float32)

def get_render_data(submap_id, id_imgs, depth_imgs, cam_poses, img_size, invK, cfg,
                    min_pix_count_match = 400, max_bbx_size = 300, down_rate = 1):
    render_data = {"frame_id": [], "T_wc": [], "rays_fg": [], "rays_bg": [], "depth_fg": [], "depth_bg": [], "pix_fg": [], "pix_bg": [], "count": 0}
    
    cfg_render = cfg['opt']['render']
    fg_pix_count, bg_pix_count, bg_pad = cfg_render['n_fg_pix'], cfg_render['n_bg_pix'], cfg_render['n_bg_pad']
    dev = cfg["device"]
    dtype = torch.float32

    # for each frame
    for item in id_imgs.items():
        img_id = item[0] # id 
        submap_id_img = item[1] # img
        depth_img = depth_imgs[img_id] # get corresponding depth img
        mask_bool = (submap_id_img==submap_id)
        valid_depth_bool = (depth_img > 0.)
        valid_mask_bool = mask_bool & valid_depth_bool
        match = submap_id_img[valid_mask_bool]    
        if (match.shape[0] < min_pix_count_match): # at least to have so many pixels to be a valid match
            continue
        mask_v, mask_u = np.where(valid_mask_bool) # v, u (x, y)
        min_v = max(min(mask_v) - bg_pad, 0)
        max_v = min(max(mask_v) + bg_pad, img_size[0]-1) # height
        min_u = max(min(mask_u) - bg_pad, 0)
        max_u = min(max(mask_u) + bg_pad, img_size[1]-1) # width
        bbx_h, bbx_w = max_v - min_v + 1, max_u - min_u + 1
        if bbx_h > max_bbx_size or bbx_w > max_bbx_size:
            print("Too large bbx, possibly wrong data association, skip this frame")
            continue
        hh = np.linspace(min_v, max_v, int(bbx_h / down_rate)).astype(np.int32)
        ww = np.linspace(min_u, max_u, int(bbx_w / down_rate)).astype(np.int32)
        crop_h, crop_w = hh.shape[0], ww.shape[0]
        hh = hh[:, None].repeat(crop_w, axis=1)
        ww = ww[None, :].repeat(crop_h, axis=0)
        sampled_pixels = np.concatenate([hh[:, :, None], ww[:, :, None]], axis=-1).reshape(-1, 2)
        vv, uu = sampled_pixels[:, 0], sampled_pixels[:, 1]
        valid_bg = ~mask_bool[vv, uu] # we do not care about the depth measurements for bg samples
        sampled_pixels_bg = np.concatenate([uu[valid_bg, None], vv[valid_bg, None]], axis=-1) # u,v
        depth_bg = depth_img[vv[valid_bg], uu[valid_bg]]
        if sampled_pixels_bg.shape[0] > bg_pix_count: # max sample_bg_pix_count
            sample_ind = np.random.choice(sampled_pixels_bg.shape[0], bg_pix_count, replace=False)
            #sample_ind = np.linspace(0, sampled_pixels_bg.shape[0]-1, bg_pix_count).astype(np.int32) 
            sampled_pixels_bg = sampled_pixels_bg[sample_ind, :]
            depth_bg = depth_bg[sample_ind]
        rays_bg = get_rays(sampled_pixels_bg, invK).astype(np.float32)
            
        valid_fg = valid_mask_bool[vv, uu] 
        # valid_fg = mask_bool[vv, uu] # we also don't force the foreground to have valid depth
        sampled_pixels_fg = np.concatenate([uu[valid_fg, None], vv[valid_fg, None]], axis=-1) # u,v
        depth_fg = depth_img[vv[valid_fg], uu[valid_fg]]
        if sampled_pixels_fg.shape[0] > fg_pix_count: # max sample_fg_pix_count
            sample_ind = np.random.choice(sampled_pixels_fg.shape[0], fg_pix_count, replace=False)
            #sample_ind = np.linspace(0, sampled_pixels_fg.shape[0]-1, fg_pix_count).astype(np.int32)
            sampled_pixels_fg = sampled_pixels_fg[sample_ind, :]
            depth_fg = depth_fg[sample_ind]
        rays_fg = get_rays(sampled_pixels_fg, invK).astype(np.float32)

        render_data["frame_id"].append(img_id)
        render_data["rays_fg"].append(torch.tensor(rays_fg, device=dev, dtype=dtype))
        render_data["rays_bg"].append(torch.tensor(rays_bg, device=dev, dtype=dtype))
        render_data["depth_fg"].append(torch.tensor(depth_fg, device=dev, dtype=dtype))
        render_data["depth_bg"].append(torch.tensor(depth_bg, device=dev, dtype=dtype))
        render_data["T_wc"].append(torch.tensor(cam_poses[img_id], device=dev, dtype=dtype))
        # just for vis
        render_data["pix_fg"].append(sampled_pixels_fg) # u,v
        render_data["pix_bg"].append(sampled_pixels_bg) # u,v
        render_data["count"] += 1
        # print("FG:", sampled_pixels_fg.shape[0], " BG:", sampled_pixels_bg.shape[0])
    
    # print("Found render data for %i frames" %render_data["count"])
    return render_data


def get_gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        is_grads_batched=False
    )[0]
    return points_grad


def sdf_to_occupancy(sdf_tensor, th=0.01):
    """
    :param sdf_tensor: torch tensor
    :param th: cut-off threshold, o(x>th) = 0, o(x<-th) = 1
    :return: occ_tensor: torch tensor
    """
    occ_tensor = torch.clamp(sdf_tensor, min=-th, max=th)
    occ_tensor = 0.5 - occ_tensor / (2 * th)
    return occ_tensor


def sdf_to_occupancy_log(sdf_tensor, sigma=0.01):
    """
    :param sdf_tensor: torch tensor
    :param sigma: logistic function sigma
    :return: occ_tensor: torch tensor
    """
    return torch.sigmoid(-sdf_tensor/sigma)

def decode_sdf(decoder, lat_vec, x, max_batch=64**3):
    """
    :param decoder: DeepSDF Decoder
    :param lat_vec: torch.Tensor (code_len,), latent code
    :param x: torch.Tensor (N, 3), query positions
    :return: batched outputs (N, )
    :param max_batch: max batch size
    :return:
    """

    num_samples = x.shape[0]

    head = 0

    # get sdf values given query points
    sdf_values_chunks = []
    with torch.no_grad():
        while head < num_samples:
            x_subset = x[head : min(head + max_batch, num_samples), 0:3].cuda()

            latent_repeat = lat_vec.expand(x_subset.shape[0], -1)
            fp_inputs = torch.cat([latent_repeat, x_subset], dim=-1)
            sdf_values = decoder(fp_inputs).squeeze()

            sdf_values_chunks.append(sdf_values)
            head += max_batch

    sdf_values = torch.cat(sdf_values_chunks, 0).cuda()
    return sdf_values


def get_batch_sdf_jacobian(decoder, lat_vec, x):
    """
    :param decoder: DeepSDF Decoder
    :param lat_vec: torch.Tensor (code_len,), latent code
    :param x: torch.Tensor (N, 3), query position
    :return: the sdf predictions and the batched Jacobian (N, 1, code_len + 3)
    """
    n = x.shape[0]
    latent_repeat = lat_vec.expand(n, -1)
    
    inputs = torch.cat([latent_repeat, x], 1)

    inputs = inputs.unsqueeze(1)  # (n, 1, in_dim)
    inputs.requires_grad = True
    y = decoder(inputs)  # (n, 1, 1)

    g = get_gradient(inputs, y) # (n, 1, code_len+3)

    return y.detach(), g.detach()


# Note that SE3 is ordered as (translation, rotation)
def get_points_to_pose_jacobian_se3(points):
    """
    :param points: Transformed points y = Tx = Rx + t, T in SE(3)
    :return: batched Jacobian of transformed points y wrt pose T using Lie Algebra (left perturbation)
    """
    n = points.shape[0]
    eye = torch.eye(3).view(1, 3, 3)
    batch_eye = eye.repeat(n, 1, 1).cuda()
    zero = torch.zeros(n).cuda() # N,1
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # N,1
    negate_hat = torch.stack(
        [torch.stack([zero, -z, y], dim=-1),
         torch.stack([z, zero, -x], dim=-1),
         torch.stack([-y, x, zero], dim=-1)],
        dim=-1
    )
    jac = torch.cat((batch_eye, negate_hat), dim=-1)
    return jac


def exp_se3(x):
    """
    :param x: Cartesian vector of Lie Algebra se(3)
    :return: exponential map of x [translation, rotation]
    """
    v = x[:3]  # translation
    w = x[3:6]  # rotation
    w_hat = torch.tensor([[0., -w[2], w[1]],
                          [w[2], 0., -w[0]],
                          [-w[1], w[0], 0.]]).cuda()
    w_hat_second = torch.mm(w_hat, w_hat)

    theta = torch.norm(w)
    theta_2 = theta ** 2
    theta_3 = theta ** 3
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    eye_3 = torch.eye(3).cuda()

    eps = 1e-8

    if theta <= eps:
        e_w = eye_3
        j = eye_3
    else:
        e_w = eye_3 + w_hat * sin_theta / theta + w_hat_second * (1. - cos_theta) / theta_2
        k1 = (1 - cos_theta) / theta_2
        k2 = (theta - sin_theta) / theta_3
        j = eye_3 + k1 * w_hat + k2 * w_hat_second

    rst = torch.eye(4).cuda()
    rst[:3, :3] = e_w
    rst[:3, 3] = torch.mv(j, v)

    return rst


def get_points_to_pose_jacobian_sim3(points):
    """
    :param points: Transformed points x = Ty = Ry + t, T in Sim(3)
    :return: batched Jacobian of transformed points wrt pose T
    """
    n = points.shape[0]
    eye = torch.eye(3).view(1, 3, 3)
    batch_eye = eye.repeat(n, 1, 1).cuda()
    zero = torch.zeros(n).cuda()
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    negate_hat = torch.stack(
        [torch.stack([zero, -z, y], dim=-1),
         torch.stack([z, zero, -x], dim=-1),
         torch.stack([-y, x, zero], dim=-1)],
        dim=-1
    )

    return torch.cat((batch_eye, negate_hat, points[..., None]), dim=-1)


def exp_sim3(x):
    """
    :param x: Cartesian vector of Lie Algebra sim(3)
    :return: exponential map of x
    """
    v = x[:3]  # translation
    w = x[3:6]  # rotation
    s = x[6]  # scale

    w_hat = torch.tensor([[0., -w[2], w[1]],
                          [w[2], 0., -w[0]],
                          [-w[1], w[0], 0.]]).cuda()
    w_hat_second = torch.mm(w_hat, w_hat)

    theta = torch.norm(w)
    theta_2 = theta ** 2
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    e_s = torch.exp(s)
    s_2 = s ** 2
    eye_3 = torch.eye(3).cuda()

    eps = 1e-8
    if theta <= 1e-8:
        if s == 0:
            e_w = eye_3
            j = eye_3
        else:
            e_w = eye_3
            c = (e_s - 1.) / s
            j = c * eye_3
    else:
        e_w = eye_3 + w_hat * sin_theta / theta + w_hat_second * (1. - cos_theta) / theta_2
        a = e_s * sin_theta
        b = e_s * cos_theta
        c = 0. if s <= eps else (e_s - 1.) / s
        k_0 = c * eye_3
        k_1 = (a * s + (1 - b) * theta) / (s_2 + theta_2)
        k_2 = c - ((b - 1) * s + a * theta) / (s_2 + theta_2)
        j = k_0 + k_1 * w_hat / theta + k_2 * w_hat_second / theta_2

    rst = torch.eye(4).cuda()
    rst[:3, :3] = e_s * e_w
    rst[:3, 3] = torch.mv(j, v)

    return rst


def huber_norm_weights(x, b=0.02):
    """
    :param x: norm of residuals, torch.Tensor (N,)
    :param b: threshold
    :return: weight vector torch.Tensor (N, )
    """
    # x is residual norm
    res_norm = torch.zeros_like(x)
    res_norm[x <= b] = x[x <= b] ** 2
    res_norm[x > b] = 2 * b * x[x > b] - b ** 2
    x[x == 0] = 1.
    weight = torch.sqrt(res_norm) / x # = 1 in the window and < 1 out of the window

    return weight


def get_robust_res(res, b):
    """
    :param res: residual vectors
    :param b: threshold
    :return: residuals after applying huber norm
    """
    # print(res.shape[0])
    res = res.view(-1, 1, 1)
    res_norm = torch.abs(res)
    # print(res.shape[0])
    w = huber_norm_weights(res_norm, b=b)
    # print(w.shape[0])
    robust_res = w * res
    # loss = torch.mean(robust_res ** 2) # use l2 loss

    return robust_res, w**2

def rotation_matrix_to_axis_angle(R: torch.Tensor):

    # Ensure the input matrix is a valid rotation matrix
    assert torch.is_tensor(R) and R.shape == (3, 3), "Invalid rotation matrix"
    # Compute the trace of the rotation matrix
    trace = torch.trace(R)
    # Compute the angle of rotation
    angle = torch.acos((trace - 1) / 2)

    return angle # rad 

def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor):
    angle = axis_angle.norm()
    axis = axis_angle/angle
    eye = torch.eye(3, device=axis_angle.device,dtype=axis_angle.dtype)
    S = skew(axis)
    R = eye + angle.sin()*S + (1-angle.cos())*(S@S)

    return R 


def skew(v):
    S = torch.zeros(3, 3, device=v.device,dtype=v.dtype)
    S[0, 1] = -v[2]
    S[0, 2] = v[1]
    S[1, 2] = -v[0]
    return S - S.T


def clean_mesh(cur_mesh, sample_point_count = 5000, cluster_dist_thre = 0.01, outlier_point_ratio = 0.02,
               filter_isolated_mesh = False, filter_cluster_min_tri = 20):

    if filter_isolated_mesh:
        triangle_clusters, cluster_n_triangles, cluster_area = (cur_mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < filter_cluster_min_tri
        cur_mesh.remove_triangles_by_mask(triangles_to_remove)

    # # use possion sampling
    # cur_pcd = cur_mesh.sample_points_poisson_disk(number_of_points=sample_point_count, init_factor=5)
    cur_pcd = cur_mesh.sample_points_uniformly(number_of_points=sample_point_count)
    cur_pcd_clean = clean_pcd(cur_pcd, cluster_dist_thre, outlier_point_ratio)

    return cur_pcd_clean

def clean_pcd(cur_pcd, cluster_dist_thre = 0.01, outlier_point_ratio = 0.02):
    cur_point_count = len(cur_pcd.points)
    min_instance_pts = int(cur_point_count * outlier_point_ratio)
    cur_cluster_labels = np.array(cur_pcd.cluster_dbscan(eps=cluster_dist_thre, min_points=min_instance_pts), dtype='int')
    cluster_counter = Counter(cur_cluster_labels)
    all_labels = cluster_counter.most_common()   # Returns all unique items and their counts
    mode_label = cluster_counter.most_common(1)[0][0]  # Returns the highest occurring item
    # print("Clusters:", all_labels, "-->", mode_label)
    main_cluster_indices = np.where(cur_cluster_labels==mode_label)[0].tolist()
    cur_pcd_clean = cur_pcd.select_by_index(main_cluster_indices)
    return cur_pcd_clean


def get_pose_init(cur_pcd, bg_pcd, bbx_pad = 0.01, min_bbx_size = 0.03, max_bbx_size = 0.16, 
    min_nearby_bg_pts = 10, max_init_rot_deg = 45):
    valid_flag = True

    cur_box = cur_pcd.get_axis_aligned_bounding_box()
    cur_center, cur_extent = cur_box.get_center(), cur_box.get_extent()
    # print(cur_extent)

    bbx_size = max(cur_extent) + bbx_pad
    print("Init bbx size (m):", bbx_size)
    
    if bbx_size > max_bbx_size:
        print("Too large bbx, could not be a valid object, skip")
        valid_flag = False
    if bbx_size < min_bbx_size:
        print("Too small bbx, could not be a valid object, skip")
        valid_flag = False
    
    init_rot_y_rad = 0. # initial guess of the rotation around y axis, clockwise as postive

    max_init_rot = max_init_rot_deg/180.*math.pi

    if valid_flag:
        cur_center[1] += ((bbx_size - cur_extent[1])*0.5) # get the translation initial guess
        if cur_extent[1] == max(cur_extent): # then may due to noise, further shift the center behind a bit
            cur_center[1] += 0.01
        # get rotation inital guess by finding the support for the peduncle
        box_bg_min = [cur_center[0]-0.6*bbx_size, cur_center[1]-0.8*bbx_size, cur_center[2]+0.2*bbx_size]
        box_bg_max = [cur_center[0]+0.6*bbx_size, cur_center[1]+1.0*bbx_size, cur_center[2]+1.2*bbx_size]
        box_bg = o3d.geometry.AxisAlignedBoundingBox(box_bg_min, box_bg_max)
        bg_pcd_crop = copy.deepcopy(bg_pcd)
        bg_pcd_crop = bg_pcd_crop.crop(box_bg)
        if len(bg_pcd_crop.points) > min_nearby_bg_pts:
            bg_points_shift = np.asarray(bg_pcd_crop.points) - cur_center
            rot_vec = np.mean(bg_points_shift, 0)
            init_rot_y_rad = 0.5*math.pi - np.arctan2(rot_vec[2], rot_vec[0]) 
            init_rot_y_rad = max(min(init_rot_y_rad, max_init_rot), -max_init_rot) # limited to [-45, 45] deg
        print("Init rot around y axis (deg):", init_rot_y_rad * 180. / math.pi)
    
    return cur_center, init_rot_y_rad, bbx_size, valid_flag
        

def get_deg_between_vectors(v1, v2):
    # Calculate the dot product of the two vectors
    dot_product = np.dot(v1, v2)

    # Calculate the magnitudes of the two vectors
    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)

    # Calculate the cosine of the angle between the two vectors
    cosine_angle = dot_product / (m1 * m2)

    # Calculate the angle between the two vectors in radians
    angle = np.arccos(cosine_angle)

    # Convert the angle from radians to degrees
    angle_degrees = np.degrees(angle)

    return angle_degrees


def set_view(vis, dist=100., theta=np.pi/6.):
    """
    :param vis: o3d visualizer
    :param dist: eye-to-world distance, assume eye is looking at world origin
    :param theta: tilt-angle around x-axis of world coordinate
    """
    vis_ctr = vis.get_view_control()
    cam = vis_ctr.convert_to_pinhole_camera_parameters()
    # world to eye
    T = np.array([[1., 0., 0., 0.],
                  [0., np.cos(theta), -np.sin(theta), 0.],
                  [0., np.sin(theta), np.cos(theta), dist],
                  [0., 0., 0., 1.]])

    cam.extrinsic = T
    vis_ctr.convert_from_pinhole_camera_parameters(cam)


def read_calib_file(filepath):
    """Read in a KITTI calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            if line == "\n":
                break
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


class ForceKeyErrorDict(Dict):
    def __missing__(self, key):
        raise KeyError(key)


def get_configs(cfg_file):
    with open(cfg_file) as f:
        cfg_dict = json.load(f)
    return ForceKeyErrorDict(**cfg_dict)


def get_decoder(configs):
    return config_decoder(configs.DeepSDF_DIR, configs.checkpoint) # contain latent codes, model parameters, optimizer parameters

def get_init_latent_code(configs):
    return load_latent_vectors(configs.DeepSDF_DIR, configs.checkpoint)


def create_voxel_grid(vol_dim=128):
    # in the [-1, 1] cube
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (vol_dim - 1)

    overall_index = torch.arange(0, vol_dim ** 3, 1, out=torch.LongTensor())
    values = torch.zeros(vol_dim ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    values[:, 2] = overall_index % vol_dim
    values[:, 1] = (overall_index.long() / vol_dim) % vol_dim
    values[:, 0] = ((overall_index.long() / vol_dim) / vol_dim) % vol_dim

    # transform first 3 columns
    # to be the x, y, z coordinate
    values[:, 0] = (values[:, 0] * voxel_size) + voxel_origin[2]
    values[:, 1] = (values[:, 1] * voxel_size) + voxel_origin[1]
    values[:, 2] = (values[:, 2] * voxel_size) + voxel_origin[0]

    return values


def convert_sdf_voxels_to_mesh(pytorch_3d_sdf_tensor, cube_radius):
    """
    Convert sdf samples to mesh
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :return vertices and faces of the mesh
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.cpu().detach().numpy()
    voxels_dim = numpy_3d_sdf_tensor.shape[0]
    voxel_size = 2.0 / (voxels_dim - 1)
    verts, faces, normals, values = measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )
    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    voxel_grid_origin = np.array([-1., -1., -1.])
    verts[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    verts[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    verts[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    verts*=cube_radius

    return verts, faces


def write_mesh_to_ply(v, f, ply_filename_out):
    # try writing to the ply file

    num_verts = v.shape[0]
    num_faces = f.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(v[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((f[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)


def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()

# set up weight and bias
def setup_wandb():
    print("Weight & Bias logging option is on. Disable it by setting  wandb_vis_on: False  in the config file.")
    username = getpass.getuser()
    # print(username)
    wandb_key_path = username + "_wandb.key"
    if not os.path.exists(wandb_key_path):
        wandb_key = input(
            "[You need to firstly setup and login wandb] Please enter your wandb key (https://wandb.ai/authorize):"
        )
        with open(wandb_key_path, "w") as fh:
            fh.write(wandb_key)
    else:
        print("wandb key already set")
    os.system('export WANDB_API_KEY=$(cat "' + wandb_key_path + '")')


def set_random_seed(seed):
    o3d.utility.random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed) 
