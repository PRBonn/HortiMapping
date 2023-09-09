'''
 Collection of commonly used 3d reconstruction metrics
 By Federico Magistri
'''

import numpy as np

from metrics_3d.metric import Metrics3D

# TODO: add the std
class ChamferDistance(Metrics3D):

    def __init__(self):
        self.cd_array = []

    def update(self, gt, pt):
        if self.prediction_is_empty(pt):
            self.cd_array.append(0)
            return

        gt_pcd = self.convert_to_pcd(gt)
        pt_pcd = self.convert_to_pcd(pt)
        dist_pt_2_gt = np.asarray(pt_pcd.compute_point_cloud_distance(gt_pcd))
        dist_gt_2_pt = np.asarray(gt_pcd.compute_point_cloud_distance(pt_pcd))
        d = (np.mean(dist_gt_2_pt) + np.mean(dist_pt_2_gt)) / 2
        self.cd_array.append(d)

    def reset(self):
        self.cd_array = []

    def compute(self):
        cd = sum(self.cd_array) / len(self.cd_array)
        # print('chamfer distance: {}'.format(cd))
        return cd
