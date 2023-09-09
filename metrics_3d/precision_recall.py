'''
 Collection of commonly used 3d reconstruction metrics
 By Federico Magistri
'''

import numpy as np
import scipy

from metrics_3d.metric import Metrics3D

class PrecisionRecall(Metrics3D):

    def __init__(self, min_t, max_t, num):
        self.thresholds = np.linspace(min_t, max_t, num)
        self.pr_dict = {t: [] for t in self.thresholds}
        self.re_dict = {t: [] for t in self.thresholds}
        self.f1_dict = {t: [] for t in self.thresholds}

    def update(self, gt, pt):
        if self.prediction_is_empty(pt):
            for t in self.thresholds:
                self.pr_dict[t].append(0)
                self.re_dict[t].append(0)
                self.f1_dict[t].append(0)
            return

        gt_pcd = self.convert_to_pcd(gt)
        pt_pcd = self.convert_to_pcd(pt)

        # precision: predicted --> ground truth
        dist_pt_2_gt = np.asarray(pt_pcd.compute_point_cloud_distance(gt_pcd))

        # recall: ground truth --> predicted
        dist_gt_2_pt = np.asarray(gt_pcd.compute_point_cloud_distance(pt_pcd))

        for t in self.thresholds:
            p = np.where(dist_pt_2_gt < t)[0]
            p = 100 / len(dist_pt_2_gt) * len(p)
            self.pr_dict[t].append(p)

            r = np.where(dist_gt_2_pt < t)[0]
            r = 100 / len(dist_gt_2_pt) * len(r)
            self.re_dict[t].append(r)

            # fscore
            if p == 0 or r == 0:
                f = 0
            else:
                f = 2 * p * r / (p + r)
            self.f1_dict[t].append(f)

    def reset(self):
        self.pr_dict = {t: [] for t in self.thresholds}
        self.re_dict = {t: [] for t in self.thresholds}
        self.f1_dict = {t: [] for t in self.thresholds}

    def compute_at_threshold(self, threshold):
        t = self.find_nearest_threshold(threshold)
        # print('computing metrics at threshold:', t)
        pr = sum(self.pr_dict[t]) / len(self.pr_dict[t])
        re = sum(self.re_dict[t]) / len(self.re_dict[t])
        f1 = sum(self.f1_dict[t]) / len(self.f1_dict[t])
        # print('precision: {}'.format(pr))
        # print('recall: {}'.format(re))
        # print('fscore: {}'.format(f1))
        return pr, re, f1, t

    def compute_auc(self):
        dx = self.thresholds[1] - self.thresholds[0]
        perfect_predictor = scipy.integrate.simpson(np.ones_like(self.thresholds), dx=dx)

        pr, re, f1 = self.compute_at_all_thresholds()

        pr_area = scipy.integrate.simpson(pr, dx=dx)
        norm_pr_area = pr_area / perfect_predictor

        re_area = scipy.integrate.simpson(re, dx=dx)
        norm_re_area = re_area / perfect_predictor

        f1_area = scipy.integrate.simpson(f1, dx=dx)
        norm_f1_area = f1_area / perfect_predictor

        # print('computing area under curve')
        # print('precision: {}'.format(norm_pr_area))
        # print('recall: {}'.format(norm_re_area))
        # print('fscore: {}'.format(norm_f1_area))

        return norm_pr_area, norm_re_area, norm_f1_area

    def compute_at_all_thresholds(self):
        pr = [sum(self.pr_dict[t]) / len(self.pr_dict[t]) for t in self.thresholds]
        re = [sum(self.re_dict[t]) / len(self.re_dict[t]) for t in self.thresholds]
        f1 = [sum(self.f1_dict[t]) / len(self.f1_dict[t]) for t in self.thresholds]
        return pr, re, f1

    def find_nearest_threshold(self, value):
        idx = (np.abs(self.thresholds - value)).argmin()
        return self.thresholds[idx]
