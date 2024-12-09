import argparse
import math
import pickle
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import SVC


class SVMSolver:
    def __init__(self, train_loader, test_loader, svm_params_list):
        self.positive_samples = positive_batch[0]
        self.negative_samples = negative_batch[0]
        self.svm_params_list = svm_params_list
        self.train_loader = train_loader
        self.test_loader = test_loader

    def compute_margin(self):
        # Combine all of training data
        all_batches = [x for x in self.train_loader]
        train_data = np.vstack(all_batches)
        # iterate through all svm params and compute necessary margins
        result = dict()
        for params in svm_params:
            params_key = str(params)
            result[params_key] = defaultdict(list)
            for test_point, label in self.test_loader:
                X = np.vstack((test_point, train_data))
                Y = np.vstack((np.ones(1), -np.ones(len(self.train_loader))))
                # Train SVM using the stored parameters
                model = SVC(**params)
                model.fit(X, Y)
                # Extract support vectors and dual coefficients
                support_vectors = model.support_vectors_
                dual_coefs = model.dual_coef_[0]
                kernel_params = {
                    key: params[key]
                    for key in ["gamma", "degree", "coef0"]
                    if key in params
                }
                # Compute the kernel matrix for the support vectors only and norm
                kernel_matrix = pairwise_kernels(
                    X=support_vectors,
                    metric=params["kernel"],
                    **kernel_params,
                )
                w_norm_sq = np.dot(np.dot(dual_coefs, kernel_matrix), dual_coefs.T)
                if w_norm_sq > 0:
                    result[params_key].append(1 / math.sqrt(w_norm_sq))
        with pickle.open("results_dict.pkl", "wb") as f:
            pickle.dump(result, f)
