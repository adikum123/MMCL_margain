import argparse
import math
import pickle
from itertools import product

import numpy as np
import torch
import tqdm
from sklearn.metrics.pairwise import pairwise_kernels
from thundersvm import SVC


class SVMSolver:

    def __init__(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

    @staticmethod
    def get_svm_params():
        # Define the parameter grid
        kernel_types = ["linear", "rbf", "poly", "sigmoid"]
        C_values = [0.1, 1, 10, 100]
        gamma_values = [0.001, 0.01, 0.1]
        degrees = [2, 3, 4, 10]
        coef0_values = [0, 0.5, 1]
        params = []
        for kernel in kernel_types:
            if kernel == "linear":
                for C in C_values:
                    params.append({"kernel": kernel, "C": C})
            elif kernel == "rbf":
                for C, gamma in product(C_values, gamma_values):
                    params.append({"kernel": kernel, "C": C, "gamma": gamma})
            elif kernel == "poly":
                for C, gamma, degree, coef0 in product(
                    C_values, gamma_values, degrees, coef0_values
                ):
                    params.append(
                        {
                            "kernel": kernel,
                            "C": C,
                            "gamma": gamma,
                            "degree": degree,
                            "coef0": coef0,
                        }
                    )
            elif kernel == "sigmoid":
                for C, gamma, coef0 in product(C_values, gamma_values, coef0_values):
                    params.append(
                        {"kernel": kernel, "C": C, "gamma": gamma, "coef0": coef0}
                    )
        return params

    def compute_margin(self):
        # Combine all of training data
        all_batches = [x[0] for x in self.train_loader]
        train_data = np.vstack(all_batches)
        # iterate through all svm params and compute necessary margins
        result = dict()
        for params in SVMSolver.get_svm_params():
            params_key = str(params)
            print(f"Computing margins for: {params_key}")
            result[params_key] = []
            for test_point in tqdm.tqdm(self.test_loader):
                X = np.vstack((test_point[0], train_data))
                Y = np.vstack(
                    (np.ones((1, 1)), -np.ones((len(self.train_loader.dataset), 1)))
                )
                # Train SVM using the stored parameters
                model = SVC(**params)
                model.fit(X, Y.ravel())
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
