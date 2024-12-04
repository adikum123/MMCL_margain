import argparse
import math

import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import SVC


class SVMSolver:
    def __init__(
        self,
        positive_batch,
        negative_batch,
        **svm_params,  # Accept any SVM parameters dynamically
    ):
        # Store positive and negative samples
        self.positive_samples = positive_batch[0]
        self.negative_samples = negative_batch[0]
        # Store SVM parameters (e.g., kernel, C, degree, gamma, etc.)
        self.svm_params = svm_params

    def solve_svm(self):
        # Assign labels: 1 for positive samples, -1 for negative samples
        positive_labels = np.ones(len(self.positive_samples))
        negative_labels = -np.ones(len(self.negative_samples))
        # Combine positive and negative samples into X
        X = np.vstack((self.positive_samples, self.negative_samples))
        # Combine positive and negative labels into Y
        Y = np.hstack((positive_labels, negative_labels))
        # Train SVM using the stored parameters
        model = SVC(**self.svm_params)
        model.fit(X, Y)
        # Access dual coefficients and support vectors
        dual_coefs = model.dual_coef_[0]  # Shape is (1, n_support_vectors)
        support_indices = model.support_
        support_vectors = model.support_vectors_
        # Map dual coefficients to all samples
        full_dual_coefs = np.zeros(len(X))
        full_dual_coefs[support_indices] = dual_coefs
        return full_dual_coefs

    def compute_margain(self):
        full_dual_coefs = self.solve_svm()
        # Compute the kernel matrix using the specified kernel
        kernel_params = {
            key: self.svm_params[key]
            for key in ["gamma", "degree", "coef0"]
            if key in self.svm_params
        }
        kernel_matrix = pairwise_kernels(
            X=np.vstack((self.positive_samples, self.negative_samples)),
            metric=self.svm_params["kernel"],
            **kernel_params,
        )
        w_norm_sq = np.dot(np.dot(full_dual_coefs, kernel_matrix), full_dual_coefs.T)
        if w_norm_sq > 0:
            return 1 / math.sqrt(w_norm_sq)
        print(f"Suspicious norm squared: {w_norm_sq}")
        return -1
