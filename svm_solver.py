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

    def compute_margin(self):
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
        # Extract support vectors and dual coefficients
        support_vectors = model.support_vectors_
        dual_coefs = model.dual_coef_[0]  # Shape is (1, n_support_vectors)
        # Kernel parameters for pairwise computation
        kernel_params = {
            key: self.svm_params[key]
            for key in ["gamma", "degree", "coef0"]
            if key in self.svm_params
        }
        # Compute the kernel matrix for the support vectors only
        kernel_matrix = pairwise_kernels(
            X=support_vectors,
            metric=self.svm_params["kernel"],
            **kernel_params,
        )
        # Compute the norm of the weight vector in feature space
        w_norm_sq = np.dot(np.dot(dual_coefs, kernel_matrix), dual_coefs.T)
        # Calculate and return the margin
        if w_norm_sq > 0:
            return 1 / math.sqrt(w_norm_sq)
        else:
            print(f"Suspicious norm squared: {w_norm_sq}")
            return -1
