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
        kernel_type="rbf",
        C=1.0,
        degree=3,  # only for polynomial kernel
    ):
        # set kernel and C
        self.kernel_type = kernel_type
        self.C = C
        # set positive and negative batch
        self.positive_samples = positive_batch[0]
        self.negative_samples = negative_batch[0]

    def solve_svm(self):
        # Assign labels: 1 for positive samples, -1 for negative samples
        positive_labels = np.ones(len(self.positive_samples))
        negative_labels = -np.ones(len(self.negative_samples))
        # Combine positive and negative samples into X
        X = np.vstack((self.positive_samples, self.negative_samples))
        # Combine positive and negative labels into Y
        Y = np.hstack((positive_labels, negative_labels))
        # Train SVM
        model = SVC(kernel=self.kernel_type, C=self.C)
        model.fit(X, Y)
        # Access dual coefficients and support vectors
        dual_coefs = model.dual_coef_[0]
        # Flatten the matrix (shape is (1, n_support_vectors))
        support_indices = model.support_
        support_vectors = model.support_vectors_
        # Map dual coefficients to all samples
        full_dual_coefs = np.zeros(len(X))
        full_dual_coefs[support_indices] = dual_coefs
        return full_dual_coefs

    def compute_margain(self):
        full_dual_coefs = self.solve_svm()
        kernel_matrix = pairwise_kernels(
            X=np.vstack((self.positive_samples, self.negative_samples)),
            metric=self.kernel_type,
        )
        return 1 / math.sqrt(
            np.dot(np.dot(full_dual_coefs, kernel_matrix), full_dual_coefs.T)
        )
