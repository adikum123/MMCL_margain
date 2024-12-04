import json
from itertools import product

import torch
import yaml

from models.ssl import SSLEval
from svm_solver import SVMSolver


class YamlObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                value = YamlObject(value)
            setattr(self, key, value)


def load_yaml_as_object(yaml_file):
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)
    return YamlObject(data)


def get_svm_params():
    # Define the parameter grid
    kernel_types = ["linear", "rbf", "poly", "sigmoid"]
    C_values = [0.1, 1, 10, 100]
    gamma_values = [0.001, 0.01, 0.1]
    degrees = [2, 3, 4, 10]  # Relevant for 'poly' kernel
    coef0_values = [0, 0.5, 1]  # Used in 'poly' and 'sigmoid' kernels
    # Generate all combinations of parameters
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


hparams = load_yaml_as_object("configs/imagenet100_eval.yaml")
setattr(hparams, "arch", "ResNet50")

arguments = {
    "arch": "linear",
    "finetune": False,
    "augmentation": "RandomResizedCrop",
    "scale_lower": 0.08,
    "aug": False,
}

for arg, value in arguments.items():
    setattr(hparams, arg, value)

output_file = "svm_results.txt"

with open(output_file, "w") as f:
    f.write(f"{'SVM Parameters':<50} | {'Computed Margin':<20}\n")
    f.write("-" * 70 + "\n")

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"========> Using device in test.py: {device}")
model = SSLEval(hparams=hparams, device=device)
model.prepare_data_new(train_size=1000, test_size=1)
train_loader, test_loader = model.dataloaders()
for test_batch in test_loader:
    for train_batch in train_loader:
        for svm_params in get_svm_params():
            svm_solver = SVMSolver(
                positive_batch=test_batch,
                negative_batch=train_batch,
                **svm_params,
            )
            margin = svm_solver.compute_margain()
            print(f"For params: {svm_params} margain is: {margin}")
            with open(output_file, "a") as f:
                f.write(f"{str(svm_params):<50} | {margin:<20.4f}\n")
