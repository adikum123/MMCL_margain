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

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SSLEval(hparams=hparams, device=device)
model.prepare_data_new(train_size=1000, test_size=1)
train_loader, test_loader = model.dataloaders()
for test_batch in test_loader:
    for train_batch in train_loader:
        svm_solver = SVMSolver(
            positive_batch=test_batch,
            negative_batch=train_batch,
            kernel_type="linear",
        )
        print(svm_solver.compute_margain())
