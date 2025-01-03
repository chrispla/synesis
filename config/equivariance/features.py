from torch import nn
from torch.optim import Adam

configs = {
    "default": {
        "model": {
            "type": "regressor",
            "params": {"hidden_units": [512, 512]},
        },
        "training": {
            "criterion": nn.MSELoss,
            "optimizer": {"class": Adam, "params": {"lr": 0.001, "weight_decay": 0.01}},
            "batch_size": 32,
            "num_epochs": 5,
            "patience": 3,
            "feature_aggregation": False,
        },
        "evaluation": {
            "criterion": nn.MSELoss,
            "feature_aggregation": False,
            "batch_size": 32,
        },
    },
}
