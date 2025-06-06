from torch import nn
from torch.optim import Adam

configs = {
    "default": {
        "model": {
            "type": "regressor",
            "params": {"hidden_units": [512, 256]},
        },
        "training": {
            "criterion": nn.MSELoss,
            "optimizer": {
                "class": Adam,
                "params": {"lr": 0.001, "weight_decay": 0.001},
            },
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 10,
            "feature_aggregation": False,
        },
        "evaluation": {
            "criterion": nn.MSELoss,
            "feature_aggregation": False,
            "batch_size": 32,
        },
    },
    "regression_MLP": {},
    "regression_SLP": {
        "model": {
            "params": {"hidden_units": []},
        },
    },
}
