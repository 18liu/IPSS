import json
import torch
""" configuration json """
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

        
def config():
    config = Config({

        "dataset_name": "JUFE-10K",

        # optimization
        "batch_size": 4,
        "learning_rate": 1e-4,
        "weight_decay": 5e-4,
        "epoch": 50,
        "val_freq": 1,
        "num_workers": 8,
        "T_max": 50,
        "eta_min": 0,

        
        # data
        "train_csv": "/mnt/10T/lzy/csvfiles/train.csv",
        "test_csv": "/mnt/10T/lzy/csvfiles/test.csv",

        # load & save checkpoint
        "model_name": "IPSS",
        "type_name": "FR",
        "ckpt_path": "/mnt/10T/lzy/2025_TCSVT/output/models/",  
        "log_path": "/mnt/10T/lzy/2025_TCSVT/output/log/",
        "log_file": ".log",
        'device': torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
    })

    return config
