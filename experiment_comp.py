import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import time

import torch
import os

import nets as models
import functions as fns
from non_iid_generator.customDataset import CustomDataset

from tqdm import tqdm
import json 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tester(key, test_path, data_path, no_clients):

    resources = []
    train_accs = []
    test_accs = []

    for iter_no in range(1,16):
        try:
            model = torch.load(os.path.join(test_path,f"master/iter_{iter_no}_best_model.pth.tar"), map_location=torch.device(DEVICE))
            network_def = fns.get_network_def_from_model(model, (3,32,32))
            resource = fns.compute_resource(network_def, "FLOPS")
            resources.append(resource)
            train_acc, test_acc = federated_eval(model, no_clients, data_path)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        except:
            resources.append(None)
            train_accs.append(None)
            test_accs.append(None)

    return {
        "resource" : resources,
        "train_acc" : train_accs,
        "test_acc" : test_accs
    }


# {
#     test_name : {
#         "resource" : [],
#         "train_acc" : [],
#         "test_acc" : []
#     }
# }


tests = [
    {
        "test_name": "fedavg_56c_basic",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_56c_3",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
        

    {
        "test_name": "fedavg_56c_network",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_network_optimized_a03_56c",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
        

    {
        "test_name": "fedavg_56c_optimized",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_56c",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
        

    {
        "test_name": "fedavg_112c_basic",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_112c_1",
        "no_clients": 112,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_112c_a03"
    },
        

    {
        "test_name": "fedavg_112c_network",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_network_optimized_a03_112c_1",
        "no_clients": 112,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_112c_a03"
    },
        

    {
        "test_name": "fedavg_112c_optimized",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_112c_1",
        "no_clients": 112,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_112c_a03"
    },
        

    {
        "test_name": "fedavg_210c_basic",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_210c_1",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },
        

    {
        "test_name": "fedavg_210c_network",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_network_optimized_a03_210c_1",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },
        

    {
        "test_name": "fedavg_210c_optimized",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_210c_1",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },
        

    {
        "test_name": "fedavg_210c_optimized_nr50",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_210c_nr50",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },
        

    {
        "test_name": "fedprox_56c_basic",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_basic_a03_56c_fedprox_1",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
        

    {
        "test_name": "fedprox_56c_network",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_network_optimized_a03_56c_fedprox_1",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
        

    {
        "test_name": "fedprox_56c_optimized",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_optimized_a03_56c_fedprox_1",
        "no_clients": 56,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/32_Cifar10_NIID_56c_a03"
    },
        

    {
        "test_name": "fedprox_210c_basic",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_basic_a03_210c_fedprox_1",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },
        

    {
        "test_name": "fedprox_210c_network",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_network_optimized_a03_210c_fedprox_1",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },
        

    {
        "test_name": "fedprox_210c_optimized",
        "test_path": "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_optimized_a03_210c_fedprox_1",
        "no_clients": 210,
        "data_path": "/users/goktug.ocal/thesis/netadapt-x-flower/data/alpha/Cifar10_NIID_210c_a03"
    },
]



# tests = {
#     "fedavg_56c_basic" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_56c_3",
#     "fedavg_56c_network" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_network_optimized_a03_56c",
#     "fedavg_56c_optimized" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_56c",

#     "fedavg_112c_basic" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_112c_1",
#     "fedavg_112c_network" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_network_optimized_a03_112c_1",
#     "fedavg_112c_optimized" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_112c_1",

#     "fedavg_210c_basic" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_basic_a03_210c_1",
#     "fedavg_210c_network" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_network_optimized_a03_210c_1",
#     "fedavg_210c_optimized" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_210c_1",

#     "fedavg_210c_optimized_nr50" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fed_master_optimized_a03_210c_nr50",
    
#     "fedprox_56c_basic" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_basic_a03_56c_fedprox_1",
#     "fedprox_56c_network" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_network_optimized_a03_56c_fedprox_1",
#     "fedprox_56c_optimized" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_optimized_a03_56c_fedprox_1",

#     "fedprox_210c_basic" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_basic_a03_210c_fedprox_1",
#     "fedprox_210c_network" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_network_optimized_a03_210c_fedprox_1",
#     "fedprox_210c_optimized" : "/users/goktug.ocal/thesis/netadapt-x-flower/projects/fed_master/fedprox/fed_master_optimized_a03_210c_fedprox_1",
# }


test_results = {

}
for test in tests:

    print(test["test_name"])

    test_results[test["test_name"]] = tester(
        test["test_name"],
        test["test_path"],
        test["data_path"],
        test["no_clients"],
        )
    
    print("ok.")

with open("../projects/fed_master/comp.json","w") as f:
    json.dump(test_results,f)

print("DONE...")