import numpy as np
import pandas as pd
import json
import io
import os
import sys
import pickle
import math
import random

from pymoo.core.problem import ElementwiseProblem
from non_iid_generator.customDataset import CustomDataset

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize


# def EMD(Z_i, Z_global):
#     magnitude = lambda vector: math.sqrt(sum(pow(element, 2) for element in vector))
#     return magnitude(Z_i/magnitude(Z_i) - Z_global/magnitude(Z_global))

# def group_distribution(label_vectors, groups, no_groups):
#     group_dist = []
#     for gid in range(no_groups):
#         group_vector = label_vectors[groups == gid]
#         group_dist.append(np.sum(group_vector, axis = 0))
#     return group_dist

# def group_max_size(label_vectors, groups, no_groups):
#     group_max = []
#     for gid in range(no_groups):
#         group_vector = label_vectors[groups == gid]
#         group_max.append(np.amax(np.sum(group_vector, axis = 1)))
#     return group_max

# def emd_obj(group_dist):
#     Z_global = np.sum(group_dist, axis = 0)
#     return np.sum([EMD(group, Z_global) for group in group_dist])



# class MyProblem(ElementwiseProblem):

#     def __init__(self, main_label_vectors, no_clients, no_groups, no_classes):
#         self.NO_CLIENTS = no_clients
#         self.NO_GROUPS = no_groups
#         self.NO_CLASSES = no_classes

#         super().__init__(n_var=self.NO_CLIENTS,
#                          n_obj=2,
#                         #  n_eq_constr=NO_CLASSES,
#                          n_ieq_constr=self.NO_CLASSES,
#                          xl=np.ones(self.NO_CLIENTS) * 0,
#                          xu=np.ones(self.NO_CLIENTS) * self.NO_GROUPS,
#                          vtype=int)
#         self.main_label_vectors = main_label_vectors

#     def _evaluate(self, x, out, *args, **kwargs):
#         group_dist = group_distribution(self.main_label_vectors, x, self.NO_GROUPS)
#         f1 = emd_obj(group_dist)

#         try:
#             group_max = group_max_size(self.main_label_vectors, x, self.NO_GROUPS)
#             dif = np.amax(group_max) - np.amin(group_max)
#             f2 = dif
#         except:
#             f2 = 0

#         similarity = 12
#         hs = (np.unique(x, return_counts=True)[1] - similarity).tolist()
#         if len(hs) != self.NO_CLASSES: hs = np.zeros(self.NO_CLASSES) - 14

#         gs = (similarity - np.unique(x, return_counts=True)[1]).tolist()
#         if len(gs) != self.NO_CLASSES: gs = 14 - np.zeros(self.NO_CLASSES)
                
#         out["F"] = [f1, f2]
#         out["G"] = gs

# class ClientSelector():
#     def __init__(self, no_clients, no_groups, no_classes, dataset_path):
#         self.NO_CLIENTS = no_clients
#         self.NO_GROUPS = no_groups
#         self.NO_CLASSES = no_classes

#         conf = json.loads(open(os.path.join(dataset_path, "config.json"), "r").read())
#         data = [dict(zip(np.array(cli)[:,0], np.array(cli)[:,1])) for cli in conf["Size of samples for labels in clients"]]

#         main_label_vectors = np.zeros((self.NO_CLIENTS,self.NO_CLASSES))
#         for client_id in range(self.NO_CLIENTS):
#             for class_id in range(self.NO_CLASSES):
#                 if class_id in data[client_id].keys():
#                     main_label_vectors[client_id][class_id] = data[client_id][class_id]
#         self.main_label_vectors = main_label_vectors

#     def find_groups(self):

#         problem = MyProblem(
#             main_label_vectors = self.main_label_vectors,
#             no_clients = self.NO_CLIENTS,
#             no_groups = self.NO_GROUPS,
#             no_classes = self.NO_CLASSES
#         )

#         algorithm = NSGA2(
#             pop_size=40,
#             n_offsprings=10,
#             sampling=FloatRandomSampling(),
#             crossover=SBX(prob=0.1, eta=1, vtype=int),
#             mutation=PM(eta=1, vtype=int),
#             eliminate_duplicates=True
#         )

#         termination = get_termination("n_gen", 1000)

#         res = minimize(problem,
#                     algorithm,
#                     termination,
#                     seed=1,
#                     save_history=True,
#                     verbose=True)

#         self.groups_result = res.X

#         if len(self.groups_result) < 14: self.groups = self.groups_result[0]
#         else: self.groups = self.groups_result

#         return self.groups

#     def get_clients(self, group_no = 0):

#         return [idx for idx, item in enumerate(self.groups) if item == group_no]


class ClientSelector:
    def __init__(self, no_clients, no_groups, dataset_path):
        self.no_clients = no_clients
        self.no_groups = no_groups

        self.df = pd.read_csv(dataset_path)
        self.groupdf = self.df.copy()
    
    def random_grouping(self):
        groups = np.repeat(np.arange(self.no_groups),self.no_clients//self.no_groups)
        random.shuffle(groups)
        self.groupdf["group"] = groups
    
    def network_optimized_grouping(self, model_metadata):
        grp_no = 0
        replace_dict = {}
        for k, v in model_metadata.items():
            replace_dict[grp_no] = k
            # clients = bw_table[bw_table["group"] == grp_no]["client_id"].values.tolist()
            # model_metadata[k]["clients"] = clients
            grp_no +=1
        
        for idx, row in self.groupdf.iterrows():
            group_no = row.group
            self.groupdf.at[idx,"group"] = replace_dict[group_no]
        

    def get_clients(self, group_no = 0):
        selected_group = self.groupdf[self.groupdf["group"] == group_no]
        selected_clients = selected_group["client_id"].values.tolist()
        return selected_clients

if __name__ == "__main__":

    dataset_path = "./data/32_Cifar10_NIID_56c_a03/bws_groups.csv"
    no_clients=56
    no_groups=7

    client_selector = ClientSelector(no_clients, no_groups, dataset_path)
    client_selector.random_grouping()
    for gid in range(no_groups):
        print(client_selector.get_clients(group_no=gid))

#     dataset_path = "data/Cifar10_NIID_a05_140c"
#     no_clients = 140
#     no_groups = 10
#     no_classes = 10

#     client_selector = ClientSelector(no_clients, no_groups, no_classes, dataset_path)
#     client_selector.find_groups()
#     print(client_selector.get_clients())