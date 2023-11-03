from dotenv import load_dotenv
load_dotenv()

from argparse import ArgumentParser
import os
import io
import base64

import json
import pickle
import time
import torch
from shutil import copyfile
import subprocess
import sys
import warnings
import common

from server import flower_server_execute, NetStrategy

from copy import deepcopy
from collections import OrderedDict

import logging

_MASTER_FOLDER_FILENAME = 'master'
_WORKER_FOLDER_FILENAME = 'worker'
_CLIENT_FOLDER_FILENAME = 'client'
_SERVER_FOLDER_FILENAME = 'server'

def load_data(train_dataset_path, test_dataset_path):
    """Load CIFAR-10 (training and test set)."""

    batch_size = 128

    train_data = pickle.load(open(train_dataset_path, "rb"))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True)

    test_data = pickle.load(open(test_dataset_path, "rb"))
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True)
    
    return train_loader, test_loader

def test(net, testloader):
    _NUM_CLASSES = 10
    """Validate the model on the test set."""
    criterion = torch.nn.BCEWithLogitsLoss()
    correct, loss = 0, 0.0
    latency_measurements = []
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            
            target_onehot = torch.FloatTensor(labels.shape[0], _NUM_CLASSES)
            target_onehot.zero_()
            target_onehot.scatter_(1, labels, 1)
            labels = labels.to(DEVICE)
            
            s = time()
            outputs = net(images.to(DEVICE))
            labels_one_hot = target_onehot.to(DEVICE)
            latency_measurements.append(time() - s)

            loss += criterion(outputs, labels_one_hot).item()
            correct += (torch.max(outputs.data, 1)[1] == labels.squeeze_(1)).sum().item()

    accuracy = correct / len(testloader.dataset)
    print("Test Accuracy :", round(accuracy, 3), "| Total no samples :", len(testloader.dataset))
    return loss, accuracy, latency_measurements

def federated_eval(model, args):

    train_dataset_path = f"./data/Cifar10/server/train.pkl"
    test_dataset_path = f"./data/Cifar10/server/test.pkl"
    trainloader, testloader = load_data(train_dataset_path, test_dataset_path)
    _, accuracy, _ = test(net=model,testloader=testloader)
    logging.info(f"Server | Acc: {round(accuracy, 2)}")

    for client_id in range(args.nc):
        train_dataset_path = f"./data/Cifar10/train/{client_id}.pkl"
        test_dataset_path = f"./data/Cifar10/test/{client_id}.pkl"
        trainloader, testloader = load_data(train_dataset_path, test_dataset_path)
        _, accuracy, _ = test(net=model,testloader=testloader)
        logging.info(f"Client {client_id} | Acc: {round(accuracy, 2)}")



def folder_things(args):
    master_folder = os.path.join(args.working_folder, _MASTER_FOLDER_FILENAME)
    worker_folder = os.path.join(args.working_folder, _WORKER_FOLDER_FILENAME)
    client_folder = os.path.join(args.working_folder, _CLIENT_FOLDER_FILENAME)
    server_folder = os.path.join(args.working_folder, _SERVER_FOLDER_FILENAME)
    # Create the folder structure.
    if not os.path.exists(args.working_folder):
        os.makedirs(args.working_folder)
        print('Create directory', args.working_folder)
    if not os.path.exists(master_folder):
        os.mkdir(master_folder)
        print('Create directory', master_folder)
    elif os.listdir(master_folder):
        errMsg = 'Find previous files in the master directory {}. Please use `--resume` or delete those files'.format(master_folder)
        raise ValueError(errMsg)
        
    if not os.path.exists(worker_folder):
        os.mkdir(worker_folder)
        print('Create directory', worker_folder)
    elif os.listdir(worker_folder):
        errMsg = 'Find previous files in the worker directory {}. Please use `--resume` or delete those files'.format(worker_folder)
        raise ValueError(errMsg)

    if not os.path.exists(server_folder):
        os.mkdir(server_folder)
        print('Create directory', server_folder)
    elif os.listdir(server_folder):
        errMsg = 'Find previous files in the server directory {}. Please use `--resume` or delete those files'.format(server_folder)
        raise ValueError(errMsg)
    
    for cn in range(args.nc):
        client_folder_name = os.path.join(args.working_folder, _CLIENT_FOLDER_FILENAME + "_" + str(cn))
        if not os.path.exists(client_folder_name):
            os.mkdir(client_folder_name)
            print('Create directory', client_folder_name)
            with open(os.path.join(client_folder_name,"logs.txt"), "w") as f:
                f.write("Iteration,Block,Round,Accuracy,Loss\n")
        elif os.listdir(client_folder_name):
            errMsg = 'Find previous files in the client directory {}. Please use `--resume` or delete those files'.format(client_folder_name)
            raise ValueError(errMsg)



if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument('working_folder', type=str,
                            help='Root folder where models, related files and history information are saved.')
    arg_parser.add_argument('-nc', '--nc', type=int,
                            help="Number of clients for federated learning.")
    args = arg_parser.parse_args()
    
    folder_things(args)

    logfilename = os.path.join(args.working_folder,"logs.txt")
    os.makedirs(os.path.dirname(logfilename), exist_ok=True)
    logging.basicConfig(
        filename=logfilename,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    try:
        no_clients = args.nc

        logging.info("Model")
        model = torch.load("models/alexnet/model_cuda.pth.tar", map_location=torch.device('cpu'))
        logging.info("Model loaded")

        ### TORCHSCRIPT
        scripted_model = torch.jit.script(model)

        buffer = io.BytesIO()
        torch.jit.save(scripted_model, buffer)
        model_bytes = buffer.getvalue()
        logging.info("Size of the encoded model : %d", len(model_bytes))
        buffer.close()

        print("########## FLOWER ##########")
        netadapt_info = {
            "netadapt_iteration" : 0,
            "block_id" : 0
        }
        strategy = NetStrategy(
            model_bytes,
            netadapt_info,
            min_fit_clients = no_clients,
            min_available_clients = no_clients,
            min_evaluate_clients = no_clients
            )
        logging.info("> Strategy defined")
        hist = flower_server_execute(
            strategy=strategy,
            no_rounds=10)
        logging.info("> Server Closed")
        print("########## FLOWER ##########")

        fine_tuned_model = deepcopy(simplified_model)
        params_dict = zip(fine_tuned_model.state_dict().keys(), strategy.parameters_aggregated)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        fine_tuned_model.load_state_dict(state_dict, strict=True)

        federated_eval(fine_tuned_model, args)

    except Exception as e:
        logging.error(e)