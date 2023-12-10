import logging
import time
import os
import argparse
import numpy as np
import random

from core.emulator.coreemu import CoreEmu
from core.emulator.data import IpPrefixes
from core.emulator.enumerations import EventTypes
from core.nodes.base import CoreNode
from core.nodes.docker import DockerNode
from core.nodes.network import SwitchNode
from core.emulator.data import LinkOptions

from custom_nodes.dockerserver import DockerServer
from custom_nodes.dockerclient import DockerClient
# from fl_src import config

logging.basicConfig(level=logging.DEBUG)

user = os.getlogin()
if user == "goktug":
    #MODEL_PATH = "projects/define_pretrained_fed_sim_NIID_alpha03/alexnet.pth.tar"
    MODEL_PATH = "projects/test/test-5/worker/iter_1_block_0_model.pth.tar"
    DATASET_PATH = "32_Cifar10_NIID_56c_a03"
else:
    MODEL_PATH = "projects/test/test-5/master/iter_0_best_model.pth.tar"
    DATASET_PATH = "32_Cifar10_NIID_80c_a03"

WORKING_PATH = "projects/12dec/network_test_8c_iter_1_block_0/"
MAX_ITER = 5
NO_CLIENTS = 8
NO_ROUNDS = 1
CPU_PER_CLIENT = 8
MEM_LIMIT_PER_CLIENT = "6g"


# WEAK_NETWORK = np.arange(500_000, 2_500_000, 500_000)
# WEAK_NETWORK = np.array([2_000_000, 2_500_000])
WEAK_NETWORK = np.array([2_500_000, 3_500_000])
NORMAL_NETWORK = np.arange(8_000_000, 32_000_000, 1_000_000)
STRONG_NETWORK = np.arange(50_000_000, 100_000_000, 10_000_000)

client_networks = {
    0 : WEAK_NETWORK,
    1 : NORMAL_NETWORK,
    2 : STRONG_NETWORK,
    3 : WEAK_NETWORK,
    4 : NORMAL_NETWORK,
    5 : STRONG_NETWORK,
    6 : WEAK_NETWORK,
    7 : NORMAL_NETWORK,
    8 : STRONG_NETWORK,
    9 : WEAK_NETWORK,
    10 : NORMAL_NETWORK,
    11 : NORMAL_NETWORK,
    12 : STRONG_NETWORK,
    13 : WEAK_NETWORK,
    14 : NORMAL_NETWORK,
    15 : STRONG_NETWORK,
    16 : WEAK_NETWORK,
    17 : NORMAL_NETWORK,
    18 : STRONG_NETWORK,
    19 : WEAK_NETWORK,
}

def main():
    parser = argparse.ArgumentParser(description="Core")
    args = parser.parse_args()

    # create core session
    coreemu = CoreEmu()
    session = coreemu.create_session()
    session.set_state(EventTypes.CONFIGURATION_STATE)

    try:
        prefixes = IpPrefixes(ip4_prefix="10.83.0.0/16")
        # prefixes = IpPrefixes(ip4_prefix="10.0.0.0/24")

        # create switch
        switch = session.add_node(SwitchNode)

        # Server Node
        options = DockerNode.create_options()
        options.image = "pynode"
        options.binds = [
            (os.getcwd() + "/models", "/app/models"),
            (os.getcwd() + "/data", "/app/data"),
            (os.getcwd() + "/projects", "/app/projects")
            ]
        options.no_cpus = CPU_PER_CLIENT
        options.mem_limit = MEM_LIMIT_PER_CLIENT
        servernode = session.add_node(DockerServer, options=options)
        iface1_data = prefixes.create_iface(servernode)

        # Client Nodes
        clients = []
        client_ifaces = []
        for i in range(NO_CLIENTS):
            new_client = session.add_node(DockerClient, options=options)
            clients.append(new_client)
            client_ifaces.append(prefixes.create_iface(new_client))

        # add links
        session.add_link(servernode.id, switch.id, iface1_data)
        # link_options = LinkOptions(bandwidth=55_000_000)
        link_options = None
        if link_options:
            for client, iface in zip(clients, client_ifaces):
                session.add_link(client.id, switch.id, iface, options=link_options)
        else:
            for client, iface in zip(clients, client_ifaces):
                session.add_link(client.id, switch.id, iface)

        # =========== instantiate ===========
        session.instantiate()

        # Execute python scripts
        servernode.host_cmd(
            f"docker exec -td DockerServer2 python pure_server.py "
            f"{WORKING_PATH} "
            f"-nc {NO_CLIENTS} "
            f"-m {MODEL_PATH} "
            f"-nr {NO_ROUNDS}",
            wait=True
        )
        # models/alexnet/alexnet32_a03_server.pth.tar
        
        # time.sleep(5)  # Give server enough time to start

        for i, client in enumerate(clients):
            # DockerClient starts from 3 because 1 is switch and 2 is server
            client.host_cmd(
                f"docker exec -td DockerClient{3 + i} python client.py "
                f"{WORKING_PATH} "
                f"--server_ip {iface1_data.ip4} "
                f"-dn {DATASET_PATH} "
                f"--no {i} "
            )

            # 32_Cifar10_NIID_20c_a03 

        for client in clients:
            new_options = LinkOptions(
                bandwidth=random.choice(client_networks[client.id-3]),
                )
            session.update_link(switch.id, client.id, client.id-2, 0, options=new_options)
        
        time.sleep(10)
        temp_line = ""
        start = time.time()
        while(True):
            
            elapsed = time.time() - start
            if elapsed > 300:
                for client in clients:
                    new_options = LinkOptions(
                        bandwidth=random.choice(client_networks[client.id-3]),
                        )
                    session.update_link(switch.id, client.id, client.id-2, 0, options=new_options)
                start = time.time()

            with open(os.path.join(WORKING_PATH,"logs_debug.txt"), "r") as f:
                last_line = f.readlines()[-1]
                # logging.debug(last_line)
                if temp_line != last_line:
                    logging.debug(last_line)
                    temp_line = last_line
                if "DONE" in last_line:
                    raise KeyboardInterrupt

        # servernode.host_cmd(
        #     f"docker exec -td DockerServer2 python pure_server.py "
        #     f"{WORKING_PATH} "
        #     f"-nc {NO_CLIENTS}",
        #     wait=False
        # )


        # Wait for the last client process to end
        # clients[-1].host_cmd(
        #     f"docker exec -t DockerClient{2 + len(clients)} python client.py "
        #     f"{WORKING_PATH} "
        #     f"--server_ip {iface1_data.ip4} "
        #     f"--no {len(clients) - 1} ",
        #     wait=True,
        # )

    except KeyboardInterrupt:
        coreemu.shutdown()
    
    except Exception as e:
        logging.debug(e, exc_info=True)
        print(">>>>>>>>>>>>>>>", e)
        coreemu.shutdown()
        
    finally:
        coreemu.shutdown()

if __name__ == "__main__":
    main()
    print("DONE.")