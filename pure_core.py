import logging
import time
import os
import argparse

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

WORKING_PATH = "models/alexnet/fed/pure-test-0"
MAX_ITER = 5
NO_CLIENTS = 10
CPU_PER_CLIENT = 8
MEM_LIMIT_PER_CLIENT = "4g"

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
        options.binds = [(os.getcwd() + "/models", "/app/models")]
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
            f"-nc {NO_CLIENTS}",
            wait=True
        )
        
        # time.sleep(5)  # Give server enough time to start

        for i, client in enumerate(clients):
            # DockerClient starts from 3 because 1 is switch and 2 is server
            client.host_cmd(
                f"docker exec -td DockerClient{3 + i} python client.py "
                f"{WORKING_PATH} "
                f"--server_ip {iface1_data.ip4} "
                f"--no {i} "
            )
        
        time.sleep(5)

        temp_line = ""
        while(True):
            with open(os.path.join(WORKING_PATH,"logs.txt"), "r") as f:
                last_line = f.readlines()[-1]
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
        
    finally:
        coreemu.shutdown()

if __name__ == "__main__":
    main()
    print("DONE.")