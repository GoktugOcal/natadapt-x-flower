# netadapt-x-flower
A federated NAS study

### Build Docker
```bash
docker build -t pynode -f Dockerfile.python .
```

```bash
sudo iptables --policy FORWARD ACCEPT
```

### Basic Run with Binding
```bash
docker run -it --name gocker -v /home/goktug/Desktop/thesis/netadapt-fl-docker/models/:/app/models/ pynode /bin/bash
```

### Build and run FL-NETADAPT
```bash
sudop python core_up.py
```

```bash
source /home/goktug/python_envs/easyfl/bin/activate; cd /home/goktug/Desktop/thesis/netadapt-x-flower/
```

sudop nohup python pure_core.py > logs/core_NIID_a1_20c_alexnet.log &