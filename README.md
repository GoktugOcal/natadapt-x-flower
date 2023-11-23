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


docker run --rm -it --cpus=4 --name gocker -v /home/goktug.ocal/thesis/netadapt-x-flower/logs/:/app/logs/ -v /home/goktug.ocal/thesis/netadapt-x-flower/data/:/app/data/ pynode /bin/bash


docker run --rm -it \
    -v /home/goktug/Desktop/thesis/netadapt-x-flower/models/:/app/models/ \
    -v /home/goktug/Desktop/thesis/netadapt-x-flower/data/:/app/data/ \
    -v /home/goktug/Desktop/thesis/netadapt-x-flower/projects/:/app/projects/ \
    pynode \
    /bin/bash


docker run -d --rm --cpus=4 --name gocker -v /home/goktug.ocal/thesis/netadapt-x-flower/logs/:/app/logs/ -v /home/goktug.ocal/thesis/netadapt-x-flower/data/:/app/data/ pynode python load_test.py


sudop nohup python pure_core.py > logs/core_NIID_a1_20c_alexnet.log &