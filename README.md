# netadapt-x-flower
A federated NAS study

### Build Docker
```bash
docker build -t pynode -f Dockerfile.python .
```

### Basic Run with Binding
```bash
docker run -it --name gocker -v /home/goktug/Desktop/thesis/netadapt-fl-docker/models/:/app/models/ pynode /bin/bash
```

### Build and run FL-NETADAPT
```bash
sudop python core_up.py
```