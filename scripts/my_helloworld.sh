python federated_master.py models/helloworld/fed 3 32 32 \
    -gp 0 1 2 -mi 1 -bur 0.25 -rt FLOPS \
    -irr 0.025 -rd 1.0 -lr 0.001 -st 5 \
    -im models/helloworld/model_torch2.pth.tar \
    -lt models/helloworld/lut.pkl  -dp data/ \
    --arch helloworld -si 1