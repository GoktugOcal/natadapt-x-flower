python fed_train_sim.py \
    -pf ./projects/pretrained_model/predefined_a03_56c_fedprox_001/ \
    -dp ./data/32_Cifar10_NIID_56c_a03 \
    -m alexnet.pth.tar \
    -c 10 \
    -nc 56 \
    -nr 300 \
    --fine_tuning_epochs 50 \
    --epochs 200 \
    --arch alexnet_reduced \
    --client_selection \
    --fedprox \
    --mu 0.01