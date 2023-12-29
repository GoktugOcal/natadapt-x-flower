python fed_train_sim.py \
    -pf ./projects/pretrained_model/predefined_a03_56c_fedavg/ \
    -dp ./data/32_Cifar10_NIID_56c_a03 \
    -m alexnet.pth.tar \
    -nc 56 \
    -nr 300 \
    --fine_tuning_epochs 50 \
    --epochs 200 \
    --arch alexnet_reduced \
    --client_selection
