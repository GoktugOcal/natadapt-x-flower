python fed_train_sim.py \
    -pf ./projects/pretrained_model/cifar100_predefined_a03_56c_fedavg/ \
    -dp ./data/32_Cifar100_NIID_56c_a03 \
    -m alexnet.pth.tar \
    -dn cifar100 \
    -c 100 \
    -nc 56 \
    -nr 300 \
    --fine_tuning_epochs 50 \
    --epochs 200 \
    --lr 0.0001 \
    --arch alexnet_reduced \
    --client_selection