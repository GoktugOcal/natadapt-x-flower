python test_fed_master.py \
    projects/fed_master/test_fed_master\
    3 32 32 \
    -im projects/define_pretrained_fed_sim_NIID_alpha03/alexnet.pth.tar -gp 0 \
    -mi 15 -bur 0.25 -rt FLOPS -irr 0.025 -rd 0.96 \
    -lr 0.001 -st 500 \
    -dp ./data/32_Cifar10_NIID_56c_a03 \
    --arch alexnet_reduced \
    -nc 56 \
    -nr 20 \
    --fine_tuning_epochs 50
    # --client_selection