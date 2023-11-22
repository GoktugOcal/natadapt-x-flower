for alpha in 0.5; do
    str_alpha=$(echo "$alpha" | tr -d '.')
    python fed_train_sim.py \
        -pf ./projects/test_1_fed_sim_NIID_alpha${str_alpha}_no_pretrained_50/ \
        -m alexnet.pth.tar \
        -nc 20 \
        -nr 50 \
        --fine_tuning_epochs 50 \
        -niid noniid \
        -b - \
        -p dir \
        --alpha $alpha \
        --epochs 50 \
        --arch alexnet
done


