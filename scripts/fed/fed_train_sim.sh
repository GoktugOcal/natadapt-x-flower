for alpha in 0.3 0.5; do
    str_alpha=$(echo "$alpha" | tr -d '.')
    python fed_train_sim.py \
        -pf ./projects/define_pretrained_fed_sim_NIID_alpha${str_alpha}/ \
        -m alexnet.pth.tar \
        -nc 140 \
        -nr 50 \
        --fine_tuning_epochs 10 \
        -niid noniid \
        -b - \
        -p dir \
        --alpha $alpha \
        --epochs 200 \
        --arch alexnet_reduced
done


