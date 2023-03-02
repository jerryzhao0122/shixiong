
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=noniid --update_type=weight --dataset=cifar10 --client_bs=1024 --client_lr=0.1 --model=alexnet --name=R_Krumavg_v2_replace_noniid_10 --attack=True --attack_type=replace --attack_num=10 --attack_replace_round=250 --robust_agg=True --robust_agg_type=krum 
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=noniid --update_type=weight --dataset=cifar10 --client_bs=1024 --client_lr=0.1 --model=alexnet --name=R_Krumavg_v2_replace_noniid_12 --attack=True --attack_type=replace --attack_num=12 --attack_replace_round=250 --robust_agg=True --robust_agg_type=krum
python FL_robust.py --round=300 --client_ep=2 --data_distributed=noniid --update_type=weight --dataset=cifar10 --client_bs=1024 --client_lr=0.1 --model=alexnet --name=R_Krumavg_v2_replace_noniid_16 --attack=True --attack_type=replace --attack_num=16 --attack_replace_round=250 --robust_agg=True --robust_agg_type=krum
python FL_robust.py --round=300 --client_ep=2 --data_distributed=noniid --update_type=weight --dataset=cifar10 --client_bs=1024 --client_lr=0.1 --model=alexnet --name=R_Krumavg_v2_replace_noniid_18 --attack=True --attack_type=replace --attack_num=18 --attack_replace_round=250 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=1000 --data_distributed=noniid --update_type=direction --dataset=cifar10 --client_bs=1024 --model=alexnet --majority_vote=True --name=R_MajorityVote_v2_replace_noniid_16 --attack=True --attack_type=replace --attack_num=16 --attack_replace_round=800