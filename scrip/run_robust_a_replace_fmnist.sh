
# FedAvg iid
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_iid_2 --attack=True --attack_type=replace --attack_num=2 --attack_replace_round=150
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_iid_4 --attack=True --attack_type=replace --attack_num=4 --attack_replace_round=150
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_iid_6 --attack=True --attack_type=replace --attack_num=6 --attack_replace_round=150
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_iid_8 --attack=True --attack_type=replace --attack_num=8 --attack_replace_round=150
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_iid_10 --attack=True --attack_type=replace --attack_num=10 --attack_replace_round=150
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_iid_12 --attack=True --attack_type=replace --attack_num=12 --attack_replace_round=150
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_iid_14 --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=150
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_iid_16 --attack=True --attack_type=replace --attack_num=16 --attack_replace_round=150
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_iid_18 --attack=True --attack_type=replace --attack_num=18 --attack_replace_round=150


# FedAVG-Krum iid
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_iid_2 --attack=True --attack_type=replace --attack_num=2 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_iid_4 --attack=True --attack_type=replace --attack_num=4 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_iid_6 --attack=True --attack_type=replace --attack_num=6 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_iid_8 --attack=True --attack_type=replace --attack_num=8 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_iid_10 --attack=True --attack_type=replace --attack_num=10 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum 
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_iid_12 --attack=True --attack_type=replace --attack_num=12 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_iid_14 --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_iid_16 --attack=True --attack_type=replace --attack_num=16 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_iid_18 --attack=True --attack_type=replace --attack_num=18 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum


# FedSIGN-DC iid
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_iid_2 --attack=True --attack_type=replace --attack_num=2 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_iid_4 --attack=True --attack_type=replace --attack_num=4 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_iid_6 --attack=True --attack_type=replace --attack_num=6 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_iid_8 --attack=True --attack_type=replace --attack_num=8 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_iid_10 --attack=True --attack_type=replace --attack_num=10 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_iid_12 --attack=True --attack_type=replace --attack_num=12 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_iid_14 --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_iid_16 --attack=True --attack_type=replace --attack_num=16 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
python FL_robust.py --round=200 --client_ep=5 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_iid_18 --attack=True --attack_type=replace --attack_num=18 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 


# Majority-Vote noniid
python FL_robust.py --round=4000 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_noniid_2 --attack=True --attack_type=replace --attack_num=2 --attack_replace_round=3000
python FL_robust.py --round=4000 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_noniid_4 --attack=True --attack_type=replace --attack_num=4 --attack_replace_round=3000
python FL_robust.py --round=4000 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_noniid_6 --attack=True --attack_type=replace --attack_num=6 --attack_replace_round=3000
python FL_robust.py --round=4000 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_noniid_8 --attack=True --attack_type=replace --attack_num=8 --attack_replace_round=3000
python FL_robust.py --round=4000 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_noniid_10 --attack=True --attack_type=replace --attack_num=10 --attack_replace_round=3000
python FL_robust.py --round=4000 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_noniid_12 --attack=True --attack_type=replace --attack_num=12 --attack_replace_round=3000
python FL_robust.py --round=4000 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_noniid_14 --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=3000
python FL_robust.py --round=4000 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_noniid_16 --attack=True --attack_type=replace --attack_num=16 --attack_replace_round=3000
python FL_robust.py --round=4000 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_noniid_18 --attack=True --attack_type=replace --attack_num=18 --attack_replace_round=3000

# Majority-Vote iid
python FL_robust.py --round=4000 --data_distributed=iid --client_ep=5 --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_iid_2 --attack=True --attack_type=replace --attack_num=2 --attack_replace_round=3000
python FL_robust.py --round=4000 --data_distributed=iid --client_ep=5 --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_iid_4 --attack=True --attack_type=replace --attack_num=4 --attack_replace_round=3000
python FL_robust.py --round=4000 --data_distributed=iid --client_ep=5 --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_iid_6 --attack=True --attack_type=replace --attack_num=6 --attack_replace_round=3000
python FL_robust.py --round=4000 --data_distributed=iid --client_ep=5 --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_iid_8 --attack=True --attack_type=replace --attack_num=8 --attack_replace_round=3000
python FL_robust.py --round=4000 --data_distributed=iid --client_ep=5 --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_iid_10 --attack=True --attack_type=replace --attack_num=10 --attack_replace_round=3000
python FL_robust.py --round=4000 --data_distributed=iid --client_ep=5 --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_iid_12 --attack=True --attack_type=replace --attack_num=12 --attack_replace_round=3000
python FL_robust.py --round=4000 --data_distributed=iid --client_ep=5 --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_iid_14 --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=3000
python FL_robust.py --round=4000 --data_distributed=iid --client_ep=5 --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_iid_16 --attack=True --attack_type=replace --attack_num=16 --attack_replace_round=3000
python FL_robust.py --round=4000 --data_distributed=iid --client_ep=5 --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_FMNIST_MajorityVote_replace_iid_18 --attack=True --attack_type=replace --attack_num=18 --attack_replace_round=3000

# # FedAvg noniid
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_noniid_2 --attack=True --attack_type=replace --attack_num=2 --attack_replace_round=150
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_noniid_4 --attack=True --attack_type=replace --attack_num=4 --attack_replace_round=150
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_noniid_6 --attack=True --attack_type=replace --attack_num=6 --attack_replace_round=150
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_noniid_8 --attack=True --attack_type=replace --attack_num=8 --attack_replace_round=150
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_noniid_10 --attack=True --attack_type=replace --attack_num=10 --attack_replace_round=150
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_noniid_12 --attack=True --attack_type=replace --attack_num=12 --attack_replace_round=150
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_noniid_14 --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=150
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_noniid_16 --attack=True --attack_type=replace --attack_num=16 --attack_replace_round=150
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_FedAVG_replace_noniid_18 --attack=True --attack_type=replace --attack_num=18 --attack_replace_round=150

# # FedAVG-Krum noniid
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_noniid_2 --attack=True --attack_type=replace --attack_num=2 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_noniid_4 --attack=True --attack_type=replace --attack_num=4 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_noniid_6 --attack=True --attack_type=replace --attack_num=6 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_noniid_8 --attack=True --attack_type=replace --attack_num=8 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_noniid_10 --attack=True --attack_type=replace --attack_num=10 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_noniid_12 --attack=True --attack_type=replace --attack_num=12 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_noniid_14 --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_noniid_16 --attack=True --attack_type=replace --attack_num=16 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=R_FMNIST_Krumavg_replace_noniid_18 --attack=True --attack_type=replace --attack_num=18 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum

# # FedSIGN
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_sign_replace_noniid_2 --attack=True --attack_type=replace --attack_num=2 --attack_replace_round=150 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_sign_replace_noniid_4 --attack=True --attack_type=replace --attack_num=4 --attack_replace_round=150 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_sign_replace_noniid_6 --attack=True --attack_type=replace --attack_num=6 --attack_replace_round=150 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_sign_replace_noniid_8 --attack=True --attack_type=replace --attack_num=8 --attack_replace_round=150
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_sign_replace_noniid_10 --attack=True --attack_type=replace --attack_num=10 --attack_replace_round=150 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_sign_replace_noniid_12 --attack=True --attack_type=replace --attack_num=12 --attack_replace_round=150
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_sign_replace_noniid_14 --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=150 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_sign_replace_noniid_16 --attack=True --attack_type=replace --attack_num=16 --attack_replace_round=150
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_sign_replace_noniid_18 --attack=True --attack_type=replace --attack_num=18 --attack_replace_round=150 

# # FedSIGN-Krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_Krum_replace_noniid_2 --attack=True --attack_type=replace --attack_num=2 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_Krum_replace_noniid_4 --attack=True --attack_type=replace --attack_num=4 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_Krum_replace_noniid_6 --attack=True --attack_type=replace --attack_num=6 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_Krum_replace_noniid_8 --attack=True --attack_type=replace --attack_num=8 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_Krum_replace_noniid_10 --attack=True --attack_type=replace --attack_num=10 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_Krum_replace_noniid_12 --attack=True --attack_type=replace --attack_num=12 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_Krum_replace_noniid_14 --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_Krum_replace_noniid_16 --attack=True --attack_type=replace --attack_num=16 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_Krum_replace_noniid_18 --attack=True --attack_type=replace --attack_num=18 --attack_replace_round=150 --robust_agg=True --robust_agg_type=krum

# FedSIGN-DC noniid
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_noniid_2 --attack=True --attack_type=replace --attack_num=2 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_noniid_4 --attack=True --attack_type=replace --attack_num=4 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_noniid_6 --attack=True --attack_type=replace --attack_num=6 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_noniid_8 --attack=True --attack_type=replace --attack_num=8 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_noniid_10 --attack=True --attack_type=replace --attack_num=10 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_noniid_12 --attack=True --attack_type=replace --attack_num=12 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_noniid_14 --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_noniid_16 --attack=True --attack_type=replace --attack_num=16 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 
# python FL_robust.py --round=200 --client_ep=5 --data_distributed=noniid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_alpha=0.001 --name=R_FMNIST_mipc_replace_noniid_18 --attack=True --attack_type=replace --attack_num=18 --attack_replace_round=150 --robust_agg=True --robust_agg_type=mipc 