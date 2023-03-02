# # FedSIGN 70%
# ## label 70
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.01 --sign_alpha=0.00001 --name=R_NON_FedSIGN_label_70_IID_MNIST --attack=True --attack_type=label --attack_num=14
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.01 --sign_alpha=0.00001 --name=R_AD_FedSIGN_label_70_IID_MNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=label --attack_num=14
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.01 --sign_alpha=0.00001 --name=R_KRUM_FedSIGN_label_70_IID_MNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=label --attack_num=14
# ## gaussian 70
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.01 --sign_alpha=0.00001 --name=R_NON_FedSIGN_gaussian_70_IID_MNIST --attack=True --attack_type=gaussian --attack_num=14
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.01 --sign_alpha=0.00001 --name=R_AD_FedSIGN_gaussian_70_IID_MNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=gaussian --attack_num=14
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.01 --sign_alpha=0.00001 --name=R_KRUM_FedSIGN_gaussian_70_IID_MNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=gaussian --attack_num=14
# ## pixel 70
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.01 --sign_alpha=0.00001 --name=R_NON_FedSIGN_pixel_70_IID_MNIST --attack=True --attack_type=pixel --attack_num=14
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.01 --sign_alpha=0.00001 --name=R_AD_FedSIGN_pixel_70_IID_MNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=pixel --attack_num=14
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.01 --sign_alpha=0.00001 --name=R_KRUM_FedSIGN_pixel_70_IID_MNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=pixel --attack_num=14
# ## replace 70
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.01 --sign_alpha=0.00001 --name=R_NON_FedSIGN_replace_70_IID_MNIST --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=200
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.01 --sign_alpha=0.00001 --name=R_AD_FedSIGN_replace_70_IID_MNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=200
# python FL_robust.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.01 --sign_alpha=0.00001 --name=R_KRUM_FedSIGN_replace_70_IID_MNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=200

# # DPRSA
# ## label 70
# python FL_robust.py --round=500 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_Non_DPRSA_F_label_70_IID_MNIST --attack=True --attack_type=label --attack_num=14
# python FL_robust.py --round=500 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_AD_DPRSA_F_label_70_IID_MNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=label --attack_num=14
# python FL_robust.py --round=500 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_KRUM_DPRSA_F_label_70_IID_MNIST --robust_agg_type=krum --attack=True --attack_type=label --attack_num=14
# ## gaussian 70
# python FL_robust.py --round=500 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_Non_DPRSA_F_label_70_IID_MNIST --attack=True --attack_type=gaussian --attack_num=14
# python FL_robust.py --round=500 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_AD_DPRSA_F_label_70_IID_MNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=gaussian --attack_num=14
# python FL_robust.py --round=500 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_KRUM_DPRSA_F_label_70_IID_MNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=gaussian --attack_num=14
# ## pixel 70
# python FL_robust.py --round=500 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_Non_DPRSA_F_label_70_IID_MNIST --attack=True --attack_type=pixel --attack_num=14
# python FL_robust.py --round=500 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_AD_DPRSA_F_label_70_IID_MNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=pixel --attack_num=14
# python FL_robust.py --round=500 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_KRUM_DPRSA_F_label_70_IID_MNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=pixel --attack_num=14
## replace 70
python FL_robust.py --round=500 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_Non_DPRSA_F_label_70_IID_MNIST --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=300
python FL_robust.py --round=500 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_AD_DPRSA_F_label_70_IID_MNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=300
python FL_robust.py --round=500 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_KRUM_DPRSA_F_label_70_IID_MNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=300


# # Majority Vote
# ## label 70
# python FL_robust.py --round=1000 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_NON_MajorityVote_label_70_IID_MNIST --attack=True --attack_type=label --attack_num=14
# python FL_robust.py --round=1000 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_AD_MajorityVote_label_70_IID_MNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=label --attack_num=14
# python FL_robust.py --round=1000 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_Krum_MajorityVote_label_70_IID_MNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=label --attack_num=14
# ## gaussian 70
# python FL_robust.py --round=1000 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_NON_MajorityVote_gaussian_70_IID_MNIST --attack=True --attack_type=gaussian --attack_num=14
# python FL_robust.py --round=1000 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_AD_MajorityVote_gaussian_70_IID_MNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=gaussian --attack_num=14
# python FL_robust.py --round=1000 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_Krum_MajorityVote_gaussian_70_IID_MNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=gaussian --attack_num=14
# ## pixel 70
# python FL_robust.py --round=1000 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_NON_MajorityVote_pixel_70_IID_MNIST --attack=True --attack_type=pixel --attack_num=14
# python FL_robust.py --round=1000 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_AD_MajorityVote_pixel_70_IID_MNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=pixel --attack_num=14
# python FL_robust.py --round=1000 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_Krum_MajorityVote_pixel_70_IID_MNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=pixel --attack_num=14
# ## replace 70
# python FL_robust.py --round=1000 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_NON_MajorityVote_replace_70_IID_MNIST --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=800
# python FL_robust.py --round=1000 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_AD_MajorityVote_replace_70_IID_MNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=800
# python FL_robust.py --round=1000 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --model=lanet --majority_vote=True --name=R_Krum_MajorityVote_replace_70_IID_MNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=800