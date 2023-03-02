# # DPRSA
# ## label 70
# python FL_robust.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_Non_DPRSA_F_label_70_IID_FMNIST --attack=True --attack_type=label --attack_num=14
python FL_robust.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_AD_DPRSA_F_label_70_IID_FMNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=label --attack_num=14
python FL_robust.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_KRUM_DPRSA_F_label_70_IID_FMNIST --robust_agg_type=krum --attack=True --attack_type=label --attack_num=14
# ## gaussian 70
# python FL_robust.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_Non_DPRSA_F_gaussian_70_IID_FMNIST --attack=True --attack_type=gaussian --attack_num=14
python FL_robust.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_AD_DPRSA_F_gaussian_70_IID_FMNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=gaussian --attack_num=14
python FL_robust.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_KRUM_DPRSA_F_gaussian_70_IID_FMNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=gaussian --attack_num=14
# ## pixel 70
# python FL_robust.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_Non_DPRSA_F_pixel_70_IID_FMNIST --attack=True --attack_type=pixel --attack_num=14
python FL_robust.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_AD_DPRSA_F_pixel_70_IID_FMNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=pixel --attack_num=14
python FL_robust.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_KRUM_DPRSA_F_pixel_70_IID_FMNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=pixel --attack_num=14
## replace 70
# python FL_robust.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_Non_DPRSA_F_replace_70_IID_FMNIST --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=200
python FL_robust.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_AD_DPRSA_F_replace_70_IID_FMNIST --robust_agg=True --robust_agg_type=mipc --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=200
python FL_robust.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=R_KRUM_DPRSA_F_replace_70_IID_FMNIST --robust_agg=True --robust_agg_type=krum --attack=True --attack_type=replace --attack_num=14 --attack_replace_round=200
