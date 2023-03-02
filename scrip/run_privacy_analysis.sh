python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --rsa=True --dprsa=True --dprsa_type=F --name=test_DPRSA_F_iid_ana --white_box=True --use_pretrain=True
python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --rsa=True --dprsa=True --dprsa_type=G --name=test_DPRSA_G_iid_ana --white_box=True --use_pretrain=True

python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --name=test_P_FedSIGN_ana --sign=True --sign_alpha=0.0001 --white_box=True --use_pretrain=True

python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --update_type=weight --name=test_P_FedAVG_ana --white_box=True --use_pretrain=True

python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --name=test_P_FedAvg_LDP_20_ana --ldp=True --ldp_epsilon=20 --ldp_delta=0.00001 --ldp_clip=3 --white_box=True --use_pretrain=True
python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --name=test_P_FedAvg_CDP_20_ana --cdp=True --cdp_epsilon=20 --cdp_delta=0.00001 --cdp_clip=0.5 --white_box=True --use_pretrain=True