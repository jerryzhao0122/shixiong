# python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.001 --num_client=4 --train_num=4 --update_type=weight --name=P_FedAvg_CDP_5 --cdp=True --cdp_epsilon=5 --cdp_delta=0.00001
# python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.001 --num_client=4 --train_num=4 --update_type=weight --name=P_FedAvg_CDP_10 --cdp=True --cdp_epsilon=10 --cdp_delta=0.00001
# python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.001 --num_client=4 --train_num=4 --update_type=weight --name=P_FedAvg_CDP_15 --cdp=True --cdp_epsilon=15 --cdp_delta=0.00001
# python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.001 --num_client=4 --train_num=4 --update_type=weight --name=P_FedAvg_CDP_20 --cdp=True --cdp_epsilon=20 --cdp_delta=0.00001

# python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.01 --num_client=4 --train_num=4 --update_type=weight --model=alexnet --name=P_FedAvg_CDP_20 --cdp=True --cdp_epsilon=20 --cdp_delta=0.00001 --cdp_clip=0.5
# python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.01 --num_client=4 --train_num=4 --update_type=weight --model=alexnet --name=P_FedAvg_CDP_15 --cdp=True --cdp_epsilon=15 --cdp_delta=0.00001 --cdp_clip=0.5
# python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.01 --num_client=4 --train_num=4 --update_type=weight --model=alexnet --name=P_FedAvg_CDP_10 --cdp=True --cdp_epsilon=10 --cdp_delta=0.00001 --cdp_clip=0.5
# python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.01 --num_client=4 --train_num=4 --update_type=weight --model=alexnet --name=P_FedAvg_CDP_5 --cdp=True --cdp_epsilon=5 --cdp_delta=0.00001 --cdp_clip=0.5

# python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --name=test_P_FedAvg_CDP_10 --cdp=True --cdp_epsilon=10 --cdp_delta=0.00001 --cdp_clip=0.5 --white_box=True --client_lr=0.01
# python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --name=test_P_FedAvg_CDP_20 --cdp=True --cdp_epsilon=20 --cdp_delta=0.00001 --cdp_clip=0.5 --white_box=True --client_lr=0.01


# 借助这个跑就行
python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --dataset=mnist --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --update_type=weight --name=test_P_FedAVG_MNIST --white_box=True 

# python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=mnist --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --name=test_P_FedAvg_CDP_MNIST_10 --cdp=True --cdp_epsilon=10 --cdp_delta=0.00001 --cdp_clip=0.5 --white_box=True --client_lr=0.01
python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=mnist --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --name=test_P_FedAvg_CDP_MNIST_20 --cdp=True --cdp_epsilon=20 --cdp_delta=0.00001 --cdp_clip=0.5 --white_box=True --client_lr=0.01
python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --name=test_P_FedAvg_CDP_FMNIST_10 --cdp=True --cdp_epsilon=10 --cdp_delta=0.00001 --cdp_clip=0.5 --white_box=True --client_lr=0.01
python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --name=test_P_FedAvg_CDP_FMNIST_20 --cdp=True --cdp_epsilon=20 --cdp_delta=0.00001 --cdp_clip=0.5 --white_box=True --client_lr=0.01