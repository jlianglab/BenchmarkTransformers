#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -p rcgpu6
#SBATCH -q wildfire
#SBATCH -C V100
#SBATCH --gres=gpu:1
#SBATCH -t 6-23:59
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dongaoma@asu.edu

source activate SimMIM
module load cuda/10.2.89

#python main_classification.py --data_set ChestXray14 --data_dir /data/jliang12/dongaoma/ChestX-ray14/images --train_list dataset/Xray14_train_official.txt --val_list dataset/Xray14_val_official.txt --test_list dataset/Xray14_test_official.txt --lr 0.1 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 64 --model vit_base --init imagenet_21k
#python main_classification.py --data_set ChestXray14 --data_dir /data/jliang12/dongaoma/ChestX-ray14/images --train_list dataset/Xray14_train_official.txt --val_list dataset/Xray14_val_official.txt --test_list dataset/Xray14_test_official.txt --lr 0.1 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 64 --model vit_base --init imagenet_1k
#python main_classification.py --data_set ChestXray14 --data_dir /data/jliang12/dongaoma/ChestX-ray14/images --train_list dataset/Xray14_train_official.txt --val_list dataset/Xray14_val_official.txt --test_list dataset/Xray14_test_official.txt --lr 0.1 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 32 --model swin_base --init imagenet_21k

#python main_classification.py --data_set ChestXray14 --data_dir /data/jliang12/dongaoma/ChestX-ray14/images --train_list dataset/Xray14_train_official.txt --val_list dataset/Xray14_val_official.txt --test_list dataset/Xray14_test_official.txt --lr 0.1 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 64 --model vit_base --init moco_v3 --pretrained_weights vit-b-300ep.pth.tar

python main_classification.py --data_set ChestXray14 --data_dir /data/jliang12/dongaoma/ChestX-ray14/images --train_list dataset/Xray14_train_official.txt --val_list dataset/Xray14_val_official.txt --test_list dataset/Xray14_test_official.txt --lr 0.1 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 64 --model vit_small --init moby --pretrained_weights moby_deit_small_300ep_pretrained.pth
python main_classification.py --data_set ChestXray14 --data_dir /data/jliang12/dongaoma/ChestX-ray14/images --train_list dataset/Xray14_train_official.txt --val_list dataset/Xray14_val_official.txt --test_list dataset/Xray14_test_official.txt --lr 0.1 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 64 --model vit_base --init beit
python main_classification.py --data_set ChestXray14 --data_dir /data/jliang12/dongaoma/ChestX-ray14/images --train_list dataset/Xray14_train_official.txt --val_list dataset/Xray14_val_official.txt --test_list dataset/Xray14_test_official.txt --lr 0.1 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 64 --model vit_base --init dino
python main_classification.py --data_set ChestXray14 --data_dir /data/jliang12/dongaoma/ChestX-ray14/images --train_list dataset/Xray14_train_official.txt --val_list dataset/Xray14_val_official.txt --test_list dataset/Xray14_test_official.txt --lr 0.1 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 64 --model vit_base --init DeiT_distilled

