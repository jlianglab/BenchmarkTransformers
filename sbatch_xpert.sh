#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -p rcgpu1
#SBATCH -q wildfire
#SBATCH -C V100
#SBATCH --gres=gpu:1
#SBATCH -t 6-23:59
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dongaoma@asu.edu

source activate SimMIM
module load cuda/10.2.89

python main_classification.py --data_set CheXpert --data_dir /data/jliang12/mhossei2/Dataset/ --train_list /data/jliang12/mhossei2/Dataset/CheXpert-v1.0/dataset_files/CheXpert_train.csv  --val_list /data/jliang12/mhossei2/Dataset/CheXpert-v1.0/dataset_files/CheXpert_valid.csv --test_list /data/jliang12/mhossei2/Dataset/CheXpert-v1.0/dataset_files/CheXpert_test.csv --lr 0.1 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 64 --model swin_base --init random --trial 5

#python main_classification.py --data_set CheXpert --data_dir /data/jliang12/mhossei2/Dataset/ --train_list /data/jliang12/mhossei2/Dataset/CheXpert-v1.0/dataset_files/CheXpert_train.csv  --val_list /data/jliang12/mhossei2/Dataset/CheXpert-v1.0/dataset_files/CheXpert_valid.csv --test_list /data/jliang12/mhossei2/Dataset/CheXpert-v1.0/dataset_files/CheXpert_test.csv --lr 0.1 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 64 --model vit_base --init dino --trial 5

#python main_classification.py --data_set CheXpert --data_dir /data/jliang12/mhossei2/Dataset/ --train_list /data/jliang12/mhossei2/Dataset/CheXpert-v1.0/dataset_files/CheXpert_train.csv  --val_list /data/jliang12/mhossei2/Dataset/CheXpert-v1.0/dataset_files/CheXpert_valid.csv --test_list /data/jliang12/mhossei2/Dataset/CheXpert-v1.0/dataset_files/CheXpert_test.csv --lr 0.1 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 64 --model vit_small --init moby  --pretrained_weights moby_deit_small_300ep_pretrained.pth --trial 5