import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from shutil import copyfile
from tqdm import tqdm

from utils import vararg_callback_bool, vararg_callback_int
from dataloader import  *

import torch
from engine import classification_engine

sys.setrecursionlimit(40000)


def get_args_parser():
    parser = OptionParser()

    parser.add_option("--GPU", dest="GPU", help="the index of gpu is used", default=None, action="callback",
                      callback=vararg_callback_int)
    parser.add_option("--model", dest="model_name", help="vit_base|vit_small|swin_base|swin_tiny", default="vit_base", type="string")
    parser.add_option("--init", dest="init",
                      help="Random| ImageNet_1k| ImageNet_21k| SAM| DeiT| BEiT| DINO| MoCo_V3| MoBY | MAE| SimMIM",
                      default="Random", type="string")
    parser.add_option("--pretrained_weights", dest="pretrained_weights", help="Path to the Pretrained model", default=None, type="string")
    parser.add_option("--num_class", dest="num_class", help="number of the classes in the downstream task",
                      default=14, type="int")
    parser.add_option("--data_set", dest="data_set", help="ChestXray14|CheXpert|Shenzhen|VinDrCXR|RSNAPneumonia", default="ChestXray14", type="string")
    parser.add_option("--normalization", dest="normalization", help="how to normalize data (imagenet|chestx-ray)", default="imagenet",
                      type="string")
    parser.add_option("--img_size", dest="img_size", help="input image resolution", default=224, type="int")
    parser.add_option("--img_depth", dest="img_depth", help="num of image depth", default=3, type="int")
    parser.add_option("--data_dir", dest="data_dir", help="dataset dir",default=None, type="string")
    parser.add_option("--train_list", dest="train_list", help="file for training list",
                      default=None, type="string")
    parser.add_option("--val_list", dest="val_list", help="file for validating list",
                      default=None, type="string")
    parser.add_option("--test_list", dest="test_list", help="file for test list",
                      default=None, type="string")
    parser.add_option("--mode", dest="mode", help="train | test", default="train", type="string")
    parser.add_option("--batch_size", dest="batch_size", help="batch size", default=64, type="int")
    parser.add_option("--epochs", dest="epochs", help="num of epoches", default=200, type="int")
    parser.add_option("--exp_name", dest="exp_name", default="", type="string")

    # Optimizer parameters
    parser.add_option('--opt', default='momentum', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_option('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_option('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_option('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_option('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_option('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_option('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_option('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_option('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_option('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_option('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_option('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_option('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_option('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_option('--warmup-epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_option('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    # parser.add_option('--patience-epochs', type=int, default=10, metavar='N',
    #                     help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_option('--decay-rate', '--dr', type=float, default=0.5, metavar='RATE',
                        help='LR decay rate (default: 0.1)')


    parser.add_option("--patience", dest="patience", help="num of patient epoches", default=10, type="int")
    parser.add_option("--early_stop", dest="early_stop", help="whether use early_stop", default=True, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--trial", dest="num_trial", help="number of trials", default=1, type="int")
    parser.add_option("--start_index", dest="start_index", help="the start model index", default=0, type="int")
    parser.add_option("--clean", dest="clean", help="clean the existing data", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--resume", dest="resume", help="whether latest checkpoint", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--workers", dest="workers", help="number of CPU workers", default=8, type="int")
    parser.add_option("--print_freq", dest="print_freq", help="print frequency", default=50, type="int")
    parser.add_option("--test_augment", dest="test_augment", help="whether use test time augmentation",
                      default=True, action="callback", callback=vararg_callback_bool)
    parser.add_option("--anno_percent", dest="anno_percent", help="data percent", default=100, type="int")
    parser.add_option("--device", dest="device", help="cpu|cuda", default="cuda", type="string")
    parser.add_option("--activate", dest="activate", help="Sigmoid", default="Sigmoid", type="string")
    parser.add_option("--uncertain_label", dest="uncertain_label",
                      help="the label assigned to uncertain data (Ones | Zeros | LSR-Ones | LSR-Zeros)",
                      default="LSR-Ones", type="string")
    parser.add_option("--unknown_label", dest="unknown_label", help="the label assigned to unknown data",
                      default=0, type="int")


    (options, args) = parser.parse_args()

    return options


def main(args):
    print(args)
    assert args.data_dir is not None
    assert args.train_list is not None
    assert args.val_list is not None
    assert args.test_list is not None
    #if args.init.lower() != 'imagenet' and args.init.lower() != 'random':
    #    assert args.proxy_dir is not None
    args.exp_name = args.model_name + "_" + args.init + args.exp_name
    model_path = os.path.join("./Models/Classification",args.data_set)
    output_path = os.path.join("./Outputs/Classification",args.data_set)

    if args.data_set == "ChestXray14":
        diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        dataset_train = ChestXray14Dataset(images_path=args.data_dir, file_path=args.train_list,
                                           augment=build_transform_classification(normalize=args.normalization, mode="train"),annotation_percent=args.anno_percent)

        dataset_val = ChestXray14Dataset(images_path=args.data_dir, file_path=args.val_list,
                                         augment=build_transform_classification(normalize=args.normalization, mode="valid"))
        dataset_test = ChestXray14Dataset(images_path=args.data_dir, file_path=args.test_list,
                                          augment=build_transform_classification(normalize=args.normalization, mode="test"))

        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)


    elif args.data_set == "CheXpert":
        diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                           'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                           'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        test_diseases_name = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        test_diseases = [diseases.index(c) for c in test_diseases_name]
        dataset_train = CheXpertDataset(images_path=args.data_dir, file_path=args.train_list,
                                        augment=build_transform_classification(normalize=args.normalization, mode="train"), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label, annotation_percent=args.anno_percent)

        dataset_val = CheXpertDataset(images_path=args.data_dir, file_path=args.val_list,
                                      augment=build_transform_classification(normalize=args.normalization, mode="valid"), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label)

        dataset_test = CheXpertDataset(images_path=args.data_dir, file_path=args.test_list,
                                       augment=build_transform_classification(normalize=args.normalization, mode="test"), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label)

        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test, test_diseases)

    elif args.data_set == "Shenzhen":
        diseases = ['TB']
        dataset_train = ShenzhenCXR(images_path=args.data_dir, file_path=args.train_list,
                                    augment=build_transform_classification(normalize=args.normalization, mode="train"), annotation_percent=args.anno_percent)

        dataset_val = ShenzhenCXR(images_path=args.data_dir, file_path=args.val_list,
                                  augment=build_transform_classification(normalize=args.normalization, mode="valid"))

        dataset_test = ShenzhenCXR(images_path=args.data_dir, file_path=args.test_list,
                                   augment=build_transform_classification(normalize=args.normalization, mode="test"))


        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)
    
    elif args.data_set == "VinDrCXR":
        diseases = ['PE', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']
        dataset_train = VinDrCXR(images_path=args.data_dir, file_path=args.train_list,
                                    augment=build_transform_classification(normalize=args.normalization, mode="train"), annotation_percent=args.anno_percent)

        dataset_val = VinDrCXR(images_path=args.data_dir, file_path=args.val_list,
                                  augment=build_transform_classification(normalize=args.normalization, mode="valid"))

        dataset_test = VinDrCXR(images_path=args.data_dir, file_path=args.test_list,
                                   augment=build_transform_classification(normalize=args.normalization, mode="test"))


        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)
    elif args.data_set == "RSNAPneumonia":
        diseases = ['Normal', 'No Lung Opacity/Not Normal', 'Lung Opacity']
        dataset_train = RSNAPneumonia(images_path=args.data_dir, file_path=args.train_list,
                                    augment=build_transform_classification(normalize=args.normalization, mode="train"), annotation_percent=args.anno_percent)

        dataset_val = RSNAPneumonia(images_path=args.data_dir, file_path=args.val_list,
                                  augment=build_transform_classification(normalize=args.normalization, mode="valid"))

        dataset_test = RSNAPneumonia(images_path=args.data_dir, file_path=args.test_list,
                                   augment=build_transform_classification(normalize=args.normalization, mode="test"))


        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)

if __name__ == '__main__':
    args = get_args_parser()
    main(args)

