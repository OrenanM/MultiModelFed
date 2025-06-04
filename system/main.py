#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import sys

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serveramp import FedAMP
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverlocal import Local
from flcore.servers.serverper import FedPer
from flcore.servers.serverapfl import APFL
from flcore.servers.serverditto import Ditto
from flcore.servers.serverrep import FedRep
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverbn import FedBN
from flcore.servers.serverrod import FedROD
from flcore.servers.serverproto import FedProto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.servermoon import MOON
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverapple import APPLE
from flcore.servers.servergen import FedGen
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverfd import FD
from flcore.servers.serverala import FedALA
from flcore.servers.serverpac import FedPAC
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.servergc import FedGC
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.serverpcl import FedPCL
from flcore.servers.servercp import FedCP
from flcore.servers.servergpfl import GPFL
from flcore.servers.serverntd import FedNTD
from flcore.servers.servergh import FedGH
from flcore.servers.serverdbe import FedDBE
from flcore.servers.servercac import FedCAC
from flcore.servers.serverda import PFL_DA
from flcore.servers.serverlc import FedLC
from flcore.servers.serveras import FedAS

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):

    time_list = []
    reporter = MemReporter()
    models_name = args.models[:]
    
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
   
        for me, (model_str, dataset) in enumerate(zip(models_name, args.datasets)):
            start = time.time()

            # Generate args.models
            if model_str == "MLR": # convex
                if "MNIST" in dataset:
                    args.models[me] = Mclr_Logistic(1*28*28, num_classes=args.num_classes[me]).to(args.device)
                elif "Cifar10" in dataset:
                    args.models[me] = Mclr_Logistic(3*32*32, num_classes=args.num_classes[me]).to(args.device)
                else:
                    args.models[me] = Mclr_Logistic(60, num_classes=args.num_classes[me]).to(args.device)

            elif model_str == "CNN": # non-convex
                if "MNIST" in dataset:
                    args.models[me] = FedAvgCNN(in_features=1, num_classes=args.num_classes[me], dim=1024).to(args.device)
                elif "Cifar10" in dataset:
                    args.models[me] = FedAvgCNN(in_features=3, num_classes=args.num_classes[me], dim=1600).to(args.device)
                elif "Omniglot" in dataset:
                    args.models[me] = FedAvgCNN(in_features=1, num_classes=args.num_classes[me], dim=33856).to(args.device)
                    # args.models[i] = CifarNet(num_classes=args.num_classes[me]).to(args.device)
                elif "Digit5" in dataset:
                    args.models[me] = Digit5CNN().to(args.device)
                else:
                    args.models[me] = FedAvgCNN(in_features=3, num_classes=args.num_classes[me], dim=10816).to(args.device)

            elif model_str == "DNN": # non-convex
                if "MNIST" in dataset:
                    args.models[me] = DNN(1*28*28, 100, num_classes=args.num_classes[me]).to(args.device)
                elif "Cifar10" in dataset:
                    args.models[me] = DNN(3*32*32, 100, num_classes=args.num_classes[me]).to(args.device)
                else:
                    args.models[me] = DNN(60, 20, num_classes=args.num_classes[me]).to(args.device)
            
            elif model_str == "ResNet18":
                args.models[me] = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes[me]).to(args.device)
                
                # args.models[i] = torchvision.models.resnet18(pretrained=True).to(args.device)
                # feature_dim = list(args.models[i].fc.parameters())[0].shape[1]
                # args.models[i].fc = nn.Linear(feature_dim, args.num_classes[me]).to(args.device)
                
                # args.models[i] = resnet18(num_classes=args.num_classes[me], has_bn=True, bn_block_num=4).to(args.device)
            
            elif model_str == "ResNet10":
                args.models[me] = resnet10(num_classes=args.num_classes[me]).to(args.device)
            
            elif model_str == "ResNet34":
                args.models[me] = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes[me]).to(args.device)

            elif model_str == "AlexNet":
                args.models[me] = alexnet(pretrained=False, num_classes=args.num_classes[me]).to(args.device)
                
                # args.models[i] = alexnet(pretrained=True).to(args.device)
                # feature_dim = list(args.models[i].fc.parameters())[0].shape[1]
                # args.models[i].fc = nn.Linear(feature_dim, args.num_classes[me]).to(args.device)
                
            elif model_str == "GoogleNet":
                args.models[me] = torchvision.models.googlenet(pretrained=False, aux_logits=False, 
                                                        num_classes=args.num_classes[me]).to(args.device)
                
                # args.models[i] = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
                # feature_dim = list(args.models[i].fc.parameters())[0].shape[1]
                # args.models[i].fc = nn.Linear(feature_dim, args.num_classes[me]).to(args.device)

            elif model_str == "MobileNet":
                args.models[me] = mobilenet_v2(pretrained=False, num_classes=args.num_classes[me]).to(args.device)
                
                # args.models[i] = mobilenet_v2(pretrained=True).to(args.device)
                # feature_dim = list(args.models[i].fc.parameters())[0].shape[1]
                # args.models[i].fc = nn.Linear(feature_dim, args.num_classes[me]).to(args.device)
                
            elif model_str == "LSTM":
                args.models[me] = LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes[me]).to(args.device)

            elif model_str == "BiLSTM":
                args.models[me] = BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, 
                                                    output_size=args.num_classes[me], num_layers=1, 
                                                    embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                                                    embedding_length=args.feature_dim).to(args.device)

            elif model_str == "fastText":
                args.models[me] = fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes[me]).to(args.device)

            elif model_str == "TextCNN":
                args.models[me] = TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, 
                                    num_classes=args.num_classes[me]).to(args.device)

            elif model_str == "Transformer":
                args.models[me] = TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead =8, nlayers=2, 
                                            num_classes=args.num_classes[me], max_len=args.max_len).to(args.device)
            
            elif model_str == "AmazonMLP":
                args.models[me] = AmazonMLP().to(args.device)

            elif model_str == "HARCNN":
                if dataset == 'HAR':
                    args.models[me] = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes[me], conv_kernel_size=(1, 9), 
                                        pool_kernel_size=(1, 2)).to(args.device)
                elif dataset == 'PAMAP2':
                    args.models[me] = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes[me], conv_kernel_size=(1, 9), 
                                        pool_kernel_size=(1, 2)).to(args.device)

            else:
                raise NotImplementedError
            
        args.heads = []
        # select algorithm
        if args.algorithm == "FedAvg":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedAvg(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.head.append(head)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedRep(args, i)

        elif args.algorithm == "FedPHP":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedROD(args, i)

        elif args.algorithm == "FedProto":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = MOON(args, i)

        elif args.algorithm == "FedBABU":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        elif args.algorithm == "FedGen":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedGen(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "FD":
            server = FD(args, i)

        elif args.algorithm == "FedALA":
            server = FedALA(args, i)

        elif args.algorithm == "FedPAC":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedPAC(args, i)

        elif args.algorithm == "LG-FedAvg":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FedGC":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedGC(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedKD(args, i)

        elif args.algorithm == "FedPCL":
            for me in range(len(args.models)):
                args.models[me].fc = nn.Identity()
            server = FedPCL(args, i)

        elif args.algorithm == "FedCP":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedCP(args, i)

        elif args.algorithm == "GPFL":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = GPFL(args, i)

        elif args.algorithm == "FedNTD":
            server = FedNTD(args, i)

        elif args.algorithm == "FedGH":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedGH(args, i)

        elif args.algorithm == "FedDBE":
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedDBE(args, i)

        elif args.algorithm == 'FedCAC':
            server = FedCAC(args, i)

        elif args.algorithm == 'PFL-DA':
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = PFL_DA(args, i)

        elif args.algorithm == 'FedLC':
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedLC(args, i)

        elif args.algorithm == 'FedAS':
            for me in range(len(args.models)):
                head = copy.deepcopy(args.models[me].fc)
                args.models[me].fc = nn.Identity()
                args.models[me] = BaseHeadSplit(args.models[me], head)
                args.heads.append(head)
            server = FedAS(args, i)
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    for dataset in args.datasets:
        print('-'*79)
        print(dataset)
        average_data(dataset=dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--datasets", action='append', default=['Cifar10', 'MNIST'])
    parser.add_argument('-ncl', "--num_classes", action='append', default=[10, 10])
    parser.add_argument('-m', '--models', action='append', default=['CNN', 'CNN'])
    parser.add_argument('-lbs', "--batch_sizes", action='append', default=[10, 10])
    parser.add_argument('-lr', "--local_learning_rate", action='append', 
                        default=[0.005, 0.005], help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON / FedCAC / FedLC
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    # create dataset
    parser.add_argument('-b', '--balance', default=False)
    parser.add_argument('-niid', '--noniid', default='noniid')
    parser.add_argument('-pd', '--partition_data', default='dir')

    # Concept Drift
    parser.add_argument('-rc', '--rounds_concept_drift', action='append', type=int, default=[])
    parser.add_argument('-na', '--new_alpha', action='append', type=float, default=[])
    parser.add_argument('-dd', '--dataset_concept_drift', type=str, default='MNIST')
    parser.add_argument('-ia', '--initial_alpha', type=float, default=0.1)

    # Label Shift
    parser.add_argument('-rds', '--rounds_label_shift', action='append', type=int, default=[])
    parser.add_argument('-dls', '--dataset_label_shift', type=str)
    parser.add_argument('-rl', '--replace_labels', nargs=2, type=int, action='append')

    # Monitor Accuracy
    parser.add_argument('-at', '--acc_threthold', type=float, default=0.1)
    parser.add_argument('-ma', '--monitor_acc', action='store_true', help='Monitorar acurácia (default: False)')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)

    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
