import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn


def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='RLMRec')
    parser.add_argument('--model', type=str, default='LightGCN', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Device number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    parser.add_argument('--mix_weight', type=float, default=-1, help='mix')
    parser.add_argument('--gamma', type=float, default=-1, help='gamma')
    parser.add_argument('--kd1_weight', type=float, default=-1, help='kd1_weight')
    parser.add_argument('--loglog', type=str, default="", help='log')
    parser.add_argument('--lambda1', type=float, default=0.0, help='log')
    parser.add_argument('--lambda2', type=float, default=0.0, help='log')
    parser.add_argument('--lambda3', type=float, default=0.0, help='log')
    parser.add_argument('--lambda4', type=float, default=0.0, help='log')
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'
        # print("Read the default (blank) configuration.")

    # dataset
    if dataset is not None:
        args.dataset = dataset

    # find yml file
    if not os.path.exists('./encoder/config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")

    # read yml file
    with open('./encoder/config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()
        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
        configs['device'] = args.device
        if args.dataset is not None:
            configs['data']['name'] = args.dataset
        if args.seed is not None:
            configs['train']['seed'] = args.seed
        if args.mix_weight != -1:
            configs['model'][args.dataset]['mix_weight'] = args.mix_weight
        if args.kd1_weight != -1:
            configs['model'][args.dataset]['kd1_weight'] = args.kd1_weight

        if args.lambda1 != 0.0:
            configs['model'][args.dataset]['ssl_lambda1'] = args.lambda1
        if args.lambda2 != 0.0:
            configs['model'][args.dataset]['ssl_lambda2'] = args.lambda2
        if args.lambda3 != 0.0:
            configs['model'][args.dataset]['ssl_lambda3'] = args.lambda3
        if args.lambda4 != 0.0:
            configs['model'][args.dataset]['ssl_lambda4'] = args.lambda4
        if args.gamma != -1:
            configs['model'][args.dataset]['gamma'] = args.gamma

        if args.loglog != "":
            configs['train']['log'] = args.loglog
        # semantic embeddings
        usrprf_embeds_path = "./data/{}/usr_emb_np.pkl".format(configs['data']['name'])
        itmprf_embeds_path = "./data/{}/itm_emb_np.pkl".format(configs['data']['name'])
        with open(usrprf_embeds_path, 'rb') as f:
            configs['usrprf_embeds'] = pickle.load(f)
        with open(itmprf_embeds_path, 'rb') as f:
            configs['itmprf_embeds'] = pickle.load(f)

        return configs


configs = parse_configure()
