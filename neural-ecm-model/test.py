import configparser
import os
import sys
import time
import json
import torch
import torch.nn as nn
import utils
import metrics
import warnings
import argparse
from typing import Tuple
from dataloader import EntityRankingDataLoader
from model import ProjectionScoreModel
from transformers import get_linear_schedule_with_warmup
from dataset import EntityRankingDataset
from trainer import Trainer


def test(model, data_loader, run_file, device):
    res_dict = utils.evaluate(
        model=model,
        data_loader=data_loader,
        device=device
    )

    print('Writing run file...')
    utils.save_trec(run_file, res_dict)
    print('[Done].')


def main():
    parser = argparse.ArgumentParser("Script to test a model.")
    parser.add_argument('--model-type', help='Type of model (pairwise|pointwise).', type=str, required=True)
    parser.add_argument('--test', help='Test data.', required=True, type=str)
    parser.add_argument('--run', help='Test run file.', required=True, type=str)
    parser.add_argument('--checkpoint', help='Name of checkpoint to load.', required=True, type=str)
    parser.add_argument('--batch-size', help='Size of each batch. Default: 8.', type=int, default=8)
    parser.add_argument('--num-workers', help='Number of workers to use for DataLoader. Default: 0', type=int,
                        default=0)
    parser.add_argument('--in-emb-dim', help='Dimension of input embedding.', required=True, type=int)
    parser.add_argument('--out-emb-dim', help='Dimension of output embedding.', required=True, type=int)
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Whether or not to use CUDA. Default: False.', action='store_true')
    args = parser.parse_args()

    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))

    device = torch.device(
        cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu'
    )


    print('Reading test data...')
    test_set = EntityRankingDataset(
        dataset=args.test,
        data_type=args.model_type,
        train=False
    )
    print('[Done].')

    print('Creating data loader...')
    print('Number of workers = ' + str(args.num_workers))
    print('Batch Size = ' + str(args.batch_size))

    test_loader = EntityRankingDataLoader(
        dataset=test_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print('[Done].')

    model = ProjectionScoreModel(input_emb_dim=args.in_emb_dim, output_emb_dim=args.out_emb_dim)
    print('Loading checkpoint...')
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print('[Done].')


    print('Using device: {}'.format(device))
    model.to(device)

    print("Starting to test...")

    test(
        model=model,
        data_loader=test_loader,
        run_file=args.run,
        device=device
    )

    print('Test complete.')


if __name__ == '__main__':
    main()
