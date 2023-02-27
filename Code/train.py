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
import numpy as np
import random
from typing import Tuple
from dataloader import EntityRankingDataLoader
from model import NeuralECMModel
from transformers import get_linear_schedule_with_warmup
from dataset import EntityRankingDataset
from trainer import Trainer


def train(model, trainer, epochs, metric, qrels, valid_loader, save_path, save, run_file,
          eval_every, device):
    best_valid_metric = 0.0

    for epoch in range(epochs):

        # Train
        start_time = time.time()
        train_loss = trainer.train()

        # Validate
        if (epoch + 1) % eval_every == 0:
            res_dict = utils.evaluate(
                model=model,
                data_loader=valid_loader,
                device=device
            )

            utils.save_trec(os.path.join(save_path, run_file), res_dict)
            valid_metric = metrics.get_metric(qrels, os.path.join(save_path, run_file), metric)

            if valid_metric >= best_valid_metric:
                best_valid_metric = valid_metric
                utils.save_checkpoint(os.path.join(save_path, save), model)

            end_time = time.time()
            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(
                f'\t Train Loss: {train_loss:.3f}| Val. Metric: {valid_metric:.4f} | Best Val. Metric: {best_valid_metric:.4f}')


def main():
    parser = argparse.ArgumentParser("Script to train a model.")
    parser.add_argument('--model-type', help='Type of model (pairwise|pointwise).', type=str, required=True)
    parser.add_argument('--train', help='Training data.', required=True, type=str)
    parser.add_argument('--save-dir', help='Directory where model is saved.', required=True, type=str)
    parser.add_argument('--dev', help='Development data.', required=True, type=str)
    parser.add_argument('--qrels', help='Ground truth file in TREC format.', required=True, type=str)
    parser.add_argument('--save', help='Name of checkpoint to save. Default: bert.bin', default='bert.bin', type=str)
    parser.add_argument('--checkpoint', help='Name of checkpoint to load. Default: None', default=None, type=str)
    parser.add_argument('--run', help='Output run file in TREC format.', required=True,type=str)
    parser.add_argument('--metric', help='Metric to use for evaluation. Default: map', default='map', type=str)
    parser.add_argument('--epoch', help='Number of epochs. Default: 20', type=int, default=20)
    parser.add_argument('--batch-size', help='Size of each batch. Default: 8.', type=int, default=8)
    parser.add_argument('--learning-rate', help='Learning rate. Default: 2e-5.', type=float, default=2e-5)
    parser.add_argument('--n-warmup-steps', help='Number of warmup steps for scheduling. Default: 1000.', type=int,
                        default=1000)
    parser.add_argument('--eval-every', help='Evaluate every number of epochs. Default: 1', type=int, default=1)
    parser.add_argument('--num-workers', help='Number of workers to use for DataLoader. Default: 0', type=int,
                        default=0)
    parser.add_argument('--query-in-emb-dim', help='Dimension of query input embedding.', required=True, type=int)
    parser.add_argument('--ent-in-emb-dim', help='Dimension of entity input embedding.', required=True, type=int)
    parser.add_argument('--para-in-emb-dim', help='Dimension of paragraph input embedding.', required=True, type=int)
    parser.add_argument('--model', help='Model to retreive query and paragraph representations', default='distilbert-base-uncased', type=str)
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Whether or not to use CUDA. Default: False.', action='store_true')
    parser.add_argument('--seed', help='Random seed initialization',type=int, default=49500)
    parser.add_argument('--layer-flag', help='Flag to select the GNN model (1: GRN| 2:GAT), default = 1', type=int, default=1)
    args = parser.parse_args()

    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))

    device = torch.device(
        cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu'
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_config = json.dumps({
        'Model Type': args.model_type,
        'Metric': args.metric,
        'Epochs': args.epoch,
        'Batch Size': args.batch_size,
        'Learning Rate': args.learning_rate,
        'Warmup Steps': args.n_warmup_steps,
    })
    config_file: str = os.path.join(args.save_dir, 'config.json')
    with open(config_file, 'w') as f:
        f.write("%s\n" % model_config)

    print('Reading train data...')
    train_set = EntityRankingDataset(
        dataset=args.train,
        data_type=args.model_type,
        train=True
    )
    print('[Done].')
    print('Reading dev data...')
    dev_set = EntityRankingDataset(
        dataset=args.dev,
        data_type=args.model_type,
        train=False
    )
    print('[Done].')

    print('Creating data loaders...')
    print('Number of workers = ' + str(args.num_workers))
    print('Batch Size = ' + str(args.batch_size))
    train_loader = EntityRankingDataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    dev_loader = EntityRankingDataLoader(
        dataset=dev_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print('[Done].')

    print('Model Type: ' + args.model_type)
    model = NeuralECMModel(ent_input_emb_dim=args.ent_in_emb_dim, 
            query_input_emb_dim=args.query_in_emb_dim,
            para_input_emb_dim=args.para_in_emb_dim,
            #model=args.model,
            #tokenizer=args.model,
            device=device,
            layer_flag=args.layer_flag)

    if args.model_type == 'pairwise':
        loss_fn = nn.MarginRankingLoss(margin=1)
    elif args.model_type == 'pointwise':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError('Model type must be `pairwise` or `pointwise`.')

    if args.checkpoint is not None:
        print('Loading checkpoint...')
        model.load_state_dict(torch.load(args.checkpoint))
        print('[Done].')


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.n_warmup_steps,
        num_training_steps=len(train_set) * args.epoch // args.batch_size)

    print('Using device: {}'.format(device))
    model.to(device)
    loss_fn.to(device)

    print("Starting to train...")
    trainer = Trainer(
        model=model,
        model_type=args.model_type,
        optimizer=optimizer,
        criterion=loss_fn,
        scheduler=scheduler,
        data_loader=train_loader,
        device=device
    )

    train(
        model=model,
        trainer=trainer,
        epochs=args.epoch,
        metric=args.metric,
        qrels=args.qrels,
        valid_loader=dev_loader,
        save_path=args.save_dir,
        save=args.save,
        run_file=args.run,
        eval_every=args.eval_every,
        device=device
    )

    print('Training complete.')


if __name__ == '__main__':
    main()
