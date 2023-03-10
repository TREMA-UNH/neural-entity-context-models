import os
import json
import pytrec_eval
import torch
import torch.nn as nn
import tqdm


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_trec(rst_file, rst_dict):
    with open(rst_file, 'w') as writer:
        for q_id, scores in rst_dict.items():
            res = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
            for rank, value in enumerate(res):
                writer.write(
                    q_id + ' Q0 ' + str(value[0]) + ' ' + str(rank + 1) + ' ' + str(value[1][0][0]) + ' BERT\n')


def save_features(rst_file, features):
    with open(rst_file, 'w') as writer:
        for feature in features:
            writer.write(feature + '\n')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_checkpoint(save_path, model):
    if save_path is None:
        return

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, device):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    print(f'Model loaded from <== {load_path}')


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path is None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def evaluate(model, data_loader, device):
    rst_dict = {}
    model.eval()

    num_batch = len(data_loader)

    with torch.no_grad():
        for dev_batch in tqdm.tqdm(data_loader, total=num_batch):
            if dev_batch is not None:
                query_id, entity_id, label = dev_batch['query_id'], dev_batch['entity_id'], dev_batch['label']

                batch_score = model(
                    query_emb=dev_batch['query_emb'].to(device),
                    entity_emb=dev_batch['ent_emb'].to(device),
                    neighbors=dev_batch['entity_neighbors']
                )

                batch_score = batch_score.detach().cpu().tolist()
                for (q_id, d_id, b_s, l) in zip(query_id, entity_id, batch_score, label):
                    if q_id not in rst_dict:
                        rst_dict[q_id] = {}
                    if d_id not in rst_dict[q_id] or (type(rst_dict[q_id][d_id])==list and b_s > rst_dict[q_id][d_id][0]):
                            rst_dict[q_id][d_id] = [b_s, l]
                    else:
                        if d_id not in rst_dict[q_id] or (type(rst_dict[q_id][d_id])==float and b_s > rst_dict[q_id][d_id]):
                            rst_dict[q_id][d_id] = [b_s, l]

    return rst_dict



