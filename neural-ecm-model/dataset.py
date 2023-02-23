from typing import List, Tuple, Dict, Any
import json
import torch
from torch.utils.data import Dataset
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EntityRankingDataset(Dataset):
    def __init__(
            self,
            dataset,
            data_type: str,
            train: bool
    ):
        self._dataset = dataset
        self._data_type = data_type
        self._train = train
        self._read_data()

        self._count = len(self._examples)

    def _read_data(self):
        with open(self._dataset, 'r') as f:
            self._examples = [json.loads(line) for i, line in enumerate(f)]

    def __len__(self) -> int:
        return self._count


    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        if self._train:
            if self._data_type == 'pairwise':

                # Create the entity inputs
                #query_emb = np.average(np.array([entity['entity_embedding'] for entity in example['query']]), axis=0)
                #query_emb = example['query']['entity_embedding']
                #print(len(example['query']['query_embed']))
                #print("***************************")
                query_emb = np.array(example['query']['query_embed'])
                ent_emb_pos = np.array(example['ent_pos']['entity_embedding'])
                ent_emb_neg = np.array(example['ent_neg']['entity_embedding'])
                pos_ent_neighbors = example['ent_pos']['entity_neighbors']
                neg_ent_neighbors = example['ent_neg']['entity_neighbors']

                return {
                    'query_emb': query_emb,
                    'ent_emb_pos': ent_emb_pos,
                    'ent_emb_neg': ent_emb_neg,
                    'pos_ent_neighbors': pos_ent_neighbors,
                    'neg_ent_neighbors': neg_ent_neighbors
                }

            elif self._data_type == 'pointwise':

                # Create the entity inputs
                query_emb = np.average(np.array([entity['entity_embedding'] for entity in example['query']]), axis=0)
                ent_emb = np.array(example['entity']['entity_embedding'])

                return {
                    'query_emb': query_emb,
                    'ent_emb': ent_emb,
                    'label': example['label']
                }
            else:
                raise ValueError('Model type must be `pairwise` or `pointwise`.')
        else:

            #query_emb = np.average(np.array([entity['entity_embedding'] for entity in example['query']]), axis=0)
            query_emb = np.array(example['query']['query_embed'])
            ent_emb = np.array(example['entity']['entity_embedding'])
            neighbors = example['entity']['entity_neighbors']

            return {
                'query_emb': query_emb,
                'ent_emb': ent_emb,
                'label': example['label'],
                'query_id': example['query_id'],
                'entity_id': example['entity']['entity_id'],
                'entity_neighbors': neighbors
            }

    def collate(self, batch):
        if self._train:
            if self._data_type == 'pairwise':
                '''
                temp = []
                print("=============================")
                for item in batch:
                    print(str(type(item['query_emb']))+" "+str(len(item['query_emb'])))
                    if len(item['query_emb']) == 1:
                        print(item['query_emb'])
                        break
                    if len(item['query_emb']) == 50:
                        print(item['query_emb'])
                    temp.append(item['query_emb'])
                print(len(temp))
                print(len(temp[0]))
                new_temp = np.array(temp)
                print(new_temp.size)
                '''
                query_emb = torch.from_numpy(np.array([item['query_emb'] if len(item['query_emb']) == 50 else item['query_emb'][0] for item in batch])).float()
                ent_emb_pos = torch.from_numpy(np.array([item['ent_emb_pos'] for item in batch])).float()
                ent_emb_neg = torch.from_numpy(np.array([item['ent_emb_neg'] for item in batch])).float()
                #query_text = [item['query_text'] for item in batch]
                pos_ent_neighbors = [item['pos_ent_neighbors'] for item in batch]
                neg_ent_neighbors = [item['neg_ent_neighbors'] for item in batch]
                return {
                    'query_emb': query_emb,
                    'ent_emb_pos': ent_emb_pos,
                    'ent_emb_neg': ent_emb_neg,
                    'pos_ent_neighbors': pos_ent_neighbors,
                    'neg_ent_neighbors': neg_ent_neighbors
                }
            elif self._data_type == 'pointwise':
                query_emb = torch.from_numpy(np.array([item['query_emb'] for item in batch])).float()
                ent_emb = torch.from_numpy(np.array([item['ent_emb'] for item in batch])).float()
                label = torch.from_numpy(np.array([item['label'] for item in batch])).float()
                return {
                    'query_emb': query_emb,
                    'ent_emb': ent_emb,
                    'label': label
                }
            else:
                raise ValueError('Model type must be `pairwise` or `pointwise`.')
        else:
            query_id = [item['query_id'] for item in batch]
            entity_id = [item['entity_id'] for item in batch]
            label = [item['label'] for item in batch]
            #query_text = [item['query_text'] for item in batch]
            entity_neighbors = [item['entity_neighbors'] for item in batch]
            query_emb = torch.from_numpy(np.array([item['query_emb'] if len(item['query_emb']) == 50 else item['query_emb'][0] for item in batch])).float()
            ent_emb = torch.from_numpy(np.array([item['ent_emb'] for item in batch])).float()
            return {
                'query_emb': query_emb,
                'ent_emb': ent_emb,
                'label': label,
                'query_id': query_id,
                'entity_id': entity_id,
                'entity_neighbors': entity_neighbors
            }
