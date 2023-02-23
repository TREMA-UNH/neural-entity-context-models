from typing import Tuple, List
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import warnings
import json
import numpy as np


class GRN(nn.Module):

    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, device, concat=False, activation=nn.ELU(), 
            dropout_prob=0.01, add_skip_connection=False, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_out_features = num_out_features
        self.num_in_features = num_in_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.device = device
        self.num_of_heads = 1

        # Trainable weights: linear projection matrix (denoated as "w" in the paper)

        self.linear_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def forward(self, nodes, neighbors, attention_scores):
        
        in_nodes_features = nodes


        #shape = (N, FIN) where N-number of nodes and FIN - number of features for each node
        # We apply drop out to all of the input nodes as mentioned in the paper

        in_nodes_features = self.dropout(in_nodes_features)

        #nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.linear_proj(in_nodes_features)

        nodes_features_proj = self.dropout(nodes_features_proj)

        
        for i in range(len(neighbors)):
            #neighbors_proj = self.linear_proj(neighbors[i]).view(-1, self.num_of_heads, self.num_out_features)
            neighbors_proj = self.linear_proj(neighbors[i])
            neighbors_proj = self.dropout(neighbors_proj)
            #print(str(neighbors_proj.shape)+" "+str(attention_scores[i].shape))
            #print(attention_scores[i])
            attention_scores[i] = attention_scores[i].unsqueeze(1)
            neighbors[i] = neighbors_proj*attention_scores[i]

        out_nodes_features_list = []

        for i in range(len(neighbors)):
            individual_neighbors = torch.split(neighbors[i], 1, dim=0)
            out_nodes = torch.zeros((1,self.num_out_features), dtype=in_nodes_features.dtype, device=self.device)
            for j in range(len(individual_neighbors)):
                out_nodes += individual_neighbors[j]
            out_nodes_features_list.append(out_nodes)

        #out_nodes_features = torch.Tensor(in_nodes_features.shape[0], in_nodes_features.shape[1])
        #print(len(out_nodes_features))

        out_nodes_features = torch.cat(out_nodes_features_list, dim=0)

        out_nodes_features = self.skip_concat_bias(out_nodes_features)

        return out_nodes_features

        
    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def skip_concat_bias(self, out_nodes_features):

        #if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            #out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_of_features)
        #else:
            #shape = (N, NH, FOUT) -> (N, FOUT)
            #out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)



class NeuralECMModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            #model: str,
            #tokenizer: str,
            device: str):
        super().__init__()

        self.device = device

        #self.query_projection = nn.Linear(query_input_emb_dim, 50)
        self.entity_projection = nn.Linear(ent_input_emb_dim, 50)
        self.query_ent_projection = nn.Bilinear(in1_features=50, in2_features=50, out_features=50)
        #self.para_projection = nn.Linear(para_input_emb_dim, 50)
        self.gnn_layer = GRN(50, 50, device)
        self.rank_score = nn.Linear(50, 1)

        #self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        #self.config = AutoConfig.from_pretrained(model)
        #self.bert = AutoModel.from_pretrained(model, config=self.config)

        #for param in self.bert.parameters():
            #param.requires_grad = False

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor, neighbors: List):

        #print(len(query_text))

        # retrieve the query representation through distilBERT CLS token

        #query_tokens = self.tokenizer.batch_encode_plus(query_text, return_tensors="pt", padding=True, truncation=True)
        #query_tokens_input_ids = query_tokens.input_ids.to(self.device)
        #query_tokens_attention_masks = query_tokens.attention_mask.to(self.device)

        #query_output = self.bert(input_ids=query_tokens_input_ids, attention_mask=query_tokens_attention_masks)
        #query_cls_output = query_output[0][:, 0, :]

        # we project down the query representation to 50 dimension

        #query_embed = self.query_projection(query_cls_output)

        # Next we project down the entity representation to 50 dimension

        ent_embed = self.entity_projection(entity_emb)

        #print(torch.squeeze(query_emb).shape)
        #print(ent_embed.shape)
        #print(neighbors)
        #print('=====================================')
        #break

        node_embeddings = self.query_ent_projection(torch.squeeze(query_emb), ent_embed)

        batch_entity_neighbors_text = []
        batch_entity_neighbors_score = []
        for i,n in enumerate(neighbors):
            entity_neighbors_para_text = []
            entity_neighbors_para_score = []
            
            node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))

            for data in n:
                if 'paraembed' in data:
                    entity_neighbors_para_text.append(data['paraembed'])
                    entity_neighbors_para_score.append(data['parascore'])
                if 'entscore' in data:
                    entity_neighbors_para_score.append(data['entscore'])

            #print(entity_neighbors_para_text)

            if len(entity_neighbors_para_text) > 0:
                #para_text_tokens = self.tokenizer.batch_encode_plus(entity_neighbors_para_text, return_tensors="pt", padding=True, truncation=True)
                #para_text_tokens_input_ids = para_text_tokens.input_ids.to(self.device)
                #para_text_attention_masks = para_text_tokens.attention_mask.to(self.device)

                #para_text_output = self.bert(input_ids=para_text_tokens_input_ids, attention_mask=para_text_attention_masks)
                #para_text_cls_output = para_text_output[0][:, 0, :]

                # We project down the paragraph representation to 50 dimension

                #para_text_embed = self.para_projection(para_text_cls_output)
                para_text_embed = torch.from_numpy(np.array(entity_neighbors_para_text)).float().to(self.device)
                #print(para_text_embed.shape)
                #print(len(entity_neighbors_para_text))

                #node_embed = torch.index_select(node_embeddings, 0, torch.tensor([i]).to(self.device))

                if len(entity_neighbors_para_text) == 1:

                    para_text_embed = para_text_embed.squeeze().unsqueeze(0)
                else:
                    para_text_embed = para_text_embed.squeeze()

                #print(para_text_embed.shape)
                #print(node_embed.shape)

                para_text_embed = torch.cat((para_text_embed, node_embed), 0)
            else:
                para_text_embed = node_embed

            score_embed = torch.Tensor(entity_neighbors_para_score).to(self.device)

            #print(str(para_text_embed.shape)+" "+str(score_embed.shape))

            batch_entity_neighbors_text.append(para_text_embed)
            batch_entity_neighbors_score.append(score_embed)
        
        nodes_features = self.gnn_layer(node_embeddings, batch_entity_neighbors_text, batch_entity_neighbors_score)


        return self.rank_score(nodes_features)


