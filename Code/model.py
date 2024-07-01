from typing import Tuple, List
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import warnings
import json
import numpy as np

class GAT(nn.Module):

    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, device, concat=True, activation=nn.ELU(), dropout_prob=0.01, add_skip_connection=False, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_out_features = num_out_features
        self.num_in_features = num_in_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.device = device
        self.num_of_heads = 1

        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)

        self.linear_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, self.num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, self.num_of_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        # End of trainable weights

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation

        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def forward(self, nodes, neighbors, attnt=None):

        out_nodes_features_list = []

        # nodes are the target nodes (entity nodes)
        # neighbors are the source nodes (paragraph nodes and the entity node i.e., self-loop)

        in_nodes_features = nodes

        # apply dropout to all the target nodes as mentioned in the paper
        # shape = (N, FIN), N - number of nodes and FIN - number of input features

        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - number
        # of output features 
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)

        # Attention calculation

        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        for i in range(len(neighbors)):

            #print(len(neighbors[i]))

            # This is "W" in the paper
            neighbors_features_proj = self.linear_proj(neighbors[i]).view(-1, self.num_of_heads, self.num_out_features)
            neighbors_features_proj = self.dropout(neighbors_features_proj)

            # This "a" in the paper
            scores_source = (neighbors_features_proj * self.scoring_fn_source).sum(dim=-1)

            # get the target node for this neighbor (since the targets could belong to different queries for every neighbor)
            # we then repeat the target node shape to match with the number of neighbors i.e. number of source nodes
            # finally we apply leakyrelu to all
            # Then apply softmax to each edge
            # The above process is to calculate the numerator part of the attention

            selected_scores_target = torch.index_select(scores_target, 0, torch.tensor([i]).to(self.device)).to(self.device)

            selected_scores_target = selected_scores_target.repeat_interleave(scores_source.shape[0], dim=0)

            scores_per_edge = self.leakyReLU(scores_source + selected_scores_target)

            exp_scores_per_edge = scores_per_edge.exp()


            # Calculate the denominator.
            # We already have the calculated edge score for the entire neighborhood in exp_scores_per_edge, so just
            # sum it all up

            neighborhood_aware_denominator = exp_scores_per_edge.sum(dim = 0)

            attention_per_edge = exp_scores_per_edge/ (neighborhood_aware_denominator + 1e-16)

            attention_per_edge = attention_per_edge.unsqueeze(-1)


            neighbors[i] = neighbors_features_proj*attention_per_edge

            out_nodes = neighbors[i].sum(dim = 0)

            out_nodes_features_list.append(out_nodes)

        out_nodes_features = torch.cat(out_nodes_features_list, dim=0)

        out_nodes_features = self.skip_concat_bias(out_nodes_features)

        return out_nodes_features

    def skip_concat_bias(self, out_nodes_features):

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N,, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


    def init_params(self):

        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        nn.init.xavier_uniform_(self.scoring_fn_target)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class GRN(nn.Module):

    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, device, concat=True, activation=nn.ELU(), 
            dropout_prob=0.01, add_skip_connection=False, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_out_features = num_out_features
        self.num_in_features = num_in_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.device = device
        self.num_of_heads = 1

        # Trainable weights: linear projection matrix (denoated as "w" in the paper) and bias (not mentioned in the
        # paper but present in the official GAT repo)

        self.linear_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, self.num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def forward(self, nodes, neighbors, attention_scores):
        
        in_nodes_features = nodes

        out_nodes_features_list = []


        # shape = (N, FIN) where N-number of nodes and FIN - number of features for each node
        # We apply drop out to all of the input nodes as mentioned in the paper

        in_nodes_features = self.dropout(in_nodes_features)

        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)


        nodes_features_proj = self.dropout(nodes_features_proj)

        for i in range(len(neighbors)):
            neighbors_proj = self.linear_proj(neighbors[i])
            neighbors_proj = self.dropout(neighbors_proj)

            attention_scores[i] = attention_scores[i].unsqueeze(1)
            neighbors[i] = neighbors_proj*attention_scores[i]

            # aggregation of the neighbors i.e. linear combination

            out_nodes = neighbors[i].sum(dim=0)

            out_nodes_features_list.append(out_nodes)

        out_nodes_features = torch.cat(out_nodes_features_list, dim=0)

        out_nodes_features = self.skip_concat_bias(out_nodes_features)

        return out_nodes_features

        
    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def skip_concat_bias(self, out_nodes_features):

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            #shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)



class NeuralECMModel(nn.Module):

    def __init__(self, ent_input_emb_dim: int, 
            query_input_emb_dim: int, 
            para_input_emb_dim: int, 
            device: str,
            layer_flag: int):
        super().__init__()

        self.device = device

        self.entity_projection = nn.Linear(ent_input_emb_dim, 50)
        self.query_ent_projection = nn.Bilinear(in1_features=50, in2_features=50, out_features=50)
        self.gnn_layer = None

        if layer_flag == 1:
            self.gnn_layer = GRN(50, 50, device)
        else:
            self.gnn_layer = GAT(50, 50, device)
        self.rank_score = nn.Linear(50, 1)


    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor, neighbors: List):

        ent_embed = self.entity_projection(entity_emb)


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

            if len(entity_neighbors_para_text) > 0:

                # We project down the paragraph representation to 50 dimension

                para_text_embed = torch.from_numpy(np.array(entity_neighbors_para_text)).float().to(self.device)

                if len(entity_neighbors_para_text) == 1:

                    para_text_embed = para_text_embed.squeeze().unsqueeze(0)
                else:
                    para_text_embed = para_text_embed.squeeze()


                para_text_embed = torch.cat((para_text_embed, node_embed), 0)
            else:
                para_text_embed = node_embed

            score_embed = torch.Tensor(entity_neighbors_para_score).to(self.device)


            batch_entity_neighbors_text.append(para_text_embed)
            batch_entity_neighbors_score.append(score_embed)


        
        nodes_features = self.gnn_layer(node_embeddings, batch_entity_neighbors_text, batch_entity_neighbors_score)


        return self.rank_score(nodes_features)


