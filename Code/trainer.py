import torch
import tqdm

class Trainer:
    def __init__(self, model, model_type, optimizer, criterion, scheduler, data_loader, device):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler
        self._data_loader = data_loader
        self._model_type = model_type
        self._device = device

    def make_train_step(self):
        # Builds function that performs a step in the train loop
        def train_step(train_batch):
            # Sets model to TRAIN mode
            self._model.train()

            # Zero the gradients
            self._optimizer.zero_grad()

            # Makes predictions and compute loss
            if self._model_type == 'pairwise':
                batch_score_pos = self._model(
                    query_emb = train_batch[ 'query_emb'].to(self._device),
                    entity_emb = train_batch[ 'ent_emb_pos'].to(self._device),
                    neighbors = train_batch['pos_ent_neighbors']
                )

                batch_score_neg = self._model(
                    query_emb=train_batch['query_emb'].to(self._device),
                    entity_emb=train_batch['ent_emb_neg'].to(self._device),
                    neighbors=train_batch['neg_ent_neighbors']
                )

                batch_loss = self._criterion(
                    batch_score_pos.tanh(),
                    batch_score_neg.tanh(),
                    torch.ones(batch_score_pos.size()).to(self._device)
                )

            elif self._model_type == 'pointwise':
                batch_score = self._model(
                    query_emb=train_batch['query_emb'].to(self._device),
                    entity_emb=train_batch['ent_emb'].to(self._device),
                )

                batch_loss = self._criterion(batch_score, torch.unsqueeze(train_batch['label'], dim=1).to(self._device))
            else:
                raise ValueError('Model type must be `pairwise` or `pointwise`.')

            # Computes gradients
            batch_loss.backward()

            # Updates parameters
            self._optimizer.step()
            self._scheduler.step()

            # Returns the loss
            return batch_loss.item()

        # Returns the function that will be called inside the train loop
        return train_step

    def train(self):
        train_step = self.make_train_step()
        epoch_loss = 0
        num_batch = len(self._data_loader)

        for _, batch in tqdm.tqdm(enumerate(self._data_loader), total=num_batch):
            batch_loss = train_step(batch)
            epoch_loss += batch_loss

        return epoch_loss


