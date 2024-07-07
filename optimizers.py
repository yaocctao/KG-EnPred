import tqdm
import torch
from torch import nn
from torch import optim

from models import TKBCModel
from regularizers import Regularizer
from datasets import TemporalDataset

class TKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose


    def epoch(self, examples: torch.LongTensor, writer, global_steps):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()
                
                global_steps += 1
                predictions, factors, time = self.model.forward(input_batch)
                truth = input_batch[:, 2].long()
                # truth = torch.tensor([i for i in range(input_batch.shape[0])]).long().cuda()


                l_fit = loss(predictions, truth)
                l_reg = self.emb_regularizer.forward(factors)
                # l_time = torch.zeros_like(l_reg)
                # if time is not None:
                #     l_time = self.temporal_regularizer.forward(time)


                # l = l_fit + l_reg + l_time
                l = l_fit + l_reg
                # l = l_fit
                
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                self.model.update(input_batch)

                if global_steps % 50 == 0:
                    # for name, parms in self.model.named_parameters():
                    #     if parms.grad == None:
                    #         continue
                    #     elif parms.grad.data.is_sparse:
                    #         grad = parms.grad.data.to_dense().flatten()
                    #     else:
                    #         grad = parms.grad.data.flatten()
                    #     writer.add_histogram(name+" grad", grad, global_steps)
                    writer.add_histogram("user grad", self.model.users.grad.flatten())
                    writer.add_histogram("stations grad", self.model.stations.grad.flatten())
                    writer.add_histogram("enity_emb", self.model.enity_emb_tmp.flatten(), global_steps)
                
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{l_fit.item():.2f}',
                    reg=f'{l_reg.item():.2f}',
                    # cont=f'{l_time.item():.0f}'
                )