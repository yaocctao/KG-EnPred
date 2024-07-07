from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import pickle
import torch.nn.functional as F
from embeddings import EnityEmb, TemporalEmb
import torch
from torch import nn
import numpy as np

class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries, filters, year2id = {},
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)
                    
                    scores = q @ rhs 
                    targets = self.score(these_queries)

                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(int(query[0].item()), int(query[1].item()), int(query[3].item()), int(query[4].item()), int(query[5].item()))]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
    
class KGEM(nn.Module):
    
    def __init__(self, sizes: Tuple[int, int, int, int], emb_size: int, enity_info_path: str, T_emb_size: int) -> None:
        super(KGEM, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, emb_size, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[0], sizes[0], sizes[0], sizes[3]] # without no_time_emb
        ])
        self.enity_info = pickle.load(open(enity_info_path, 'rb'))
        self.week_emb = nn.Embedding(7, T_emb_size)
        self.month_emb = nn.Embedding(31, T_emb_size)
        self.interval_emb = nn.Sequential(
            nn.Linear(1, T_emb_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        pass
        
    
    def similarityByJaccard(self, query, k):
        n_samples, n_time_dim, n_features = query.shape
        enity_num = self.enity_info.shape[0]

        query = query.view(n_samples, -1)
        enity_info = self.enity_info.view(enity_num, -1)
        
        # compute Jaccard similarity
        intersection = torch.sum(torch.min(query.unsqueeze(1), enity_info.unsqueeze(0)), dim=2)
        union = torch.sum(torch.max(query.unsqueeze(1), enity_info.unsqueeze(0)), dim=2)
        jaccard_sim = intersection / union
        
        # find k most similar samples
        top_k_similarities, top_k_similarities_indices = torch.topk(jaccard_sim, k, dim=1, largest=True, sorted=True)
        
        # find k most unsimilar samples
        top_k_unsimilarities, top_k_unsimilarities_indices = torch.topk(jaccard_sim, k, dim=1, largest=False, sorted=True)
        
        return top_k_similarities_indices, top_k_unsimilarities_indices
    
    def similarityByDTW(self):
        pass
        
    def similarityByEuclidean(self):
        pass
    
    def similarityByPerson(self):
        pass



class KGEM(nn.Module):
    
    def __init__(self, sizes: Tuple[int, int, int, int], emb_size: int, enity_info_path: str, T_emb_size: int) -> None:
        super(KGEM, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, emb_size, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[0], sizes[0], sizes[0], sizes[3]] # without no_time_emb
        ])
        self.enity_info = pickle.load(open(enity_info_path, 'rb'))
        self.week_emb = nn.Embedding(7, T_emb_size)
        self.month_emb = nn.Embedding(31, T_emb_size)
        self.interval_emb = nn.Sequential(
            nn.Linear(1, T_emb_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        pass
        
    
    def similarityByJaccard(self, query, k):
        n_samples, n_time_dim, n_features = query.shape
        enity_num = self.enity_info.shape[0]

        query = query.view(n_samples, -1)
        enity_info = self.enity_info.view(enity_num, -1)
        
        # compute Jaccard similarity
        intersection = torch.sum(torch.min(query.unsqueeze(1), enity_info.unsqueeze(0)), dim=2)
        union = torch.sum(torch.max(query.unsqueeze(1), enity_info.unsqueeze(0)), dim=2)
        jaccard_sim = intersection / union
        
        # find k most similar samples
        top_k_similarities, top_k_similarities_indices = torch.topk(jaccard_sim, k, dim=1, largest=True, sorted=True)
        
        # find k most unsimilar samples
        top_k_unsimilarities, top_k_unsimilarities_indices = torch.topk(jaccard_sim, k, dim=1, largest=False, sorted=True)
        
        return top_k_similarities_indices, top_k_unsimilarities_indices
    
    def similarityByDTW(self):
        pass
        
    def similarityByEuclidean(self):
        pass
    
    def similarityByPerson(self):
        pass


class TeLM(TKBCModel):
    """2nd-grade Temporal Knowledge Graph Embeddings using Geometric Algebra
        :::     Scoring function: <h, r, t_conjugate, T>
        :::     2-grade multivector = scalar + Imaginary * e_1 + Imaginary * e_2 + Imaginary * e_3 + Imaginary * e_12
    """
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, time_granularity: int = 1,
):
        super(TeLM, self).__init__()
        
        self.sa1 = SpatialAttention()
        
        self.sizes = sizes
        self.rank = rank
        self.cov_output_chennel = 3
        self.input_emb_size = self.cov_output_chennel * self.rank
        self.emb_height = 20
        self.emb_width = 50
        self.stride = 70
        self.output_emb_size = self.rank * 70
        self.W = nn.Embedding(rank,1,sparse=True)
        self.W.weight.data *= 0
        # self.X = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (800, 800, 800)), 
        #                             dtype=torch.float, device="cuda", requires_grad=True))

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[0], sizes[0], sizes[0], sizes[3]] # without no_time_emb
        ])
        for i in range(len(self.embeddings)):
            self.embeddings[i].weight.data *= init_size
        
        self.conv21 = torch.nn.Conv2d(4, 2, kernel_size=1, stride=1, bias=True)
        self.conv22 = torch.nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=True)

        self.sum_pool1 = torch.nn.LPPool2d(1,(4,1),stride=1)
        
        self.sum_pool = torch.nn.LPPool1d(1,self.stride,stride=self.stride)
        
        self.linear_E = torch.nn.Linear(self.input_emb_size,self.output_emb_size)
        self.linear_R = torch.nn.Linear(self.input_emb_size,self.output_emb_size)
        self.linear_d = torch.nn.Linear(self.input_emb_size,self.output_emb_size)
        
        self.bn_MRN_e1 = torch.nn.BatchNorm2d(2)
        self.bn_DRN_e1 = torch.nn.BatchNorm2d(4)
        
        self.branch_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU())
        self.branch_r = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU())
        self.branch_d = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU())
            
        self.branch1_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch1_r = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch1_d = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch1_s = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch1_s1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
            
        self.branch2_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch2_r = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch2_d = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch2_s = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
            
        self.branch3_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch3_r = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch3_d = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch3_s = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())   
            
        self.branch4_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch4_r = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch4_d = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch4_s = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        
        
        self.w = nn.Parameter(torch.ones(2))

        self.no_time_emb = no_time_emb

        self.time_granularity = time_granularity

    @staticmethod
    def has_time():
        return True
    
    def QDM(self,e1, r, d):
        e1 = self.branch_e1(e1)
        r = self.branch_r(r)
        d = self.branch_d(d)
        
        shared = torch.zeros(e1.shape).cuda()
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        h = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate
        
        h = h.view(-1, 1, 1, self.rank)
        
        return h, r, d, shared
    
    def MRN(self, x, is_left=False):
        if is_left:
            rhs = self.embeddings[0](x[:, 2])
            rhs1 = self.embeddings[3](x[:, 2])
            rhs2 = self.embeddings[4](x[:, 2])
            rhs3 = self.embeddings[5](x[:, 2])

            h1 = rhs.view(-1,1,1,self.rank)
            h2 = rhs1.view(-1,1,1,self.rank)
            h3 = rhs2.view(-1,1,1,self.rank)
            h4 = rhs3.view(-1,1,1,self.rank)
        else:
            lhs = self.embeddings[0](x[:, 0])
            lhs1 = self.embeddings[3](x[:, 0])
            lhs2 = self.embeddings[4](x[:, 0])
            lhs3 = self.embeddings[5](x[:, 0])

            h1 = lhs.view(-1,1,1,self.rank)
            h2 = lhs1.view(-1,1,1,self.rank)
            h3 = lhs2.view(-1,1,1,self.rank)
            h4 = lhs3.view(-1,1,1,self.rank)
        
        h = torch.cat((h1,h2,h3,h4),1)
        
        h = self.conv21(h)
        h = self.bn_MRN_e1(h)
        h = F.relu(h)
        h = self.conv22(h)  
        e1 = h.view(-1,1,self.emb_height,self.emb_width)     
      
        return e1, h1
    
    def DRN(self, h, h1, h2, h3, h4):
        e1_At = torch.sigmoid(h)
        e1 = torch.cat((torch.mul(h1,e1_At),torch.mul(h2,e1_At),torch.mul(h3,e1_At),torch.mul(h4,e1_At)),1)
        e1 = self.bn_DRN_e1(e1)
        e1 = F.relu(e1)
        e1 = e1.view(-1,4,self.rank)
        #e1_1 = e1_1.view(-1,5,200)
        e1 = self.sum_pool1(e1)

        h_MR = e1.view(-1,1,self.emb_height,self.emb_width)
        
        return h_MR
    
    def IEM(self, h_MR, r, d, shared):
        e1 = self.branch1_e1(h_MR)
        r = self.branch1_r(r)
        d = self.branch1_d(d)
        shared = self.branch1_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate   
        
        e1 = self.branch2_e1(e1)
        r = self.branch2_r(r)
        d = self.branch2_d(d)
        shared = self.branch2_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch3_e1(e1)
        r = self.branch3_r(r)
        d = self.branch3_d(d)
        shared = self.branch3_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch4_e1(e1)
        r = self.branch4_r(r)
        d = self.branch4_d(d)
        shared = self.branch4_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        entity = e1.view(-1, self.input_emb_size)
        relation = r.view(-1, self.input_emb_size)
        temporal = d.view(-1, self.input_emb_size)
        
        e1 = self.linear_E(entity)
        full_rel = self.linear_R(relation)
        d = self.linear_d(temporal)
        e1 = F.tanh(e1)
        full_rel = F.tanh(full_rel)
        d = F.tanh(d) 
        r_ds = torch.mul(torch.mul(e1,full_rel),d)
        r_ds = r_ds.view(-1,1,self.output_emb_size)
        
        return r_ds, entity, relation, temporal
    
    def score(self, x):

        rel = self.embeddings[1](x[:, 1])
        rnt = self.embeddings[6](x[:, 3] // self.time_granularity)
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity)
        #rnt = self.embeddings[3](x[:, 1])
        
        d = time.view(-1,1,self.emb_height,self.emb_width) 
#MRN_e1        
        e1, h1, h2, h3, h4 = self.MRN(x)
        r = rel.view(-1,1,self.emb_height,self.emb_width)
#QDM        
        h, r, d, shared = self.QDM(e1, r, d)
#DRN_e1       
        h_MR = self.DRN(h, h1, h2, h3, h4)
#IEM
        r_ds, _, _, _ = self.IEM(h_MR, r, d, shared)
        
        x = self.sum_pool(r_ds)
        x = F.normalize(x,p=2,dim=2)
        x = x.view(-1,self.rank)
        x = torch.mul(x, rnt)

        return torch.sum(x *rhs, 1, keepdim = True)

    def forward(self, x):
        rel = self.embeddings[1](x[:, 1])
        rnt = self.embeddings[6](x[:, 3] // self.time_granularity)
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity) 
        #rnt = self.embeddings[3](x[:, 1])
        
        d = time.view(-1,1,self.emb_height,self.emb_width) 
#MRN_e1        
        e1, h1, h2, h3, h4 = self.MRN(x)
        r = rel.view(-1,1,self.emb_height,self.emb_width)
#QDM             
        h, r, d, shared = self.QDM(e1, r, d)
#DRN_e1       
        h_MR = self.DRN(h, h1, h2, h3, h4)
#IEM
        r_ds, entity, relation, temporal = self.IEM(h_MR, r, d, shared)
        right = self.embeddings[0].weight
        x = self.sum_pool(r_ds)
        x = F.normalize(x,p=2,dim=2)
        x = x.view(-1,self.rank)
        x = torch.mul(x, rnt)

        regularizer = (entity, relation,temporal,rhs)
        return ((
               x @ right.t()
            ), regularizer,
               self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        rel = self.embeddings[1](queries[:, 1])
        rnt = self.embeddings[6](queries[:, 3] // self.time_granularity)
        time = self.embeddings[2](queries[:, 3] // self.time_granularity) 
        
        d = time.view(-1,1,self.emb_height,self.emb_width) 
#MRN_e1        
        e1, h1, h2, h3, h4 = self.MRN(queries)    
        r = rel.view(-1,1,self.emb_height,self.emb_width)
#QDM        
        h, r, d, shared = self.QDM(e1, r, d)
#DRN_e1       
        h_MR = self.DRN(h, h1, h2, h3, h4)
#IEM
        r_ds, _, _, _ = self.IEM(h_MR, r, d, shared)
        x = self.sum_pool(r_ds)
        x = F.normalize(x,p=2,dim=2)
        x = x.view(-1,self.rank)
        x = torch.mul(x, rnt)
       
        return x

    def get_lhs_queries(self, queries: torch.Tensor):
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3] // self.time_granularity)
        rnt = self.embeddings[6](queries[:, 3] // self.time_granularity)
        
        d = time.view(-1,1,self.emb_height,self.emb_width) 
#MRN_e1        
        e1, h1, h2, h3, h4 = self.MRN(x, is_left=True)
        r = rel.view(-1,1,self.emb_height,self.emb_width)
#QDM      
        h, r, d, shared = self.QDM(e1, r, d)
#DRN_e1       
        h_MR = self.DRN(h, h1, h2, h3, h4)
#IEM
        r_ds, _, _, _ = self.IEM(h_MR, r, d, shared)
        x = self.sum_pool(r_ds)
        x = F.normalize(x,p=2,dim=2)
        x = x.view(-1,self.rank)
        x = torch.mul(x, rnt)    

        return x
    
class KGEnPred(TKBCModel):
    
    def __init__(self, sizes: Tuple[int, int, int, int], rank: int, no_time_emb=False, init_size: float = 1e-2, time_granularity: int = 1):
        super(KGEnPred, self).__init__()
        self.sa1 = SpatialAttention()
        
        self.sizes = sizes
        self.rank = rank
        self.cov_output_chennel = 3
        self.input_emb_size = self.cov_output_chennel * self.rank
        self.emb_height = 20
        self.emb_width = 50
        self.stride = 70
        self.output_emb_size = self.rank * 70
        # 流式处理时 car信息从milvus中取
        # car_info_path = './data/fujian/car_info.pickle'
        
        self.enity_emb = EnityEmb(sizes, rank)
        self.relation_emb = nn.Embedding(sizes[1], rank, sparse=True)
        self.temporal_emb = TemporalEmb(self.rank)
        
        self.conv21 = torch.nn.Conv2d(1, 2, kernel_size=1, stride=1, bias=True)
        self.conv22 = torch.nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=True)

        self.sum_pool1 = torch.nn.LPPool2d(1,(4,1),stride=1)
        
        self.sum_pool = torch.nn.LPPool1d(1,self.stride,stride=self.stride)
        
        self.linear_E = torch.nn.Linear(self.input_emb_size,self.output_emb_size)
        self.linear_R = torch.nn.Linear(self.input_emb_size,self.output_emb_size)
        self.linear_d = torch.nn.Linear(self.input_emb_size,self.output_emb_size)
        
        self.bn_MRN_e1 = torch.nn.BatchNorm2d(2)
        self.bn_DRN_e1 = torch.nn.BatchNorm2d(1)
        
        self.branch_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU())
        self.branch_r = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU())
        self.branch_d = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU())
            
        self.branch1_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch1_r = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch1_d = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch1_s = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch1_s1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
            
        self.branch2_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch2_r = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch2_d = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch2_s = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
            
        self.branch3_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch3_r = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch3_d = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch3_s = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())   
            
        self.branch4_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch4_r = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch4_d = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch4_s = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.cov_output_chennel, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())

        self.no_time_emb = no_time_emb
        
        self.register_buffer('enity_emb_tmp', torch.randn((sizes[0],rank), device='cuda'))

    def forward(self, x):
        self.users = self.enity_emb_tmp[x[:, 2].int()].clone().requires_grad_(True)
        self.stations = self.enity_emb_tmp[x[:, 0].int()].clone().requires_grad_(True)
        rel = self.relation_emb(x[:, 1].int())
        rhs = self.enity_emb(self.users, x[:, 7], x[:, 8])
        rnt = self.temporal_emb(x[:, 3:7])
        
        d = rnt.view(-1,1,self.emb_height,self.emb_width)
        #MRN_e1
        e1, h1 = self.MRN(self.stations)
        r = rel.view(-1,1,self.emb_height,self.emb_width)
        #QDM
        h, r, d, shared = self.QDM(e1, r, d)
        #DRN_e1
        h_MR = self.DRN(h, h1)
        #IEM
        r_ds, entity, relation, temporal = self.IEM(h_MR, r, d, shared)
        right = self.enity_emb_tmp
        x = self.sum_pool(r_ds)
        x = F.normalize(x, p=2, dim=2)
        x = x.view(-1,self.rank)
        
        x = torch.mul(x, rnt)

        regularizer = (entity, relation, temporal, rhs)
        
        return ((
               x @ right.t()
            ), regularizer,
               None
        )

    def update(self, x, lr=1e-2):
        self.enity_emb_tmp[x[:, 2].int()] = self.enity_emb_tmp[x[:, 2].int()] + lr * self.users.grad
        self.enity_emb_tmp[x[:, 0].int()] = self.enity_emb_tmp[x[:, 0].int()] + lr * self.stations.grad
    
    def score(self, x):
        stations = self.enity_emb_tmp[x[:, 2].int()]
        users = self.enity_emb_tmp[x[:, 0].int()]
        rel = self.relation_emb(x[:, 1].int())
        rhs = self.enity_emb(stations, x[:, 7], x[:, 8])
        rnt = self.temporal_emb(x[:, 3:7])
        
        d = rnt.view(-1,1,self.emb_height,self.emb_width) 
#MRN_e1        
        e1, h1 = self.MRN(users)
        r = rel.view(-1,1,self.emb_height,self.emb_width)
#QDM        
        h, r, d, shared = self.QDM(e1, r, d)
#DRN_e1       
        h_MR = self.DRN(h, h1)
#IEM
        r_ds, _, _, _ = self.IEM(h_MR, r, d, shared)
        
        x = self.sum_pool(r_ds)
        x = F.normalize(x,p=2,dim=2)
        x = x.view(-1,self.rank)
        x = torch.mul(x, rnt)

        return torch.sum(x *rhs, 1, keepdim = True)
    
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.enity_emb_tmp[
                chunk_begin:chunk_begin + chunk_size
                ].transpose(0, 1)
        
    def get_queries(self, queries: torch.Tensor):
        users = self.enity_emb_tmp[queries[:, 0].int()]
        rel = self.relation_emb(queries[:, 1].int())
        rnt = self.temporal_emb(queries[:, 3:7])
        
        d = rnt.view(-1,1,self.emb_height,self.emb_width) 
#MRN_e1        
        e1, h1 = self.MRN(users)
        r = rel.view(-1,1,self.emb_height,self.emb_width)
#QDM        
        h, r, d, shared = self.QDM(e1, r, d)
#DRN_e1       
        h_MR = self.DRN(h, h1)
#IEM
        r_ds, _, _, _ = self.IEM(h_MR, r, d, shared)
        x = self.sum_pool(r_ds)
        x = F.normalize(x,p=2,dim=2)
        x = x.view(-1,self.rank)
        x = torch.mul(x, rnt)
       
        return x

    def get_lhs_queries(self, queries: torch.Tensor):
        users = self.enity_emb_tmp[queries[:, 0].int()]
        rel = self.relation_emb(queries[:, 1].int())
        rnt = self.temporal_emb(queries[:, 3:7])
        
        d = rnt.view(-1,1,self.emb_height,self.emb_width) 
#MRN_e1        
        e1, h1 = self.MRN(users, queries[:, 7], queries[:, 8])
        r = rel.view(-1,1,self.emb_height,self.emb_width)
#QDM      
        h, r, d, shared = self.QDM(e1, r, d)
#DRN_e1       
        h_MR = self.DRN(h, h1)
#IEM
        r_ds, _, _, _ = self.IEM(h_MR, r, d, shared)
        x = self.sum_pool(r_ds)
        x = F.normalize(x,p=2,dim=2)
        x = x.view(-1,self.rank)
        x = torch.mul(x, rnt)    

        return x
    
    def QDM(self,e1, r, d):
        e1 = self.branch_e1(e1)
        r = self.branch_r(r)
        d = self.branch_d(d)
        
        shared = torch.zeros(e1.shape).cuda()
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        h = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate
        
        h = h.view(-1, 1, 1, self.rank)
        
        return h, r, d, shared
            
    def MRN(self, x, x_car_type=None, x_user_type=None):
        if x_car_type == None or x_user_type == None:
            lhs = self.enity_emb(x)
            h1 = lhs.view(-1,1,1,self.rank)
        else:
            rhs = self.enity_emb(x, x_car_type, x_user_type)
            h1 = rhs.view(-1,1,1,self.rank)
            
 
        h = self.conv21(h1)
        h = self.bn_MRN_e1(h)
        h = F.relu(h)
        h = self.conv22(h)  
        e1 = h.view(-1,1,self.emb_height,self.emb_width)
      
        return e1, h1
    
    def DRN(self, h, h1):
        e1_At = torch.sigmoid(h)
        e1 = torch.mul(h1,e1_At)
        e1 = self.bn_DRN_e1(e1)
        e1 = F.relu(e1)
        h_MR = e1.view(-1,1,self.emb_height,self.emb_width)
        
        return h_MR
    
    def IEM(self, h_MR, r, d, shared):
        e1 = self.branch1_e1(h_MR)
        r = self.branch1_r(r)
        d = self.branch1_d(d)
        shared = self.branch1_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate   
        
        e1 = self.branch2_e1(e1)
        r = self.branch2_r(r)
        d = self.branch2_d(d)
        shared = self.branch2_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch3_e1(e1)
        r = self.branch3_r(r)
        d = self.branch3_d(d)
        shared = self.branch3_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch4_e1(e1)
        r = self.branch4_r(r)
        d = self.branch4_d(d)
        shared = self.branch4_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        entity = e1.view(-1, self.input_emb_size)
        relation = r.view(-1, self.input_emb_size)
        temporal = d.view(-1, self.input_emb_size)
        
        e1 = self.linear_E(entity)
        full_rel = self.linear_R(relation)
        d = self.linear_d(temporal)
        e1 = F.tanh(e1)
        full_rel = F.tanh(full_rel)
        d = F.tanh(d) 
        r_ds = torch.mul(torch.mul(e1,full_rel),d)
        r_ds = r_ds.view(-1,1,self.output_emb_size)
        
        return r_ds, entity, relation, temporal
