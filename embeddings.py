from typing import Tuple
import pickle
import torch
from torch import nn, embedding
    
class EnityEmb(nn.Module):
    def __init__(self, sizes: Tuple[int, int, int, int], emb_size: int, init_size: float = 1e-2) -> None:
        super(EnityEmb, self).__init__()
        self.emb_size = emb_size
            

        # self.enityAtt = EnityAttention(32,8)
        # self.enityAtt = nn.Linear(emb_size, emb_size)
        self.enityAtt = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_size, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ReLU(),
        )
        self.car_type_emb = nn.Embedding(27, emb_size)
        self.user_type_emb = nn.Embedding(29, emb_size)
    
    def forward(self, x, x_car_type = None, x_user_type = None, l = 1e-2):
        
        if x_car_type == None or x_user_type == None:
            enity_emb = l * self.enityAtt(x) / torch.sqrt(torch.tensor(self.emb_size, dtype=torch.float32))

            # enity_emb = x + _enity_emb
            # enity_emb = _enity_emb
            
            #update enity_emb
            # if self.training:
            #     x = x + _enity_emb.detach()
                # x = _enity_emb.detach()
        else:
            x_car_type = x_car_type.int()
            x_user_type = x_user_type.int()
            
            enity_emb = l * (self.car_type_emb(x_car_type) / torch.sqrt(torch.tensor(self.emb_size, dtype=torch.float32)) + \
            self.user_type_emb(x_user_type) / torch.sqrt(torch.tensor(self.emb_size, dtype=torch.float32)) + \
            self.enityAtt(x) / torch.sqrt(torch.tensor(self.emb_size, dtype=torch.float32)))

            # enity_emb = x + _enity_emb
            # enity_emb = _enity_emb
            
            #update enity_emb
            # if self.training:
            #     x = x + _enity_emb.detach()
                # x = _enity_emb.detach()
            
        return enity_emb + x
    
    def update(self, x, index, l = 1e-2):
        self.enity_emb[index] = self.enity_emb[index] + l*x.detach()
        
    
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
    
class TemporalEmb(nn.Module):
    def __init__(self, T_emb_size:str) -> None:
        super(TemporalEmb, self).__init__()
        self.hour_emb = nn.Embedding(24, T_emb_size)
        self.week_emb = nn.Embedding(7, T_emb_size)
        self.month_emb = nn.Embedding(31, T_emb_size)
        self.interval_emb = nn.Sequential(
            nn.Linear(1, T_emb_size),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(4*T_emb_size, T_emb_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        day_of_month = self.month_emb(x[:,0].int())
        day_of_week = self.week_emb(x[:,1].int())
        hour_of_day = self.hour_emb(x[:,2].int())
        intervals = self.interval_emb(x[:,3:])
        
        time = self.fc(torch.cat((day_of_month, day_of_week, hour_of_day, intervals), dim=1))
    
        return time
    
class EnityAttention(nn.Module):
    
    def __init__(self, embed_dim, num_heads):
        super(EnityAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers to project inputs to queries, keys, and values
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # Linear layer to project concatenated outputs of all heads
        self.fc = nn.Linear(embed_dim, embed_dim)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        
        # Project inputs to queries, keys, and values
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Reshape to (batch_size, num_heads, seq_length, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = self.softmax(scores)
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate outputs of all heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        
        # Final linear layer
        output = self.fc(attn_output)
        
        return output
