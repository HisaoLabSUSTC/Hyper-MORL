import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class Hypernet(nn.Module):
    def __init__(self, target_policy, num_objs, num_features=5, W_variance=0):
        super(Hypernet, self).__init__()
        hyper_hidden_size=128
        self.pref_mlp = nn.Sequential(
            nn.Linear(num_objs, hyper_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hyper_hidden_size, hyper_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hyper_hidden_size, num_features),
        )
        self.target_policy_info = []
        self.num_params = 0
        flat_params_list = []
        for name, param in target_policy.named_parameters():
            self.target_policy_info.append((name, param.shape))
            self.num_params += torch.numel(param)
            flat_params_list.append(param.flatten())
        # # initialization
        self.W = nn.Parameter((torch.rand(num_features, self.num_params)*2-1)*W_variance)
        self.b = nn.Parameter(torch.zeros(self.num_params))
        self.b.data = deepcopy(torch.cat(flat_params_list).detach())
        
    def forward(self, pref):
        features = self.pref_mlp(pref)
        flat_params = torch.matmul(features, self.W)+self.b
        #print('grad=',weights.grad_fn)
        sorted_params = {}
        cnt = 0
        for name, param_shape in self.target_policy_info:
            l = np.prod(list(param_shape))
            sorted_params[name] = flat_params[cnt:cnt+l].reshape(
                param_shape)
            cnt = cnt + l
        
        return flat_params, sorted_params
