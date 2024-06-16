import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from hypernet import Hypernet
from copy import deepcopy
class HyperPolicy(nn.Module):
    def __init__(self, num_objs, num_features, W_variance, target_policy:nn.Module):
        super(HyperPolicy, self).__init__()
        self.target_policy = target_policy
        self.hypernet = Hypernet(target_policy, num_objs, num_features, W_variance)
        self.train()
    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.target_policy.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError
    
    def cal_loss(self):
        grads_list = []
        with torch.no_grad():
            for name, param in self.target_policy.named_parameters():
                if param.grad is None:
                    grads_list.append(torch.zeros_like(param.data).flatten())
                else:
                    grads_list.append(param.grad.flatten())
            grad_tensor = torch.cat(grads_list)
        #print(grad_tensor)
        return torch.dot(grad_tensor, self.flat_params)

    def pref_forward(self, pref):
        self.flat_params, self.sorted_params = self.hypernet(pref)
        with torch.no_grad():
            for name, param in self.target_policy.named_parameters():
                param.copy_(self.sorted_params[name]) 
        # for name, param in self.target_policy.named_parameters():
        #     print(param.data_ptr(), self.sorted_params[name].data_ptr())

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        return self.target_policy.act(inputs, rnn_hxs, masks, deterministic)

    def get_value(self, inputs, rnn_hxs, masks):
        return self.target_policy.get_value(inputs, rnn_hxs, masks)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        return self.target_policy.evaluate_actions(inputs, rnn_hxs, masks, action)

    def get_weights(self, pref):
        with torch.no_grad():
            flat_params, sorted_params = self.hypernet(pref)
        #print(sorted_params['dist.logstd._bias'])
        return sorted_params