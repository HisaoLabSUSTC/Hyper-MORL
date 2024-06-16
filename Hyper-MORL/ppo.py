import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time


class PPO_PSL():
    def __init__(self, args, 
                 policy, process_num, iteration_num):
        
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.use_clipped_value_loss = True
        self.optimizer = optim.Adam(policy.hypernet.parameters(), lr=args.psl_lr, eps=1e-5)
        self.optimizer_t = optim.Adam(policy.target_policy.parameters(), lr=args.psl_lr, eps=1e-5)
    def update(self, rollouts_list, prefs_list, obj_var, scalarization):
        advantages_list = []
        for i, rollouts in enumerate(rollouts_list):
            op_axis = tuple(range(len(rollouts.returns.shape) - 1))
            scalarization.update_weights(weights = prefs_list[i])
            # recover the raw returns
            returns = rollouts.returns * torch.Tensor(np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.returns
            value_preds = rollouts.value_preds * torch.Tensor(np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.value_preds

            advantages = scalarization.evaluate(returns[:-1]) - scalarization.evaluate(value_preds[:-1])
            advantages_list.append(advantages)
        # normalizae the advantages (version 1)
        mean_val =  torch.cat(advantages_list,0).mean(axis=op_axis)
        std_val = torch.cat(advantages_list,0).std(axis=op_axis)
        for i in range(len(advantages_list)):
            advantages_list[i] = (advantages_list[i] - mean_val) / (std_val
                 + 1e-5)
            
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        for e in range(self.ppo_epoch):
            data_generator_list = []
            st =time.time()
            for i, advantages in enumerate(advantages_list):
                rollouts = rollouts_list[i]
                if self.policy.is_recurrent:
                    data_generator = rollouts.recurrent_generator(
                        advantages, self.num_mini_batch)
                else:
                    data_generator = rollouts.feed_forward_generator(
                        advantages, self.num_mini_batch)
                data_generator_list.append(data_generator)
            rec_forward = np.zeros(2)
            rec_backward = np.zeros(1)
            while True:
                flag = False
                hypernet_loss = None
                for i, data_generator in enumerate(data_generator_list):
                    try:
                        obs_batch, recurrent_hidden_states_batch, actions_batch, \
                        value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                                adv_targ = next(data_generator)
                    except:
                        flag = True
                        break
                    tensor_pref = torch.Tensor(prefs_list[i])
                    self.policy.pref_forward(tensor_pref)   
                    values, action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch)
                    ratio = torch.exp(action_log_probs -
                                    old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean()

                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + \
                            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses,
                                                    value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).mean()

                    target_policy_loss = (value_loss * self.value_loss_coef + action_loss -
                            dist_entropy * self.entropy_coef) 
                    self.optimizer_t.zero_grad()
                    target_policy_loss.backward()
                    if hypernet_loss is None:
                        hypernet_loss = self.policy.cal_loss()
                    else:
                        hypernet_loss = hypernet_loss + self.policy.cal_loss()               
                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()
                    dist_entropy_epoch += dist_entropy.item()
                if not flag:
                    hypernet_loss = hypernet_loss / len(prefs_list)
                    self.optimizer.zero_grad()
                    hypernet_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.hypernet.parameters(),
                                            self.max_grad_norm)
                    self.optimizer.step()
                else:
                    break
           
        num_updates = self.ppo_epoch * self.num_mini_batch
