import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, sys
import torch
import gym
torch.set_num_threads(1)
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'Hyper-MORL/'))
sys.path.append(os.path.join(base_dir, 'externals/baselines/'))
sys.path.append(os.path.join(base_dir, 'externals/baselines/'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail/'))
from utils import parallel_evaluate, generate_prefs
from arguments import get_parser
from a2c_ppo_acktr.model import Policy
from hyper_policy import HyperPolicy
import environments
import pickle
import sys
import argparse

def get_nondominated(pop):
    index = []
    for i in range(pop.shape[0]):
        if ~np.any(np.all((pop - pop[i,:])>=0,axis=1) & np.any((pop - pop[i,:])!=0, axis=1)):
             index.append(i)
    #nondominated=np.array(nondominated)
    # remove repeated solutions
    nondominated = np.unique(pop[index,:], axis=0)
    return nondominated, index


torch.set_num_threads(1)
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str, default='MO-Walker2d-v2', help='Environment name')
parser.add_argument('--run-id', type=int,  default=0)
parser.add_argument('--num-process', type=int, default=10, help='Number of processes for parallel evaluation (default 10)')
input_args = parser.parse_args()
env_name = input_args.env_name
run_id = input_args.run_id
num_process = input_args.num_process

dir_name = os.path.abspath(os.path.dirname(__file__))
result_folder = os.path.join(dir_name,"..","pretrained/%s/Hyper-MORL/default"%(env_name))
save_folder = os.path.join(dir_name, "..", "visualization", "sample")


from argparse import Namespace
test_args = Namespace(raw=True, env_name=env_name, hypernet_dim=10, eval_num=1)
test_args.obj_num = 3 if env_name == "MO-Hopper-v3" else 2

if env_name=="MO-Hopper-v3":
    prefs_list = generate_prefs(199, 3, fix=True)
else:
    prefs_list = generate_prefs(1999, 2, fix=True)

# Save preferences
np.savez(os.path.join(save_folder, "%s_prefs_n%d.npz" % (env_name, len(prefs_list))), prefs = prefs_list)

#As with PGMORL, we use the same random seed as in the training. You can change it to other values.
with open(os.path.join(result_folder, str(run_id), "args.pkl"), 'rb') as f:
    training_args = pickle.load(f)
test_args.seed = training_args.seed

state_dict = torch.load(os.path.join(result_folder, str(run_id), "final", "policy.pt"))
with open(os.path.join(result_folder, str(run_id), "final", "normalization.pkl"), 'rb') as f:
    norm_rec = pickle.load(f)
torch.set_default_dtype(torch.float64)
tmp_env = gym.make(env_name)
target_policy = Policy(
    tmp_env.observation_space.shape,
    tmp_env.action_space,
    base_kwargs={'layernorm' : False},
        obj_num=test_args.obj_num)    
tmp_env.close()
hyper_policy = HyperPolicy(
    test_args.obj_num,
    test_args.hypernet_dim,
    0,
    target_policy
)
hyper_policy.load_state_dict(state_dict)
#prefs_list = [pref.astype(np.float32) for pref in prefs_list]
print("The number of sampled preferences is",len(prefs_list))
objs, _ = parallel_evaluate(test_args, hyper_policy, prefs_list, norm_rec, num_process=num_process, gamma=1.0)

print("\n---------------------------------------")

# Save objective values
# path = "%s/%s_objs_n%d_%d.txt"%(save_folder, env_name, len(prefs_list), run_id)
# with open(path, 'w') as fp:
#     for sample_obj in objs:
#         fp.write(('{:5f}' + (test_args.obj_num - 1) * ',{:5f}' + '\n').format(*(sample_obj)))
# print(objs)
hyper_policy.float()
res, nondominated_index = get_nondominated(objs)
flag = np.zeros(objs.shape[0])
flag[nondominated_index]=1
np.savez(os.path.join(save_folder, "%s_objs_n%d_%d.npz" % (env_name, len(prefs_list), run_id)), objs = objs, flag = flag)


# Save parameters
for i, pref in enumerate(prefs_list):
    #policy.set_preference(torch.Tensor(pref).to(device))
    net_param, _ = hyper_policy.hypernet(torch.Tensor(pref).float())
    if i==0:
        params = torch.zeros((len(prefs_list),net_param.shape[0]))
    params[i,:]=net_param

params = params.detach().numpy()
np.savez(os.path.join(save_folder, "%s_params_n%d_%d.npz" % (env_name, len(prefs_list), run_id)), params = params)
