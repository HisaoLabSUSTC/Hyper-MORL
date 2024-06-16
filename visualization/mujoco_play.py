import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))
sys.path.append(os.path.join(base_dir, 'externals/baselines/'))
sys.path.append(os.path.join(base_dir, 'Hyper-MORL/'))
import environments
from a2c_ppo_acktr.model import Policy
from copy import deepcopy
import torch
import gym
from gym import wrappers
import numpy as np
import argparse
import os
import pickle
from a2c_ppo_acktr.model import Policy
from hyper_policy import HyperPolicy
from time import time
from matplotlib import animation
import matplotlib.pyplot as plt

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


torch.set_default_dtype(torch.float64)
print('start')
# define argparser
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='MO-HalfCheetah-v2')
parser.add_argument('--model-path', type = str)
parser.add_argument('--fps', type = float, default = 120.0)
parser.add_argument('--pref1', type = float, default=0)
parser.add_argument('--pref2', type = float, default=0)
parser.add_argument('--pref3', type = float, default=0)
# parse arguments
args = parser.parse_args()

with open(os.path.join(args.model_path, "args.pkl"), "rb") as f:
    settings = pickle.load(f)

env = gym.make(settings.env_name)
env.seed(settings.seed)


with open(os.path.join(args.model_path, "final", "normalization.pkl"), 'rb') as fp:
    normalization = pickle.load(fp)

device = torch.device("cpu")
tmp_env = gym.make(args.env)
target_policy = Policy(
    tmp_env.observation_space.shape,
    tmp_env.action_space,
    base_kwargs={'layernorm' : False},
        obj_num=settings.obj_num)
tmp_env.close()
hyper_policy = HyperPolicy(
    settings.obj_num,
    10,
    0,
    target_policy,
)

state_dict = torch.load(os.path.join(args.model_path, "final", "policy.pt"))
hyper_policy.load_state_dict(state_dict)
hyper_policy.to(device)
hyper_policy.double()
hyper_policy.eval()
if settings.obj_num==3:
    pref = [args.pref1, args.pref2,  args.pref3]
else:
    pref = [args.pref1, args.pref2]

weights = hyper_policy.get_weights(torch.Tensor(pref))
ob_rms = normalization.get_norm([pref])[0]['ob_rms']
actor_critic = target_policy
for name,param in actor_critic.named_parameters():
    param.data = deepcopy(weights[name]) 

#while True:
obs = env.reset()

obj = np.zeros(settings.obj_num)
t = time()
done = False
iter = 0
t = time()
obj_list = []
frames = []
rews = []

obs_list = []
action_list = []
from time import sleep
while not done:
    if ob_rms is not None:
        obs = np.clip((obs - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
    obs_list.append(obs)

    _, action, _, _ = actor_critic.act(torch.Tensor(obs).unsqueeze(0).to(device), None, None, deterministic=True)
    obs, _, done, info = env.step(action.cpu().detach().numpy())
    obj += info['obj']
    iter += 1
    action_list.append(action.cpu().detach().numpy())
    obj_list.append([info['obj'][0]])
    while time() - t < 1 / args.fps:
        pass
    env.render()
    sleep(0.01)
print('test: obj and iter:',obj, iter)
env.close()
