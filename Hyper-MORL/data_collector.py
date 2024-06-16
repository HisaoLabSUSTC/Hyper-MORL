import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/baselines'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

import numpy as np
from copy import deepcopy
import torch

import gym
# import a2c_ppo_acktr
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.model import Policy
import time


'''
Collect Data:
    args: arguments for environment and ppo algorithm
    process_id: index of collector process
    device: torch device
    iteration: number of training iterations
    input_queue: multi-processing queue to pass agent information to sub process
    output_queue: multi-processing queue to pass collected data back to main process.
    done_event: multi-processing events for process synchronization.
'''


def CollectData(args, process_id, device, iteration_num, input_queue, output_queue, done_events):
    # create agent
    torch.set_default_dtype(torch.float64)
    tmp_env = gym.make(args.env_name)
    actor_critic = Policy(
            tmp_env.observation_space.shape,
            tmp_env.action_space,
            base_kwargs={'layernorm' : False},
                obj_num=args.obj_num).to(device)
    tmp_env.close()

    # create environment
    envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes, \
                            gamma=args.gamma, log_dir=None, device=device, allow_early_resets=False, \
                            obj_rms=args.obj_rms, ob_rms = args.ob_rms)
    
    # create rollouts for storing information
    rollouts = RolloutStorage(num_steps = args.num_steps, num_processes = args.num_processes,
                            obs_shape = envs.observation_space.shape, action_space = envs.action_space,
                            recurrent_hidden_state_size = actor_critic.recurrent_hidden_state_size, obj_num=args.obj_num)
    
    raw_accumulated_reward = np.zeros([args.num_steps+1, args.num_processes, args.obj_num])
    norm_accumulated_reward = np.zeros([args.num_steps+1, args.num_processes, args.obj_num])   
    
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # start collect data
    torch.manual_seed(args.seed)
    ideal_point = None 
    for iteration in range(1,iteration_num+1):
        obs_data = []
        obj_data = []
        # get input from message queue
        task = input_queue.get()
        while(task.process_id!=process_id):
            input_queue.put(task)
            task = input_queue.get()

        if task.norm is None:
            results = {}
            results['process_id'] = process_id
            output_queue.put(results)
            done_events[iteration%2].wait() 
            continue          

        # set environment paramters
        envs.venv.ob_rms = deepcopy(task.norm['ob_rms'])
        envs.venv.obj_rms = deepcopy(task.norm['obj_rms'])
        # set the policy parameters
        for name,param in actor_critic.named_parameters():
            param.data = deepcopy(task.network_param[name])
        # collect data
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            if str(action.device)=="cuda":
                obs, _, done, infos = envs.step(action.cpu())
            else:
                obs, _, done, infos = envs.step(action)
            obj_tensor = torch.zeros([args.num_processes, args.obj_num])
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                for info in infos])
            for i, info in enumerate(infos):
                obj_tensor[i] = torch.from_numpy(info['obj'])
                raw_accumulated_reward[step+1,i,:] = raw_accumulated_reward[step,i,:] + info['obj_raw']
                norm_accumulated_reward[step+1,i,:] = norm_accumulated_reward[step,i,:] + info['obj']
            # update the ideal point
            tmp = np.max(raw_accumulated_reward[step+1,:,:],axis=0)
            ideal_point = np.maximum(ideal_point, tmp) if ideal_point is not None else tmp 
            
            # clear the accumulated reward if done
            raw_accumulated_reward[step+1,done,:] = np.zeros_like(raw_accumulated_reward[step+1,done,:])
            norm_accumulated_reward[step+1,done,:] = np.zeros_like(norm_accumulated_reward[step+1,done,:])
    
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, obj_tensor, masks, bad_masks)

            # save the observation and objective values for later normalization
            obj_data.append(infos[0]['obj_data'])
            obs_data.append(infos[0]['obs_data'])

        
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                args.gae_lambda, args.use_proper_time_limits) 
        # store results
        results = {}
        results['process_id'] = process_id
        results['rollouts'] = rollouts
        results['normalization_data'] = {'obs_data':obs_data, 'obj_data':obj_data}
        results['accumulated_reward'] = norm_accumulated_reward
        results['ideal_point'] = ideal_point
        output_queue.put(results)
        done_events[iteration%2].wait()
        rollouts.after_update()
    envs.close()
        
        