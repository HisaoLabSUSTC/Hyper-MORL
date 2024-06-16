import torch 
import numpy as np
import gym
import time, os, pickle
from copy import deepcopy
from multiprocessing import Process, Queue, Event
from a2c_ppo_acktr import utils
from utils import parallel_evaluate, save_result, generate_prefs, NormalizationRecorder
from utils import LinearScalarization, evaluate, update_linear_schedule
#from target_policy import Policy
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.algo.ppo import PPO
from a2c_ppo_acktr.model import Policy
from hyper_policy import HyperPolicy
from data_collector import CollectData
from torch.multiprocessing import Process, Queue, Event, set_start_method
import environments
import matplotlib.pyplot as plt
class CollectorTask:
    def __init__(self, process_id, norm, network_param):
        self.process_id = process_id
        self.norm = norm
        self.network_param = network_param

def run(args):
    #set_start_method('spawn') 
    #torch.set_default_tensor_type(torch.DoubleTensor)
    # preparation
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")   

    start_time = time.time()
    # uniformly distributed preferences
    uniform_pref = generate_prefs(args.num_split_obj, args.obj_num, fix=True)
    pref_num = max_process_num = len(uniform_pref)
    # number of consumed evaluations to collect trajectories
    eval_per_tra = args.num_steps * args.num_processes
    iter_num_W = int(args.num_env_steps*args.alpha) // eval_per_tra
    iter_num_PSL = int(args.num_env_steps*(1-args.alpha)) // eval_per_tra // pref_num
    # create target network
    tmp_env = gym.make(args.env_name)
    target_policy = Policy(
        tmp_env.observation_space.shape,
        tmp_env.action_space,
        base_kwargs={'layernorm' : False},
            obj_num=args.obj_num)
    #target_policy.to(device).double()
    tmp_env.close()


    with open(os.path.join(args.save_dir, f'args.pkl'), 'wb') as fp:
        pickle.dump(args, fp) 
    print('-----------------------------')
    print('Warm-up stage will perform {} iterations.'.format(iter_num_W))
    print('Pareto set learning will perform {} iterations.'.format(iter_num_PSL))
    '''
        Warm-up stage
    '''
    print('---------- Warm-up stage ----------')
    algo = PPO(
                target_policy,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.warmup_lr,
                eps=1e-5,
                max_grad_norm=args.max_grad_norm)
    
    # create environment
    envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes, \
                            gamma=args.gamma, log_dir=None, device=device, allow_early_resets=False, \
                            obj_rms=args.obj_rms, ob_rms = args.ob_rms)
    print('start:',evaluate(args,target_policy.state_dict(), {'ob_rms':envs.venv.ob_rms, 'obj_rms':envs.venv.obj_rms}))
    
    # create rollouts for storing information
    rollouts = RolloutStorage(num_steps = args.num_steps, num_processes = args.num_processes,
                            obs_shape = envs.observation_space.shape, action_space = envs.action_space,
                            recurrent_hidden_state_size = target_policy.recurrent_hidden_state_size, obj_num=args.obj_num)
    # start collect data
    torch.manual_seed(args.seed)
    iteration = 0
    is_save = lambda x: x%10==0 or x==iter_num_W
    while True:
        # test
        env_param = {'ob_rms':envs.venv.ob_rms, 'obj_rms':envs.venv.obj_rms}
        objs = evaluate(args,target_policy.state_dict(), env_param)
        print('Iteration',iteration,':',objs)
        if is_save(iteration):
            path = os.path.join(args.save_dir,"warmup",str(iteration))
            os.makedirs(path, exist_ok = True)
            # save ep policies & env_param
            torch.save(target_policy.state_dict(), os.path.join(path, f'policy.pt'))
            with open(os.path.join(path, f'env_param.pkl'), 'wb') as fp:
                pickle.dump(env_param, fp) 
            obj_num = len(objs)
            with open(os.path.join(path, 'objs.txt'), 'w') as fp:
                fp.write(('{:5f}' + (obj_num - 1) * ',{:5f}' + '\n').format(*(objs)))    

        if iteration<iter_num_W:
            iteration+=1
        else:
            break
        if iteration==1:
            obs = envs.reset()
            rollouts.obs[0].copy_(obs)
            rollouts.to(device)
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = target_policy.act(
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
    
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, obj_tensor, masks, bad_masks)
        with torch.no_grad():
            next_value = target_policy.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                args.gae_lambda, args.use_proper_time_limits) 
        scalarization = LinearScalarization(1/args.obj_num*np.ones(args.obj_num))
        algo.update(rollouts, scalarization, envs.venv.obj_rms.var)
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            update_linear_schedule( \
                algo.optimizer, (iteration-1) * args.lr_decay_ratio, \
                iter_num_W, args.warmup_lr)
        rollouts.after_update()
    envs.close()


    '''
        Pareto Set Learning Stage
    '''
    print('---------- Pareto Set Learning stage ----------')
    # normalization recorder (for normalizing observations and rewards)
    norm_rec = NormalizationRecorder(args, device)
    norm_rec.norm = env_param
    
    state_dict = torch.load(os.path.join(args.save_dir,"warmup",str(iter_num_W),'policy.pt'))
    # reset exploartion-related parameters
    if args.reset_logstd:
        print('#Info: logstd is reset.')
        state_dict['dist.logstd._bias']*=0
    target_policy.load_state_dict(state_dict)
  
    # create hypernetwork
    hyper_policy = HyperPolicy(
        args.obj_num,
        args.hypernet_dim,
        args.W_variance,
        target_policy,
    )
    target_policy.to(device).double()
    hyper_policy.to(device).double()
    # create PPO algortihm for training 
    if args.parallel:
        from parallel_ppo import PPO_PSL
    else:
        from ppo import PPO_PSL
    algo = PPO_PSL(
        args,
        hyper_policy,
        max_process_num,
        iter_num_PSL
        )
    
    '''
    training
    '''
    # start multi-processing data collector
    processes = []
    input_queue = Queue()
    output_queue = Queue()
    done_events = [Event(),Event()]
    for process_id in range(max_process_num):
        p = Process(target = CollectData, \
            args = (args, process_id, torch.device("cpu"), iter_num_PSL, input_queue, output_queue, done_events))
        p.start()
        processes.append(p)
    # main loop
    iteration = 0
    ideal_point = None
    is_evaluate = lambda x: (x%10 == 0) or x == iter_num_PSL
    while True:
        print('\n-------------Iteration {} -------------'.format(iteration))
        # test and save results
        if is_evaluate(iteration):
            test_pref = generate_prefs(199 if args.obj_num==2 else 27, args.obj_num, fix=True)
            objs, hv = parallel_evaluate(args, hyper_policy, test_pref, norm_rec)
            cost_time = time.time()-start_time
            save_path = os.path.join(args.save_dir, str(iteration))
            save_result(save_path, objs, hv, cost_time, hyper_policy, norm_rec)
            if iteration == iter_num_PSL:
                save_path = os.path.join(args.save_dir, 'final')
                save_result(save_path, objs, hv, cost_time, hyper_policy, norm_rec)               
            print(f'Test results: hv={hv}; time={cost_time}')
        # termination condition
        if iteration == iter_num_PSL:
            break
        else:
            iteration+=1
        st = time.time()
        # preference sampling
        prefs_list = generate_prefs(args.num_split_obj, args.obj_num, fix=False)
        process_num = len(prefs_list)
        #prefs_list = [np.concatenate([pref,[np.random.random()]]) for pref in prefs_list]
        print('Sampled preferences:',prefs_list)

        norm_list = norm_rec.get_norm(prefs_list)
        # give out data to collectors
        for process_id in range(max_process_num):
            if process_id < process_num:
                weights = hyper_policy.get_weights(torch.Tensor(prefs_list[process_id]))
                task = CollectorTask(process_id, norm_list[process_id], weights)
            else:
                task = CollectorTask(process_id, None, None)
            input_queue.put(task)
        rollouts_list = [[] for _ in range(process_num)]
        norm_data_list = [[] for _ in range(process_num)]
        
        cnt_done_workers = 0
        while cnt_done_workers < max_process_num:
            rl_results = output_queue.get()
            pref_id = rl_results['process_id']
            if pref_id<process_num:
                rollouts, normalization_data  =  rl_results['rollouts'], rl_results['normalization_data']
                rollouts_list[pref_id] = rollouts
                norm_data_list[pref_id] = normalization_data
                # update the ideal point
                ideal_point = rl_results['ideal_point'] if ideal_point is None else np.maximum(ideal_point, rl_results['ideal_point'])
            cnt_done_workers += 1
        print('Time for data collection:', time.time()-st)           
        norm_rec.update(norm_data_list)
        
        print('Ideal point=',ideal_point)
        '''
            Update the hypernet using the collected data
        '''
        st = time.time()
    
        for rollouts in rollouts_list:   
            rollouts.to(device)  
        scalarization = LinearScalarization()
        # # decrease learning rate linearly
        utils.update_linear_schedule( \
            algo.optimizer, iteration * args.lr_decay_ratio, \
            iter_num_PSL, args.psl_lr)
        obj_var = norm_rec.get_norm([0,1])[0]['obj_rms'].var
        algo.update(rollouts_list, prefs_list, obj_var, scalarization)
        done_events[(iteration+1)%2].clear()
        done_events[iteration%2].set()
        print('Time for training:', time.time()-st) 