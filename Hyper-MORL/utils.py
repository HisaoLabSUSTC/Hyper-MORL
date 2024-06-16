import numpy as np
import torch
import gym
from multiprocessing import Pool
import os
from copy import deepcopy

class LinearScalarization():
    def __init__(self, weights = None):
        if weights is not None:
            self.np_weights = weights
            self.tensor_weights = torch.tensor(weights)

    def update_weights(self, weights):
        self.np_weights = weights
        self.tensor_weights = torch.tensor(weights) 

    def evaluate(self, objs):
        if torch.is_tensor(objs):
            result = (objs * self.tensor_weights).sum(axis = -1) 
            return result
        else:
            result = (objs * self.np_weights).sum(axis = -1) 
            return result
                
def point_to_line(p, q1, q2):
    d1 =  np.sqrt(np.sum((p-q1)**2))
    d2 = np.sqrt(np.sum((p-q2)**2))
    v1 = p-q1
    v2 = q2-q1
    v3 = np.dot(v1,v2)/np.dot(v2,v2)*v2
    q3 = q1+v3
    d3 = np.sqrt(np.sum((p-q3)**2))
    if len(np.where(q3<0)[0])+len(np.where(q3>1)[0])>0:
        if np.min([d1,d2])==d1:
            return d1,q1
        else:
            return d2,q2
    else:
        return d3,q3

def generate_prefs(k, m, fix=True):
    prefs = []
    if k==0 and not fix:
        tmp = np.random.rand(m)
        return [tmp/np.sum(tmp)]
    elif k==0 and fix:
        return [np.ones(m)/m]
    if m==2:
         for i in range(k+1):
            r = 1/k
            prefs.append(np.array([i*r,1-i*r]))  
    elif m==3:
        for i in range(k+1):
            for j in range(k+1-i):
                r = 1/k
                prefs.append(np.array([i*r,j*r,1-i*r-j*r]))
    if not fix:
        flag = True
        while flag:
            flag = False
            random_prefs = []
            s = 1/k/2
            prefs_np = np.array(prefs)
            solutions = UniformSphere_ExponentioalPowerDistribution(1000,np.ones([m]),1)*(1+m*s)-s
            dist = np.linalg.norm(prefs_np[:, None, :] - solutions[None, :, :], axis=-1)
            index = np.argmin(dist,axis=0)
            for i in range(len(prefs)):
                res = np.where(index==i)
                if len(res[0])==0:
                    flag=True
                    break
                else:
                    solution = solutions[res[0][0],:]
                    # project the solution to the boundard if  the solution is infeasible
                    if len(np.where(solution<0)[0])+len(np.where(solution>1)[0])>0:
                        min_d=np.inf
                        min_p=None
                        for j in range(m):
                            for k in range(j+1,m):
                                p1 = np.zeros(m); p1[j]=1
                                p2 = np.zeros(m); p2[k]=1
                                d, p = point_to_line(solution, p1,p2)
                                if d<min_d:
                                    min_d=d
                                    min_p=p
                        solution = min_p
                    random_prefs.append(solution)
        return random_prefs
    else:
        return prefs
    

def UniformSphere_ExponentioalPowerDistribution(N, p, R):
    m = p.shape[0]
    u = np.zeros([m,N])
    for i in range(N):
        v=gg6(m,1,0,1,p)
        T = np.sum(np.abs(np.transpose(v))**p)
        u[:,i]=(R/T)**(1/p)*np.transpose(v)
    u = np.abs(np.transpose(u))
    return u

def gg6(n,N,mu,beta,rho):
    mu = mu*np.ones([n])
    beta = beta*np.ones([n])
    x = np.zeros([n,N])
    for i in range(n):
        x[i,:]=mu[i] + (1/np.sqrt(beta[i])) * (np.random.gamma(1/rho[i],1,[1,N])**(1/rho[i])) * ((np.random.random([1,N])<0.5)*2-1)
    return x

import hvwfg
# return sorted indices of nondominated objs
def get_nondominated(pop):
    nondominated = []
    for i in range(pop.shape[0]):
        if ~np.any(np.all((pop - pop[i,:])>=0,axis=1) & np.any((pop - pop[i,:])!=0, axis=1)):
             nondominated.append(i)
    return pop[nondominated,:]

def cal_hv(P):
     # reference point is set as (0,0)
     if np.any(np.all(P>=0,axis=1))==0:
         return 0
     P = P[np.all(P>=0,axis=1),:]
     ideal_point = np.max(P,axis=0)
     return hvwfg.wfg(ideal_point-P,ideal_point)

def evaluate(args, weights, env_param, gamma = 1.0):
    from a2c_ppo_acktr.model import Policy
    eval_env = gym.make(args.env_name)
    objs = np.zeros(args.obj_num)

    torch.set_default_dtype(torch.float64)
    device = torch.device('cpu')
    policy = Policy(
            eval_env.observation_space.shape,
            eval_env.action_space,
            base_kwargs={'layernorm' : False},
            obj_num=args.obj_num).to(device) 
    for name, param in policy.named_parameters():
        param.data = weights[name]    
    
    with torch.no_grad():
        for eval_id in range(args.eval_num):
            eval_env.seed(args.seed + eval_id)
            ob = eval_env.reset()
            done = False
            cnt=0
            while not done:
                if env_param is not None:
                    ob = np.clip((ob - env_param['ob_rms'].mean) / np.sqrt(env_param['ob_rms'].var + 1e-8), -10.0, 10.0)
                _, action, _, _ = policy.act(torch.Tensor(ob).unsqueeze(0), None, None, deterministic=True)
                
                if action.is_cuda:
                    ob, _, done, info = eval_env.step(action.cpu())
                else:
                    ob, _, done, info = eval_env.step(action)

                objs += gamma * info['obj']
                if not args.raw:
                    gamma *= args.gamma
                cnt+=1
    eval_env.close()
    objs /= args.eval_num
    return objs

# def evaluate_wrapper(args, hyper_net, pref_list, norm_list):
#     #args, hyper_net, pref_list, norm_list = args_list
#     objs_list = []
#     for i, pref in enumerate(pref_list):
#         weights = hyper_net.get_weights(torch.Tensor(pref))
#         objs = evaluate(args, weights, norm_list[i])
#         objs_list.append(objs)
#     return objs_list

def evaluate_wrapper(args):
     return evaluate(*args)
    
# def parallel_evaluate(args, hyper_policy, prefs_list, norm_rec, num_process=6):
#     #torch.multiprocessing.set_sharing_strategy('file_system')
#     pref_num = len(prefs_list)
#     seg_siz = int(np.ceil(pref_num //  num_process))
#     norm_list = norm_rec.get_norm(prefs_list)
#     # with Pool(num_process) as pool:
#     #     results = [pool.apply_async(evaluate_wrapper, (args,hyper_policy, prefs_list[i*seg_siz:min((i+1)*seg_siz,pref_num)],
#     #                                                    norm_list[i*seg_siz:min((i+1)*seg_siz,pref_num)])) for i in range(num_process)]
#     #     output = [res.get() for res in results]
#     # objs_result = np.concatenate(output,axis=0)

#     with Pool(num_process) as p:
#         objs_result = p.map(evaluate_wrapper, [(args,hyper_policy, prefs_list[i*seg_siz:min((i+1)*seg_siz,pref_num)],
#                                                  norm_list[i*seg_siz:min((i+1)*seg_siz,pref_num)]) for i in range(num_process)])
#         objs_result = np.concatenate(objs_result,axis=0)
#     if len(prefs_list)<500:
#         hv = cal_hv(get_nondominated(objs_result))
#     else:
#         hv = None
#     return objs_result, hv

def parallel_evaluate(args, hyper_policy, prefs_list, norm_rec, num_process=12,gamma = 1.0):  
    if num_process>20:
        torch.multiprocessing.set_sharing_strategy('file_system')
    with Pool(num_process) as p:
        cnt = 0
        pref_num = len(prefs_list)
        results = []
        norm_list = norm_rec.get_norm(prefs_list)
        while cnt < pref_num:
            weights_list = []
            cnt_end = min(cnt+num_process,pref_num) 
            for i in range(cnt,cnt_end):
                weights = hyper_policy.get_weights(torch.Tensor(prefs_list[i]))
                weights_list.append(weights)
                #print(i,weights['base.critic_linear.bias'],prefs_list[i])
            objs = p.map(evaluate_wrapper, [(args, weights_list[i], norm_list[i+cnt], gamma) for i in range(cnt_end-cnt)])
            results.append(objs)
            cnt = cnt + num_process
        results = np.concatenate(results)
    if len(prefs_list)<500:
        hv = cal_hv(get_nondominated(results))
    else:
        hv = None
    return results, hv

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

import matplotlib.pyplot as plt
import matplotlib

def draw_2d_front(objs, path):
    plt.plot(objs[:,0], objs[:,1],'o')
    plt.savefig(path)
    plt.clf()

def draw_3d_front(objs, path):
    #fig = plt.figure(figsize=(7,6),dpi=150)
    ax = plt.subplot(111, projection='3d')
    ax.set_position(pos=[0.18,0.15,0.7,0.7])
    ax.scatter(objs[:,0], objs[:,1], objs[:,2],s=10, marker='o', linewidths=0.2, facecolors='r', edgecolors='black')
    ax.view_init(elev=30, azim=45)
    plt.savefig(path)
    plt.clf()

def save_result(path, objs, hv, time_cost, model, normalization_rec):
    import pickle
    obj_num = objs.shape[1]

    os.makedirs(path, exist_ok = True)
    print(path)
    # save ep policies & env_param
    torch.save(model.state_dict(), os.path.join(path, 'policy.pt'))
    with open(os.path.join(path, 'normalization.pkl'), 'wb') as fp:
        pickle.dump(normalization_rec, fp) 
    # save objs and hv
    with open(os.path.join(path, 'objs.txt'), 'w') as fp:
        fp.write('hv={:.2f}\n'.format(hv))
        fp.write('time={:.2f}\n'.format(time_cost))
        for obj in objs:
            fp.write(('{:5f}' + (obj_num - 1) * ',{:5f}' + '\n').format(*(obj)))
    # draw
    if obj_num==2:
        draw_2d_front(objs, os.path.join(path, 'ouput.png'))
    else:
        draw_3d_front(objs, os.path.join(path, 'ouput.png'))

'''
This class is used to record the normlization information for the reward and observation from environments.
Policy with different preferences share the normalization parameters.
'''
class NormalizationRecorder:

    def __init__(self, args, device):
        from a2c_ppo_acktr.envs import make_vec_envs
        tmp_envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes, \
            gamma=args.gamma, log_dir=None, device=device, allow_early_resets=False, \
            obj_rms = args.obj_rms, ob_rms = args.ob_rms)
        self.norm = {}
        self.norm['ob_rms'] = deepcopy(tmp_envs.ob_rms) if tmp_envs.ob_rms is not None else None
        self.norm['obj_rms'] = deepcopy(tmp_envs.obj_rms) if tmp_envs.obj_rms is not None else None  
        tmp_envs.close()

    def get_norm(self, prefs):
        res = []
        for pref in prefs:
            res.append(deepcopy(self.norm))
        return res
    
    def update(self, normalization_data_list):
        '''
            update paramters
        '''
        for normalization_data in normalization_data_list:
            obs = np.concatenate(normalization_data['obs_data'],0)
            obj = np.concatenate(normalization_data['obj_data'],0)
            self.norm['ob_rms'].update(obs)
            self.norm['obj_rms'].update(obj)
