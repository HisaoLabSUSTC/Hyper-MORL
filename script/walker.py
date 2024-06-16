import os, sys, signal
import random
import numpy as np
from multiprocessing import Process, Queue, current_process, freeze_support
import argparse
import torch
torch.set_num_threads(1)
parser = argparse.ArgumentParser()
parser.add_argument('--num-seeds', type=int, default=1)


args = parser.parse_args()
commands = []
exp_names = ['default']

task_num = len(exp_names)
num_processes=args.num_seeds*task_num

for k in range(task_num):
    exp_name = exp_names[k]
    # Walker
    random.seed(2000)
    for i in range(args.num_seeds):
        seed = random.randint(0, 1000000)
        command_str = 'python Hyper-MORL/run_morl.py '\
                '--env-name MO-Walker2d-v2 '\
                '--seed {} '\
                '--num-env-steps 30000000 '\
                '--eval-num 1 '\
                '--obj-rms '\
                '--ob-rms '\
                '--raw '.format(seed)
        cmd = command_str+\
        '--warmup-lr 5e-5 '\
        '--psl-lr 5e-5 '\
        '--num-split-obj 5 '\
        '--hypernet-dim 10 '\
        '--reset-logstd '\
        '--W-variance 0 '\
        '--alpha {} '\
        '--save-dir {}/Hyper-MORL/{}/{}'\
            .format(0.15, './results/MO-Walker2d-v2', exp_name, i)
        commands.append(cmd) 



def worker(input, output):
    for cmd in iter(input.get, 'STOP'):
        ret_code = os.system(cmd)
        if ret_code != 0:
            output.put('killed')
            break
    output.put('done')
# Create queues
task_queue = Queue()
done_queue = Queue()

# Submit tasks
for cmd in commands:
    task_queue.put(cmd)

# Submit stop signals
for i in range(num_processes):
    task_queue.put('STOP')

# Start worker processes
for i in range(num_processes):
    Process(target=worker, args=(task_queue, done_queue)).start()

# Get and print results
for i in range(num_processes):
    print(f'Process {i}', done_queue.get())
