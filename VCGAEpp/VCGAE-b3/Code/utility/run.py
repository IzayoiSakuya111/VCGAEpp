import multiprocessing
import threading
import time
import sys
import os
import glob
import re

cores = int(multiprocessing.cpu_count() / 2)

lr_rate_set = [1e-3,1e-4]
gnn_layer_set= [[100,100,100],[100,100,100,100]]
regs_set = [[1e-5],[1e-4]]

dataset_set = [
    ['JD',  '/vld_buy', '10690', '13465'],
    ['Tmall', '/vld_buy', '17202', '16177'],
    ['UB', '/vld_buy', '20443', '30947'],
]



def exec_command(arg):
    os.system(arg)


# params coarse tuning function
def coarse_tune():
    command = []
    index = 0
    for dataset in dataset_set:
        for lr in lr_rate_set:
            for layer in gnn_layer_set:
                for regs in regs_set:
                    cmd = 'CUDA_VISIBLE_DEVICES=0  python  LightGCN.py --dataset ' + dataset[0] + ' --tst_file ' + dataset[1] \
                          +  ' --n ' + dataset[2] \
                          + ' --m ' + dataset[3] + ' --lr ' + str(lr) \
                          + ' --layer_size ' + str(layer).replace(' ', '') + ' --regs ' + str(regs).replace(' ', '') + ' --gpu_id=0   '
                    print(cmd)
                    command.append(cmd)

        print('\n')

    pool = multiprocessing.Pool(processes=1)
    for cmd in command:
        pool.apply_async(exec_command, (cmd,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    coarse_tune()
