import sys

sys.path.append("./")
sys.path.append("../")
from mnist_load_model import *
import torch

for weight_type in ["default", "data", "noise"]:
    for seed in [1]:
        num_iters = 7000
        print(num_iters)
        for iter_num in range(num_iters):
            load_model(weight_type, seed, iter_num, num_method="ei")
