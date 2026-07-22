# goal: get a very clearly nonlinear dataset and have my network accurately classify the points!!
# also learn how this wandb business works

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import wandb

X, y = make_moons(n_samples=1000, noise=0.1, random_state=0) # make a data distribution! each (x1,x2) maps to some probability y


