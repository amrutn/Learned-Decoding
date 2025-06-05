import math, time, random
from typing import List

import numpy as np
import torch, torch.nn as nn, torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 0;  torch.manual_seed(SEED);  np.random.seed(SEED);  random.seed(SEED)

from utils.HMM_Teacher import HMMTeacher


# MAIN TRAINING LOOP:
teacher = HMMTeacher()