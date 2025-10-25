import random
import collections
from typing import Deque, Tuple
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Hiperparametros

ENV_NAME = "Blackjack-v1"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_CAPACITY = 50000
MIN_REPLAY_SIZE = 500
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 20000  # pasos para decaer epsilon
TARGET_UPDATE_FREQ = 1000  # pasos para actualizar target network
NUM_STEPS = 120_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Función para convertir el estado del entorno en un tensor

def state_to_tensor(state):
    # state: (player_sum, dealer_card, usable_ace)
    player_sum, dealer_card, usable_ace = state
    # Normalizar player_sum (2..21 -> 0..1)
    ps = (player_sum - 2) / (21 - 2)
    # dealer_card one-hot (1..10 -> 10-dim)
    d = np.zeros(10, dtype=np.float32)
    d[int(dealer_card) - 1] = 1.0
    ua = 1.0 if usable_ace else 0.0
    arr = np.concatenate(([ps], d, [ua])).astype(np.float32)
    return torch.from_numpy(arr).to(DEVICE)


INPUT_DIM = 1 + 10 + 1  # player_sum_norm + dealer_onehot + usable_ace

class DQN(nn.module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)
