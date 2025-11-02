import random
import collections
from typing import Deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Hiperparametros

ENV_NAME = "Blackjack-v1"
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_CAPACITY = 50000
MIN_REPLAY_SIZE = 500
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 20000  # pasos para decaer epsilon
TARGET_UPDATE_FREQ = 1000  # pasos para actualizar target network
NUM_STEPS = 120_000

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
    return torch.from_numpy(arr).to()


# DQN
INPUT_DIM = 1 + 10 + 1  # player_sum_norm + dealer_onehot + usable_ace

class DQN(nn.Module): # Olvidaste las mayusculas we
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)

Transition = collections.namedtuple("Transition", ["s","a","r","s2","done"])

class ReplayBuffer:
    def __init__(self, capacity: int): #Faltaba un "_"
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def push(self,*args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s = torch.stack([t.s for t in batch])
        a = torch.tensor([t.a for t in batch], dtype=torch.long)
        r = torch.tensor([t.r for t in batch], dtype=torch.float32)
        s2 = torch.stack([t.s2 for t in batch])
        done = torch.tensor([t.done for t in batch], dtype=torch.float32)
        return s, a, r, s2, done #Era r y no t

    def __len__(self):
        return len(self.buffer)

#Se inicia la lista de tamaño fijo, en este caso de 50000, se usa esto para que siempre sea fijo, es más rápido que 
#una lista, esta no va crecer indefinidamente, además de que siempre va a borrar el primer si se llena
replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
print("Replay buffer creado correctamente. Tamaño:", len(replay_buffer))

# Agente
def select_action(policy_net, state_tensor, steps_done):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1 * steps_done / EPS_DECAY)
    if random.random() < eps_threshold:
        return random.randrange(2)
    with torch.no_grad():
        qvals = policy_net(state_tensor.unsqueeze(0))
        return int(torch.argmax(qvals, dim=1).item())
        
def compute_td_loss(policy_net, target_net, replay_buffer, optimizer):
    s, a, r, s2, done = replay_buffer.sample(BATCH_SIZE)
    q_values = policy_net(s)
    q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        q_next = target_net(s2).max(1)[0]
        q_target = r + (1 - done) * GAMMA * q_next

    loss = nn.functional.mse_loss(q_value, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train():
    env = gym.make(ENV_NAME)
    n_actions = env.action_space.n
    policy_net = DQN(INPUT_DIM, n_actions)
    target_net = DQN(INPUT_DIM, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_CAPACITY)

    # Rellenar replay con transiciones aleatorias iniciales
    state, _ = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        s_t = state_to_tensor(state)
        a = random.randrange(n_actions)
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        s2_t = state_to_tensor(s2)
        replay_buffer.push(s_t, a, r, s2_t, done)
        state = env.reset()[0] if done else s2

    steps_done = 0
    losses = []
    episode_rewards = []
    ep_reward = 0.0
    state, _ = env.reset()

    while steps_done < NUM_STEPS:
        s_t = state_to_tensor(state)
        a = select_action(policy_net, s_t, steps_done)
        next_state, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        s2_t = state_to_tensor(next_state)
        replay_buffer.push(s_t, a, r, s2_t, done)

        ep_reward += r

        if len(replay_buffer) >= BATCH_SIZE:
            loss = compute_td_loss(policy_net, target_net, replay_buffer, optimizer)
            losses.append(loss)

        if steps_done % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        steps_done += 1

        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            state, _ = env.reset()
        else:
            state = next_state

        # Logging simple cada cierto número de pasos
        if steps_done % 10000 == 0:
            avg_r = np.mean(episode_rewards[-500:]) if episode_rewards else 0.0
            avg_loss = np.mean(losses[-500:]) if losses else 0.0
            print(f"Steps {steps_done}  avg_reward(last500 eps)={avg_r:.3f}  avg_loss={avg_loss:.4f}")

    # Guardar modelo final
    torch.save(policy_net.state_dict(), "dqn_blackjack.pth")
    print("Entrenamiento terminado. Modelo guardado en dqn_blackjack.pth")
    env.close()
    return policy_net


















