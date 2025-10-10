import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# --- 1. Cấu hình ---
env = gym.make("CartPole-v1", render_mode="rgb_array")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# --- 2. Mạng DQN ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    def forward(self, x):
        return self.net(x)

# --- 3. Chọn hành động ---
def act(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(action_size)
    with torch.no_grad():
        return torch.argmax(policy_net(torch.FloatTensor(state))).item()

# --- 4. Replay Buffer ---
memory = deque(maxlen=10000)
def remember(s, a, r, s2, done):
    memory.append((s, a, r, s2, done))

# --- 5. Huấn luyện mạng ---
def replay(batch_size):
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1, keepdim=True)[0]
        q_targets = rewards + gamma * max_next_q * (1 - dones)

    loss = criterion(q_values, q_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- 6. Khởi tạo ---
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())

criterion = nn.MSELoss()
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
gamma = 0.99

episodes = 300
batch_size = 64
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
target_update = 10

rewards_all = []

# --- 7. Huấn luyện ---
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        replay(batch_size)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    rewards_all.append(total_reward)
    print(f"Tập {episode:3d} | Tổng thưởng: {total_reward:.1f} | Epsilon: {epsilon:.3f}")

env.close()

# --- 8. Biểu đồ reward ---
plt.plot(rewards_all)
plt.xlabel("Tập")
plt.ylabel("Tổng thưởng")
plt.title("Biểu đồ học tập DQN")
plt.show()

# --- 9. Ghi video agent sau huấn luyện ---
import imageio

test_env = gym.make("CartPole-v1", render_mode="rgb_array")
state, _ = test_env.reset()
frames = []

done = False
while not done:
    frame = test_env.render()
    frames.append(frame)
    action = act(state, epsilon=0.0)  # không random, chỉ khai thác
    state, _, done, _, _ = test_env.step(action)

test_env.close()

# Lưu video
imageio.mimsave("cartpole_dqn_result.mp4", frames, fps=30)
print("🎥 Video được lưu thành 'cartpole_dqn_result.mp4'")
