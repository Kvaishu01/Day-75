import streamlit as st
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Policy Gradient - MountainCar", layout="wide")

st.title("🏔️ Policy Gradient (REINFORCE) – MountainCar-v0")
st.write("""
Train an agent using **Policy Gradient (REINFORCE)** to solve the 
**MountainCar-v0** environment from Gymnasium.
""")

# Sidebar controls
episodes = st.sidebar.slider("Number of Training Episodes", 100, 1000, 400, 50)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.001, 0.005, 0.01, 0.02], value=0.01)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.8, 0.999, 0.99)
run_training = st.sidebar.button("🚀 Start Training")

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 36),
            nn.ReLU(),
            nn.Linear(36, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

def select_action(policy, state):
    state = torch.FloatTensor(state)
    probs = policy(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

# Train the agent
if run_training:
    st.info("Training in progress... please wait ⏳")

    env = gym.make('MountainCar-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs, rewards = [], []
        total_reward = 0

        for _ in range(300):
            action, log_prob = select_action(policy, state)
            next_state, reward, done, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward
            state = next_state
            if done or truncated:
                break

        # Compute discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Policy gradient update
        loss = []
        for log_prob, R in zip(log_probs, discounted_rewards):
            loss.append(-log_prob * R)
        loss = torch.stack(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            st.write(f"✅ Episode {episode + 1}/{episodes} | Total Reward: {total_reward:.2f}")

    env.close()

    # Plot performance
    fig, ax = plt.subplots()
    ax.plot(episode_rewards)
    ax.set_title("Policy Gradient Training Progress")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Total Reward")
    st.pyplot(fig)

    # Save model
    torch.save(policy.state_dict(), "policy_gradient_mountaincar.pth")
    st.success("🎯 Training Complete! Model saved as `policy_gradient_mountaincar.pth`")

# Replay trained policy
st.subheader("🎮 Replay Trained Agent")

if st.button("▶️ Run Trained Model"):
    if not os.path.exists("policy_gradient_mountaincar.pth"):
        st.error("❌ Model file not found! Please train the model first using '🚀 Start Training'.")
    else:
        env = gym.make('MountainCar-v0', render_mode="rgb_array")
        policy = PolicyNetwork(2, 3)
        policy.load_state_dict(torch.load("policy_gradient_mountaincar.pth", map_location=torch.device('cpu')))
        policy.eval()

        frames = []
        state, _ = env.reset()
        for _ in range(300):
            frames.append(env.render())
            action, _ = select_action(policy, state)
            state, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
        env.close()

        st.image(frames[-1], caption="Final Frame of MountainCar", use_column_width=True)
        st.success("✅ Simulation complete!")
