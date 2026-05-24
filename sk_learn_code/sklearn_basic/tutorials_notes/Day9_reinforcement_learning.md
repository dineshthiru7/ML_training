# Day 9 — Reinforcement Learning (RL)

## What is Reinforcement Learning?
Reinforcement Learning is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. The agent takes actions, receives feedback in the form of **rewards** or penalties, and learns a strategy (policy) to maximise cumulative reward over time.

There is **no labelled dataset**. The agent learns from experience — trial and error.

---

## Key Concepts

### Agent
The learner or decision-maker. It observes the environment, chooses actions, and updates its knowledge based on rewards received.

### Environment
Everything the agent interacts with. The environment transitions from one state to another based on the agent's action and provides a reward signal.

### State (s)
A description of the current situation. It is what the agent observes before taking an action.
- Example: the position of a robot on a grid, the board layout in chess.

### Action (a)
A move the agent can make in a given state.
- Example: move left/right/up/down, play a chess piece.

### Reward (r)
A scalar feedback signal from the environment after taking an action. Positive reward = good outcome, negative reward (penalty) = bad outcome.
- Example: +1 for reaching the goal, -1 for falling into a hole, 0 otherwise.

### Policy (π)
The strategy the agent follows — a mapping from states to actions.
`π(s) → a`
- **Deterministic policy**: always take the same action in a given state.
- **Stochastic policy**: take actions with certain probabilities.

### Value Function V(s)
Expected cumulative future reward starting from state s following policy π.
`V(s) = E[ total discounted reward | starting at state s ]`

### Q-Value / Action-Value Function Q(s, a)
Expected cumulative future reward when taking action a in state s, then following policy π.
`Q(s, a) = expected reward for taking action a in state s`

### Episode
A complete sequence of interactions from the initial state to a terminal state (e.g., game won/lost).

### Discount Factor (γ, gamma)
A value between 0 and 1 that weights future rewards.
- γ = 0 → agent only cares about immediate reward.
- γ = 1 → agent values future rewards equally.
- γ = 0.9 → near-future rewards matter more than distant ones.

`Total Return G = r₁ + γ×r₂ + γ²×r₃ + γ³×r₄ + ...`

---

## Supervised vs Unsupervised vs Reinforcement Learning

| | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| Data | Labelled (X, y) | Unlabelled (X) | No dataset — learn from interaction |
| Feedback | Correct answer provided | None | Reward signal from environment |
| Goal | Predict output | Find structure | Maximise cumulative reward |
| Examples | Classification, Regression | Clustering, PCA | Games, Robotics, Trading |

---

## Q-Learning

### What is it?
Q-Learning is a popular model-free RL algorithm. The agent learns the **Q-value** for every (state, action) pair by interacting with the environment. It stores Q-values in a table called the **Q-table**.

### Bellman Equation (Q-Learning Update)
The Q-table is updated after every action using the Bellman equation:

`Q(s, a) ← Q(s, a) + α × [ r + γ × max Q(s', a') - Q(s, a) ]`

Where:
- `Q(s, a)` = current Q-value for state s, action a
- `α` (alpha) = learning rate (how much to update each step)
- `r` = reward received after taking action a in state s
- `γ` (gamma) = discount factor
- `s'` = next state after taking action a
- `max Q(s', a')` = best Q-value available in the next state (greedy)

**Plain reading:** new Q = old Q + learning_rate × (target - old Q)
where target = `r + γ × best future Q`

### Q-Table
A matrix of size (number of states × number of actions) storing Q-values.

| | Left | Right | Up | Down |
|---|---|---|---|---|
| State 0 | 0.0 | 0.5 | 0.0 | 0.2 |
| State 1 | 0.1 | 0.0 | 0.8 | 0.0 |
| ... | ... | ... | ... | ... |

The agent picks the action with the highest Q-value for its current state.

### Q-Learning Algorithm (Step by Step)
1. Initialise Q-table with zeros (or small random values).
2. Observe current state s.
3. Choose action a (exploration or exploitation — see below).
4. Take action a, observe reward r and next state s'.
5. Update Q-table using Bellman equation.
6. Set s = s'. Go to Step 3.
7. Repeat until convergence or max episodes reached.

### Simple Q-Learning Example (FrozenLake)
```python
import numpy as np
import gym

# Create environment
env = gym.make('FrozenLake-v1', is_slippery=False)
n_states  = env.observation_space.n   # 16 states (4x4 grid)
n_actions = env.action_space.n        # 4 actions: left, down, right, up

# Initialise Q-table
Q = np.zeros((n_states, n_actions))

# Hyperparameters
alpha   = 0.8   # learning rate
gamma   = 0.95  # discount factor
epsilon = 1.0   # exploration rate
epsilon_decay = 0.995
epsilon_min   = 0.01
episodes = 1000

# Training loop
for ep in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()   # explore
        else:
            action = np.argmax(Q[state])          # exploit

        next_state, reward, done, truncated, _ = env.step(action)

        # Bellman update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        state = next_state

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Training complete")
print("Q-table (first 5 states):")
print(Q[:5])

# Evaluate
wins = 0
for _ in range(100):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, reward, done, truncated, _ = env.step(action)
    if reward == 1.0:
        wins += 1
print(f"Win rate: {wins}%")
```

---

## Exploration vs Exploitation

One of the core challenges in RL is the **exploration-exploitation trade-off**.

| | Exploration | Exploitation |
|---|---|---|
| Meaning | Try new/random actions to discover better ones | Use what is already known (take best known action) |
| Risk | Wastes time on poor actions | Gets stuck in local optima |

**Epsilon-Greedy Strategy** — the most common approach:
- With probability ε → take a random action (explore).
- With probability (1-ε) → take the best known action (exploit).
- ε starts high (e.g. 1.0) and decays over training so the agent explores early and exploits later.

```python
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min   = 0.01

# Per episode:
if np.random.rand() < epsilon:
    action = env.action_space.sample()   # random exploration
else:
    action = np.argmax(Q[state])          # best known action

epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

---

## Deep Q-Network (DQN) — Brief Overview

When the state space is too large for a Q-table (e.g. image pixels in Atari games), a **neural network** approximates the Q-function.

**Q-table** → replaced by a **neural network** that takes state as input and outputs Q-values for all actions.

| Q-Table | DQN |
|---|---|
| Works for small discrete states | Works for large / continuous states |
| Exact lookup | Approximate using neural network |
| No GPU needed | Needs more compute |
| e.g. FrozenLake (16 states) | e.g. Atari games (pixel input) |

**Key additions in DQN:**
- **Experience Replay** — store (s, a, r, s') transitions in a replay buffer; sample random mini-batches for training (breaks correlations).
- **Target Network** — a separate frozen network for computing target Q-values, updated less frequently (training stability).

---

## Common RL Environments (OpenAI Gym)

```python
import gym

# Install: pip install gymnasium
env = gym.make('CartPole-v1')     # balance a pole on a cart
env = gym.make('FrozenLake-v1')   # navigate a frozen grid to goal
env = gym.make('MountainCar-v0')  # push car up a hill
env = gym.make('LunarLander-v2')  # land a spacecraft

# Environment info
print(env.observation_space)  # state space
print(env.action_space)        # action space

# Run one episode manually
state, _ = env.reset()
done = False
while not done:
    action = env.action_space.sample()   # random agent
    state, reward, done, truncated, info = env.step(action)
env.close()
```

---

## RL Algorithms Overview

| Algorithm | Type | Description |
|---|---|---|
| Q-Learning | Model-free, off-policy | Learns Q-table; small discrete environments |
| SARSA | Model-free, on-policy | Updates Q using actual next action taken |
| DQN | Deep, off-policy | Neural network approximates Q-table |
| Policy Gradient (REINFORCE) | Policy-based | Directly optimises policy |
| PPO (Proximal Policy Optimisation) | Actor-Critic | Stable, widely used in practice |
| A3C | Actor-Critic | Parallel agents, faster training |

**For beginners:** Start with Q-Learning on FrozenLake → DQN on CartPole → PPO with Stable-Baselines3.

---

## Stable-Baselines3 (High-Level RL Library)

Stable-Baselines3 provides ready-to-use RL algorithms (PPO, A2C, DQN, SAC, TD3) without writing training loops manually.

```python
# pip install stable-baselines3 gymnasium

from stable_baselines3 import PPO
import gym

env = gym.make("CartPole-v1")

# Create and train model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# Evaluate
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        obs, _ = env.reset()

env.close()
```

---

## Key Hyperparameters in RL

| Parameter | Symbol | Meaning | Typical Value |
|---|---|---|---|
| Learning rate | α | How fast to update Q-values or weights | 0.001–0.1 |
| Discount factor | γ | How much to value future rewards | 0.95–0.99 |
| Exploration rate | ε | Probability of taking random action | 1.0 → 0.01 |
| Epsilon decay | — | Rate at which ε reduces per episode | 0.99–0.999 |
| Replay buffer size | — | Number of past transitions to store (DQN) | 10,000–1,000,000 |
| Batch size | — | Samples per training step (DQN) | 32–256 |

---

## Summary
- **RL** = agent learns by taking actions in an environment and receiving rewards — no labelled data.
- **Q-Learning** — tabular method; stores Q(s,a) values; updates via Bellman equation.
- **Bellman update**: `Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') - Q(s,a)]`
- **Epsilon-greedy** — balances exploration vs exploitation; ε decays over training.
- **DQN** — uses neural network when state space is too large for a table.
- **Stable-Baselines3** — high-level library for practical RL without boilerplate.
