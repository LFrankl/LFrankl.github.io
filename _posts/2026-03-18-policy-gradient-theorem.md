---
layout: post
title: "策略梯度定理：完整推导与理论详解"
date: 2026-03-18
categories: AI
tags: [reinforcement-learning, policy-gradient, math]
mathjax: true
excerpt: "策略梯度定理是现代深度强化学习的理论基石。本文从第一性原理出发，完整推导该定理，并讨论基线、优势函数、重要性采样等核心技巧。"
---

策略梯度方法直接对策略参数求梯度，是 PPO、TRPO、SAC 等现代算法的共同理论基础。**策略梯度定理**给出了目标函数对策略参数梯度的精确表达式，它的推导并不平凡——本文从零开始，一步步把它推出来。

---

## 一、问题设置

设策略 $\pi_\theta(a \mid s)$ 由参数 $\theta$ 参数化（例如一个神经网络）。

定义**目标函数**为从初始状态分布 $d_0(s)$ 出发的期望折扣回报：

$$J(\theta) = \mathbb{E}_{s_0 \sim d_0} \left[ V^{\pi_\theta}(s_0) \right]$$

其中状态价值函数：

$$V^{\pi_\theta}(s) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s \right]$$

我们的目标是计算 $\nabla_\theta J(\theta)$，进而用梯度上升更新参数：

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

---

## 二、推导前的准备

### 2.1 一个关键恒等式

对任意可微函数 $f(\theta)$（假设 $f > 0$），有：

$$\nabla_\theta \log f(\theta) = \frac{\nabla_\theta f(\theta)}{f(\theta)}$$

因此：

$$\nabla_\theta f(\theta) = f(\theta) \cdot \nabla_\theta \log f(\theta)$$

这个变换叫 **log-derivative trick**（对数导数技巧），是策略梯度推导的核心工具。

### 2.2 轨迹概率

一条轨迹 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ 在策略 $\pi_\theta$ 下的概率：

$$p_\theta(\tau) = d_0(s_0) \prod_{t=0}^{T} \pi_\theta(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)$$

注意其中 $P(s_{t+1} \mid s_t, a_t)$ 是环境转移概率，与 $\theta$ 无关。

对 $p_\theta(\tau)$ 取对数再求导，环境部分消掉：

$$\nabla_\theta \log p_\theta(\tau) = \nabla_\theta \log \prod_{t=0}^{T} \pi_\theta(a_t \mid s_t) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

---

## 三、策略梯度定理推导

### 3.1 轨迹视角

将目标函数写成对轨迹的期望：

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ R(\tau) \right] = \int p_\theta(\tau) R(\tau) \, d\tau$$

其中 $R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$ 是轨迹总回报。

对 $\theta$ 求梯度：

$$\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau) \cdot R(\tau) \, d\tau$$

注意 $R(\tau)$ 不含 $\theta$（奖励由环境决定），可以直接提出梯度符号。

用 log-derivative trick 替换 $\nabla_\theta p_\theta(\tau)$：

$$\nabla_\theta J(\theta) = \int p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau) \cdot R(\tau) \, d\tau$$

写回期望形式，并代入 $\nabla_\theta \log p_\theta(\tau)$ 的表达式：

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right) R(\tau) \right]}$$

这是**策略梯度定理的轨迹形式**。

### 3.2 因果性改进（Reward-to-Go）

上面的形式存在冗余：$t$ 时刻之前的奖励与 $t$ 时刻的动作无关（**因果性**）。

更准确地，对于时刻 $t$ 的动作 $a_t$，只有 $t$ 时刻之后的奖励才与之相关。定义**未来回报（Reward-to-Go）**：

$$\hat{R}_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$$

可以证明，将 $R(\tau)$ 替换为 $\hat{R}_t$ 不改变梯度的期望值（过去的奖励对梯度的贡献期望为零），但**显著降低方差**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \hat{R}_t \right]$$

### 3.3 状态分布视角（经典形式）

将期望展开到每个时间步的 $(s, a)$ 对上。定义 $\gamma$-折扣**状态访问频率**：

$$d^{\pi_\theta}(s) = \sum_{t=0}^{\infty} \gamma^t \Pr(s_t = s \mid \pi_\theta)$$

则：

$$\nabla_\theta J(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \nabla_\theta \pi_\theta(a \mid s) \cdot Q^{\pi_\theta}(s, a)$$

再次用 log-derivative trick：$\nabla_\theta \pi_\theta(a \mid s) = \pi_\theta(a \mid s) \cdot \nabla_\theta \log \pi_\theta(a \mid s)$，得到最终形式：

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta},\, a \sim \pi_\theta(\cdot \mid s)} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot Q^{\pi_\theta}(s, a) \right]}$$

这就是教科书里的**策略梯度定理（Policy Gradient Theorem）**。

---

## 四、基线与优势函数

### 4.1 为什么需要基线

直接用 $Q^{\pi}(s, a)$ 作为梯度的权重，方差往往极大。考虑这样一个场景：所有动作的回报都是正数（比如 $Q$ 值为 100、200、300），梯度更新会同时增大所有动作的概率，收敛极慢。

**基线（Baseline）**：在 $Q^{\pi}(s, a)$ 上减去一个只依赖于状态 $s$ 的函数 $b(s)$，不改变梯度期望，但降低方差。

**证明**：基线不改变梯度期望，即：

$$\mathbb{E}_{a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot b(s) \right] = 0$$

推导：

$$\mathbb{E}_{a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot b(s) \right] = b(s) \sum_a \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s)$$

$$= b(s) \sum_a \nabla_\theta \pi_\theta(a \mid s) = b(s) \cdot \nabla_\theta \underbrace{\sum_a \pi_\theta(a \mid s)}_{=1} = 0 \quad \square$$

### 4.2 最优基线

从方差最小化角度，最优基线是：

$$b^*(s) = \frac{\mathbb{E}_{a \sim \pi_\theta}\left[ \left\| \nabla_\theta \log \pi_\theta(a \mid s) \right\|^2 Q^{\pi}(s, a) \right]}{\mathbb{E}_{a \sim \pi_\theta}\left[ \left\| \nabla_\theta \log \pi_\theta(a \mid s) \right\|^2 \right]}$$

这个基线计算代价较高，实践中通常用 $V^{\pi}(s)$ 作为基线，近似效果已经很好。

### 4.3 优势函数

取 $b(s) = V^{\pi_\theta}(s)$，定义**优势函数**：

$$A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$$

直觉：$A^{\pi}(s, a)$ 衡量动作 $a$ 相对于该状态下平均水平的好坏程度。

- $A > 0$：动作 $a$ 比平均更好，应增大其概率
- $A < 0$：动作 $a$ 比平均更差，应减小其概率

策略梯度变为：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot A^{\pi_\theta}(s, a) \right]$$

### 4.4 优势函数的实际估计

实践中 $Q^{\pi}$ 和 $V^{\pi}$ 都是未知的，需要估计。常用 **GAE（Generalized Advantage Estimation）**：

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 TD 残差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，$\lambda \in [0, 1]$ 控制偏差-方差权衡：

- $\lambda = 0$：$\hat{A}_t = \delta_t$，一步 TD，低方差高偏差
- $\lambda = 1$：$\hat{A}_t = \sum_{l} \gamma^l \delta_{t+l} = \hat{R}_t - V(s_t)$，Monte Carlo，高方差低偏差

---

## 五、REINFORCE 算法

基于以上推导，最简单的策略梯度算法 **REINFORCE**（Williams, 1992）如下：

**算法流程**：

```
初始化策略参数 θ
for 每个 episode:
    用 π_θ 采集轨迹 τ = (s_0, a_0, r_0, ..., s_T, a_T, r_T)
    for t = 0, 1, ..., T:
        计算 R_t = Σ γ^(t'-t) r_{t'}  （reward-to-go）
    梯度估计：g = Σ_t ∇_θ log π_θ(a_t|s_t) · R_t
    参数更新：θ ← θ + α · g
```

**梯度估计的无偏性**：REINFORCE 的梯度估计量是 $\nabla_\theta J(\theta)$ 的无偏估计，但方差极大，收敛很慢。

---

## 六、重要性采样与离策略梯度

### 6.1 On-policy 的效率问题

上面的推导都基于 **on-policy**（用当前策略 $\pi_\theta$ 采样），每次更新后样本即废弃，样本效率低。

**重要性采样**允许复用旧策略 $\pi_{\theta_{\text{old}}}$ 采集的数据：

$$\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} f(a) \right]$$

因此离策略目标：

$$J(\theta) \approx \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \sum_t \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} A^{\pi_{\theta_{\text{old}}}}(s_t, a_t) \right]$$

### 6.2 PPO 的 CLIP 目标

直接最大化上面的目标可能导致策略变化过大（重要性比率过大/过小时估计不准）。PPO 用截断解决这个问题：

令 $r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$，PPO 的目标：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

截断的直觉：如果 $\hat{A}_t > 0$（好动作），我们希望增大 $r_t$，但不超过 $1+\epsilon$；如果 $\hat{A}_t < 0$（坏动作），希望减小 $r_t$，但不低于 $1-\epsilon$。这形成了一个悲观的下界，防止过度更新。

---

## 七、PyTorch 实现要点

理论清楚后，代码非常简洁。核心就是把 $\nabla_\theta \log \pi_\theta(a \mid s) \cdot \hat{R}_t$ 变成一个可以 `backward()` 的 loss。

注意这里的技巧：梯度上升目标 $\nabla_\theta J$ 对应的 loss 是**负号**（PyTorch 默认梯度下降）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),     nn.Tanh(),
            nn.Linear(64, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)  # 返回 logits

    def get_action(self, obs):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)  # log π_θ(a|s)
        return action.item(), log_prob


def compute_returns(rewards, gamma=0.99):
    """计算 reward-to-go：G_t = Σ γ^(t'-t) r_{t'}"""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # 标准化，进一步降低方差
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def reinforce_loss(log_probs, returns):
    """
    策略梯度 loss：
    ∇J = E[∇ log π(a|s) · G_t]
    对应的 loss = -E[log π(a|s) · G_t]  （负号因为 PyTorch 梯度下降）
    """
    log_probs = torch.stack(log_probs)
    loss = -(log_probs * returns).mean()
    return loss
```

**训练循环的关键部分**：

```python
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# 采集一条轨迹
log_probs, rewards = [], []
obs = env.reset()
done = False
while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    action, log_prob = policy.get_action(obs_tensor)
    obs, reward, done, _ = env.step(action)
    log_probs.append(log_prob)
    rewards.append(reward)

# 计算梯度并更新
returns = compute_returns(rewards)
loss = reinforce_loss(log_probs, returns)

optimizer.zero_grad()
loss.backward()   # 自动计算 ∇_θ log π_θ(a|s)
optimizer.step()
```

`log_prob = dist.log_prob(action)` 这一行是关键。PyTorch 的 `Categorical` 分布会自动构建计算图，使得 `loss.backward()` 能正确计算 $\nabla_\theta \log \pi_\theta(a \mid s)$。

---

## 八、理论总结

从第一性原理出发，策略梯度的完整推导链条是：

$$J(\theta) = \mathbb{E}_\tau[R(\tau)]$$

$$\xrightarrow{\text{求梯度}} \nabla_\theta J = \int \nabla_\theta p_\theta(\tau) R(\tau) d\tau$$

$$\xrightarrow{\text{log-derivative}} = \mathbb{E}_\tau\left[\nabla_\theta \log p_\theta(\tau) \cdot R(\tau)\right]$$

$$\xrightarrow{\text{展开 log } p_\theta} = \mathbb{E}_\tau\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$$

$$\xrightarrow{\text{因果性}} = \mathbb{E}_\tau\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{R}_t\right]$$

$$\xrightarrow{\text{减基线}} = \mathbb{E}_\tau\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^\pi(s_t, a_t)\right]$$

每一步变换都保持梯度期望不变，只是逐步降低估计方差。理解这条推导链，PPO、A3C、SAC 等算法的理论基础就都清晰了。

---

*参考：*
- *Sutton et al., Policy Gradient Methods for Reinforcement Learning with Function Approximation, 2000*
- *Williams, Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning, 1992*
- *Schulman et al., High-Dimensional Continuous Control Using Generalized Advantage Estimation, 2016*
- *Schulman et al., Proximal Policy Optimization Algorithms, 2017*
