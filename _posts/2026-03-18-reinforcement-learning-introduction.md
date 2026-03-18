---
layout: post
title: "强化学习入门：从基本概念到核心算法"
date: 2026-03-18
categories: AI
tags: [reinforcement-learning, machine-learning]
mathjax: true
excerpt: "强化学习是让智能体通过与环境的交互来学习最优策略的方法论。本文系统介绍 RL 的核心概念、数学基础和主要算法族。"
---

强化学习（Reinforcement Learning，RL）是机器学习的三大范式之一。不同于监督学习需要标注数据、无监督学习挖掘数据结构，RL 研究的是**智能体如何在与环境的持续交互中，通过试错来学习最优行为策略**。

AlphaGo 击败世界冠军、OpenAI Five 在 Dota 2 中超越人类职业选手、ChatGPT 背后的 RLHF 对齐技术——这些里程碑都建立在强化学习的理论基础之上。

---

## 一、核心概念

### 1.1 基本元素

强化学习由以下几个核心元素构成：

| 元素 | 符号 | 说明 |
|------|------|------|
| 智能体 | Agent | 做决策的主体 |
| 环境 | Environment | 智能体交互的外部世界 |
| 状态 | $s \in \mathcal{S}$ | 对当前情况的描述 |
| 动作 | $a \in \mathcal{A}$ | 智能体可执行的操作 |
| 奖励 | $r \in \mathbb{R}$ | 环境对动作的即时反馈 |
| 策略 | $\pi$ | 从状态到动作的映射 |

### 1.2 交互循环

智能体与环境的交互构成一个循环：

```
时刻 t：
  Agent 观察状态 s_t
  → 根据策略 π 选择动作 a_t
  → 环境转移到新状态 s_{t+1}
  → 环境返回奖励 r_t
  → 重复
```

这个循环产生一条**轨迹（Trajectory）**：

$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)$$

### 1.3 目标：最大化累积奖励

RL 的目标不是最大化单步奖励，而是最大化长期累积奖励。引入**折扣因子** $\gamma \in [0, 1]$，定义**回报（Return）**：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

$\gamma$ 的作用：
- $\gamma = 0$：只关注即时奖励，短视
- $\gamma \to 1$：高度重视未来，有远见
- 实践中通常取 $\gamma = 0.99$

---

## 二、马尔可夫决策过程（MDP）

RL 的数学框架是**马尔可夫决策过程**，形式化定义为五元组：

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

其中：
- $\mathcal{S}$：状态空间
- $\mathcal{A}$：动作空间
- $P(s' \mid s, a)$：状态转移概率
- $R(s, a)$：奖励函数
- $\gamma$：折扣因子

**马尔可夫性质**是核心假设——未来状态只依赖于当前状态，与历史无关：

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

---

## 三、价值函数

### 3.1 状态价值函数

在策略 $\pi$ 下，从状态 $s$ 出发能获得的期望回报：

$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t = s \right]$$

### 3.2 动作价值函数（Q 函数）

在状态 $s$ 执行动作 $a$，之后遵循策略 $\pi$ 的期望回报：

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]$$

两者的关系：

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s)\, Q^\pi(s, a)$$

### 3.3 Bellman 方程

价值函数满足**Bellman 期望方程**，这是 RL 中最重要的递推关系：

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a) + \gamma V^\pi(s') \right]$$

对 Q 函数：

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \sum_{a'} \pi(a' \mid s')\, Q^\pi(s', a')$$

**最优 Bellman 方程**（取最优策略）：

$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')$$

找到 $Q^*$ 即可推导出最优策略：$\pi^*(s) = \arg\max_a Q^*(s, a)$。

---

## 四、主要算法族

RL 算法按照不同维度可以分为多个流派：

### 4.1 基于值函数（Value-Based）

**代表：Q-Learning、DQN**

直接学习 Q 函数，策略隐含其中（取 argmax）。

**Q-Learning 更新规则**（off-policy，TD 方法）：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

其中 $\alpha$ 是学习率，方括号内是 **TD 误差（Temporal Difference Error）**：

$$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$$

**DQN**（Deep Q-Network）用神经网络 $Q_\theta(s, a)$ 近似 Q 函数，引入两个关键技巧：
- **经验回放（Experience Replay）**：将 $(s, a, r, s')$ 存入 replay buffer，打破相关性
- **目标网络（Target Network）**：用滞后更新的网络计算 TD target，稳定训练

### 4.2 策略梯度（Policy Gradient）

**代表：REINFORCE、PPO、TRPO**

直接参数化策略 $\pi_\theta(a \mid s)$，用梯度上升最大化期望回报。

**策略梯度定理**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot Q^{\pi_\theta}(s, a) \right]$$

直觉理解：如果某动作带来高回报，就增大其被选中的概率；反之则降低。

**基线（Baseline）技巧**——引入优势函数 $A^\pi(s, a)$ 减少方差：

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot A^\pi(s, a) \right]$$

**PPO（Proximal Policy Optimization）** 是目前最常用的策略梯度算法，通过截断重要性比率防止策略更新步子过大：

$$L^{\text{CLIP}}(\theta) = \mathbb{E} \left[ \min\left( r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

其中 $r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$。

### 4.3 Actor-Critic

**代表：A3C、SAC、TD3**

结合上面两种思路：Actor 负责输出策略，Critic 负责评估状态价值，相互促进。

```
Actor  π_θ(a|s)  →  执行动作
Critic V_φ(s)    →  评估好坏，指导 Actor 更新
```

**Soft Actor-Critic（SAC）** 在目标中加入策略熵，鼓励探索：

$$J(\pi) = \mathbb{E} \left[ \sum_t \gamma^t \left( r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t)) \right) \right]$$

其中 $\mathcal{H}(\pi(\cdot \mid s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$ 是策略的熵。

---

## 五、探索与利用的权衡

RL 面临的核心困境：**Exploration vs. Exploitation**。

- **利用（Exploitation）**：选择当前认为最好的动作
- **探索（Exploration）**：尝试未知动作，可能发现更好的选择

常用策略：

**$\epsilon$-greedy**：以概率 $\epsilon$ 随机探索，以 $1-\epsilon$ 贪心利用：

$$a_t = \begin{cases} \text{随机动作} & \text{以概率 } \epsilon \\ \arg\max_a Q(s_t, a) & \text{以概率 } 1-\epsilon \end{cases}$$

实践中 $\epsilon$ 会随训练进行逐渐衰减（annealing）。

---

## 六、算法选择指南

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| 离散动作、状态可枚举 | Q-Learning | 简单有效 |
| 离散动作、复杂状态（图像） | DQN | 神经网络拟合 Q |
| 连续动作空间 | SAC / TD3 | 专为连续动作设计 |
| 快速收敛、稳定性优先 | PPO | 工程上最广泛使用 |
| 样本效率优先 | SAC | off-policy，复用经验 |
| LLM 对齐 | PPO + RLHF | 业界标准 |

---

## 七、小结

强化学习的核心思想可以浓缩为一句话：**通过不断试错，在与环境的交互中学习最优行为**。

关键公式回顾：

$$\underbrace{Q^*(s, a)}_{\text{最优 Q 函数}} = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \underbrace{\max_{a'} Q^*(s', a')}_{\text{下一步最优值}}$$

RL 的学习路径建议：
1. 理解 MDP 和 Bellman 方程（本文）
2. 手写 Q-Learning 解简单网格世界
3. 用 PyTorch 实现 DQN 玩 CartPole
4. 学习 PPO，跑 MuJoCo 连续控制任务
5. 研究 RLHF，理解 LLM 对齐技术

---

*参考资料：*
- *Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)*
- *Spinning Up in Deep RL - OpenAI*
- *Proximal Policy Optimization Algorithms - Schulman et al., 2017*
