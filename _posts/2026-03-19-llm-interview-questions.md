---
layout: post
title: "中国互联网大厂 LLM 算法岗面试题大全"
date: 2026-03-19
categories: 面试
tags: [LLM, interview, transformer, RLHF, DPO, 面试]
mathjax: true
mermaid: true
excerpt: "覆盖字节、阿里、腾讯、百度、华为、DeepSeek 等大厂 LLM 算法岗高频面试题，包含 Transformer、预训练、微调、对齐、推理、部署、多模态等方向，每题附详细解答。"
---

收录字节跳动、阿里巴巴、腾讯、百度、华为、DeepSeek、智谱、商汤、minimax 等大厂 LLM 算法岗真实高频面试题，按模块分类，每题附详细解答。

---

## 模块一：Transformer 基础

---

**Q1：Scaled Dot-Product Attention 为什么要除以 $\sqrt{d_k}$？**

设 $Q_i, K_j$ 的各分量独立同分布，均值为 0，方差为 1，则点积的方差为：

$$\text{Var}(Q_i \cdot K_j) = \sum_{k=1}^{d_k} \text{Var}(Q_{ik} K_{jk}) = d_k$$

当 $d_k$ 很大时（如 64、128），点积的绝对值很大，送入 softmax 后梯度接近零（饱和区）。除以 $\sqrt{d_k}$ 将方差归一化为 1，softmax 工作在梯度充分的线性区，训练更稳定。

---

**Q2：Multi-Head Attention 的作用是什么？多个 head 学到了什么？**

单个 Attention head 只能学习一种"关注模式"。Multi-Head Attention 用 $h$ 个独立的 head 并行捕捉不同类型的依赖关系：

- 部分 head 捕捉语法依赖（主谓关系）
- 部分 head 捕捉指代关系（代词 → 名词）
- 部分 head 捕捉位置关系（相邻词）
- 部分 head 捕捉语义关联（近义词）

实验上（Voita et al., 2019），可以剪掉大部分 head 而几乎不影响性能，说明很多 head 存在冗余，少数 head 承担主要功能。

参数量角度：$h$ 个 head，每个 head 维度 $d_k = d_{\text{model}}/h$，总参数量与单 head 相当，但表达能力更强。

---

**Q3：Self-Attention 的时间和空间复杂度是多少？瓶颈在哪？**

| 操作 | 时间复杂度 | 空间复杂度 |
|------|----------|----------|
| $QK^\top$ | $O(T^2 d)$ | $O(T^2)$ |
| softmax | $O(T^2)$ | $O(T^2)$ |
| $\text{Attn} \cdot V$ | $O(T^2 d)$ | $O(Td)$ |

瓶颈是**序列长度 $T$ 的平方项**：注意力矩阵 $T \times T$ 在 $T=8192$ 时大约 256M 个元素，fp16 下约 512MB（单层单头）。Multi-Head 乘以 $h$，多层乘以 $L$，内存压力极大。

Flash Attention 通过分块计算将空间复杂度从 $O(T^2)$ 降为 $O(T)$，时间复杂度不变但 IO 减少，实际速度 2-4x。

---

**Q4：Post-LN 和 Pre-LN 的区别？现代 LLM 为什么用 Pre-LN？**

**Post-LN**（原始 Transformer）：$x' = \text{LN}(x + \text{SubLayer}(x))$

**Pre-LN**（GPT-2 之后）：$x' = x + \text{SubLayer}(\text{LN}(x))$

Pre-LN 的优势：
1. 梯度流更稳定。Post-LN 中，梯度必须经过 LN 层，LN 的梯度缩放不可控；Pre-LN 中残差路径是梯度高速公路，LN 在旁路。
2. 训练更稳定，不需要精心设计的 warmup schedule，可以使用更大的学习率。
3. 支持更深的网络（100+ 层）。

Post-LN 的优势：最终精度有时略高（LN 在每层输出后，抑制了特征的偏移），但训练稳定性差。

现代 LLM（LLaMA、Mistral、Qwen 等）全部使用 Pre-LN，通常用 RMSNorm 代替 LayerNorm。

---

**Q5：RMSNorm 和 LayerNorm 的区别？**

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta, \quad \mu = \frac{1}{d}\sum x_i,\quad \sigma = \sqrt{\frac{1}{d}\sum(x_i-\mu)^2}$$

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2}$$

RMSNorm 去掉了中心化步骤（不减均值），没有偏置参数 $\beta$。

**优势**：
- 计算量更小（少一次均值计算和均值修正）
- 实验表明效果与 LayerNorm 相当甚至更好
- 参数更少

**LLaMA 系列、Mistral、Qwen、DeepSeek 全部使用 RMSNorm**。

---

**Q6：RoPE 的原理是什么？相比绝对位置编码有什么优势？**

RoPE（Rotary Position Embedding）的核心思想：通过旋转 Q 和 K 的向量来注入相对位置信息，使得 $Q_m \cdot K_n$ 只依赖相对位置 $m-n$。

对位置 $m$ 的向量，每对相邻维度 $(x_{2i}, x_{2i+1})$ 做旋转：

$$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

其中 $\theta_i = 10000^{-2i/d}$。

**关键性质**：

$$\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m-n)$$

点积只依赖相对位置差 $m-n$，与绝对位置无关。

**优势**：
1. 编码相对位置，外推性更好
2. 通过修改 $\theta$（NTK-aware Scaling、YaRN）可以扩展上下文长度
3. 无需在 embedding 上额外加位置向量，与 Attention 计算融合

---

**Q7：KV Cache 的原理？GQA 怎么减少 KV Cache？**

**KV Cache 原理**：自回归生成时，第 $t$ 步计算需要所有历史 token 的 K 和 V。如果每步都重新计算，复杂度是 $O(T^2)$。KV Cache 将历史 token 的 K、V 存下来，每步只计算新 token 的 K、V，复杂度降为 $O(T)$。

**KV Cache 显存**（每层）：$2 \times T \times n_{\text{kv\_heads}} \times d_k \times \text{bytes}$

以 LLaMA-3-70B（$L=80, n_{\text{heads}}=64, d_k=128$, fp16）为例：
- MHA：每 token 约 $80 \times 2 \times 64 \times 128 \times 2 = 26.2\text{MB}$，序列长 8192 约 200GB，完全放不下
- GQA（$n_{\text{kv\_heads}}=8$）：缩小 8 倍，约 25GB

**GQA 原理**：$h$ 个 Q head 分为 $g$ 组，每组共享一对 K、V head，KV Cache 大小从正比于 $h$ 降为正比于 $g$。

---

**Q8：Flash Attention 解决了什么问题？核心思路是什么？**

**问题**：标准 Attention 需要将 $T \times T$ 的注意力矩阵写到 HBM（显卡显存），HBM 带宽远低于 SRAM（片上缓存），大量时间花在 IO 而非计算上。

**核心思路**：IO 感知算法（IO-Aware Algorithm）。将 Q、K、V 分块，每个块在 SRAM 中完成 softmax + 加权求和，只将最终结果写回 HBM，完全避免中间的 $T \times T$ 矩阵。

关键技术：**online softmax**（log-sum-exp 分块累计），使得不需要看到全部 attention scores 就能正确计算 softmax。

**效果**：
- 内存：$O(T^2) \to O(T)$
- HBM 读写量：$O(T^2 d) \to O(Td)$
- 实际速度：2-4x 提升
- 数学上完全等价（exact attention，非近似）

Flash Attention 2 进一步优化了并行度，Flash Attention 3 针对 H100 的 tensor core 做了特化。

---

**Q9：为什么 FFN 的中间维度是 $4d_{\text{model}}$？SwiGLU 为什么改成 $\frac{8}{3}d$？**

**$4d$ 的由来**：原始论文的工程经验，在 $d_{\text{model}}=512$ 时 $d_{\text{ff}}=2048$ 效果好，没有严格理论依据。后来的研究表明 FFN 是"知识存储器"，更大的 FFN 存储更多知识，4x 是计算与性能的经验平衡点。

**SwiGLU 改为 $\frac{8}{3}d$ 的原因**：SwiGLU 有 3 个矩阵（$W_1, W_2, W_3$），若保持 $d_{\text{ff}}=4d$，参数量比原始 FFN 多 50%。为了保持与 $\text{ReLU}+4d$ 相近的参数量，调整为：

$$3 \times d \times d_{\text{ff}} = 2 \times d \times 4d \implies d_{\text{ff}} = \frac{8}{3}d \approx 2.67d$$

实践中取 256 的整数倍对齐，如 $d_{\text{model}}=4096$ 时 $d_{\text{ff}}=11008$（$\approx 2.69 \times 4096$）。

---

**Q10：Encoder-only、Decoder-only、Encoder-Decoder 三种架构各有什么适用场景？**

| 架构 | 代表模型 | Attention | 适用任务 |
|------|---------|-----------|---------|
| Encoder-only | BERT、RoBERTa | 双向（全局可见） | 分类、NER、句子匹配、信息抽取 |
| Decoder-only | GPT 系列、LLaMA | 因果（只看历史） | 文本生成、对话、推理 |
| Encoder-Decoder | T5、BART | Encoder 双向 + Decoder 因果 | 翻译、摘要、Seq2Seq |

**为什么现代大模型几乎全用 Decoder-only？**

1. **训练目标统一**：Next Token Prediction 覆盖所有 NLP 任务，不需要为不同任务设计不同目标
2. **In-context Learning**：Few-shot 示例可以直接放在 context 里，无缝融入生成过程
3. **扩展规律更好**：Chinchilla 等研究表明 Decoder-only 架构的 Scaling Law 更稳定
4. **推理简单**：只需一个模型，不需要 Encoder-Decoder 间的 Cross-Attention

---

## 模块二：预训练

---

**Q11：预训练数据的处理流程是什么？质量过滤怎么做？**

工业界的预训练数据处理流水线：

```
原始网页数据（Common Crawl）
  ↓ URL 过滤（黑名单域名、成人内容）
  ↓ 语言识别（fastText 等）
  ↓ HTML 解析（去标签、提取正文）
  ↓ 启发式过滤（最小长度、标点符号比例、重复词比例）
  ↓ 模型打分过滤（用小分类器判断内容质量）
  ↓ 去重（MinHash 近似去重、精确 URL 去重）
  ↓ 安全过滤（有害内容）
  ↓ 最终数据集
```

**关键过滤策略**：
- **去重**：训练集中重复数据会导致模型死记硬背，MinHash LSH 是工业界标准方案
- **质量打分**：用高质量数据（如维基百科、书籍）训练一个二分类器，对网页打分
- **困惑度过滤**：用小语言模型计算困惑度，过滤掉低质量或乱码文本
- **数据配比**：代码、数学、书籍、网页的比例对下游能力影响巨大（Llama-3 的技术报告中有详细描述）

---

**Q12：Scaling Law 的核心结论是什么？Chinchilla 修正了什么？**

**Kaplan et al.（2020）的 Scaling Law**：

$$L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}$$

模型 loss 与参数量 $N$ 和数据量 $D$ 的幂律关系，幂指数约 $\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$。

**Kaplan 的结论**：给定计算预算，优先扩大参数量（$N$ 的指数更大），数据量跟着少量增加即可。这导致了 GPT-3（175B 参数，但数据相对少）的设计。

**Hoffmann et al.（Chinchilla，2022）的修正**：

重新实验后发现 Kaplan 的结论是错的——$\alpha_N \approx \alpha_D$，即**参数量和数据量应该等比例扩展**。

最优配比：$D \approx 20N$（即每个参数对应 20 个训练 token）。

Chinchilla-70B 用与 Gopher-280B 相同的计算预算，但参数量更小、数据量更大，效果显著更好。

**影响**：Llama 系列训练数据大幅超过 Chinchilla 最优（Llama-3 用 15T token 训练 8B/70B 模型，远超最优），换来更强的推理能力（over-trained 对推理更有利）。

---

**Q13：为什么预训练要用 BFloat16 而不是 Float16？**

| 格式 | 指数位 | 尾数位 | 数值范围 | 精度 |
|------|--------|--------|---------|------|
| FP32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | 高 |
| FP16 | 5 | 10 | $\pm 6.5 \times 10^4$ | 低 |
| BF16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | 低 |

**BF16 的优势**：与 FP32 有相同的指数位（8位），数值范围相同，不容易溢出（overflow）或下溢（underflow）。

**FP16 的问题**：指数位只有 5 位，最大值约 65504，训练中梯度很容易溢出，必须配合 loss scaling（将 loss 乘以大数再反向传播，计算梯度后再除回来），工程复杂。

**实践**：
- 预训练：BF16（A100/H100 对 BF16 有硬件加速）
- 推理量化：INT8、INT4（进一步压缩显存）

---

**Q14：预训练中的学习率调度策略是什么？为什么用 Cosine Decay？**

现代大模型的学习率调度：

```
Linear Warmup（前 1-2% 步）→ Cosine Decay（主体）→ 常数（可选）
```

**Warmup 的作用**：训练初期参数随机，梯度方差极大，大学习率会导致参数震荡。Warmup 从 0 线性增大到目标学习率，等参数稳定后再开始正常训练。

**Cosine Decay 的优势**：
- 初期下降慢，让模型充分探索
- 后期下降快，收敛到更好的极小值
- 数学形式：$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$

**Llama-3 的实践**：学习率 $3 \times 10^{-4}$，Warmup 2000 步，Cosine Decay 到 $3 \times 10^{-5}$（最大学习率的 10%）。

---

**Q15：梯度裁剪（Gradient Clipping）的作用？阈值怎么设？**

$$\text{if } \|\nabla\| > \text{threshold}: \quad \nabla \leftarrow \frac{\text{threshold}}{\|\nabla\|} \nabla$$

**作用**：防止梯度爆炸——大模型训练中偶发的大梯度会让参数剧烈震荡，进而引发 loss spike（损失突然跳高）甚至 NaN。

**阈值选择**：通常取 1.0，实践中监控训练过程中的梯度范数分布，超过阈值的比例在 5% 以下为健康状态。

**Loss Spike**：大模型训练中时常出现 loss 突然增大后缓慢恢复的现象，通常因为某批高困惑度数据引发梯度爆炸。工程上的处理：从 spike 前的 checkpoint 恢复，跳过这批数据继续训练。

---

## 模块三：微调（SFT / PEFT）

---

**Q16：SFT 和预训练的区别？SFT 为什么只在 response 上计算 loss？**

**区别**：
- 预训练：无监督，在海量文本上做 Next Token Prediction，学习语言模式和知识
- SFT：有监督，在 (instruction, response) 对上微调，学习遵循指令的格式

**只在 response 上计算 loss 的原因**：

若 instruction 部分也算 loss，模型会把优化目标分散到 prompt 的语言分布上，而 prompt 是固定的输入，我们希望模型学的是"给定 prompt 后如何生成好的 response"。在 prompt 上算 loss 等于让模型学"如何写提问"，对我们的目标没有帮助，甚至会干扰学习。

实现上：把 prompt 对应的 label 设为 -100，`CrossEntropyLoss` 会自动跳过这些位置（`ignore_index=-100`）。

---

**Q17：LoRA 的原理？为什么低秩分解是合理的假设？**

**LoRA（Low-Rank Adaptation）**：冻结预训练模型的全部参数，只训练低秩的增量矩阵：

$$W' = W_0 + \Delta W = W_0 + BA$$

其中 $W_0 \in \mathbb{R}^{d \times k}$（frozen），$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d, k)$。

前向传播：$h = W_0 x + BAx = W_0 x + \frac{\alpha}{r} BAx$，$\alpha$ 是缩放因子。

**低秩假设的合理性**：

Aghajanyan et al.（2020）的 intrinsic dimensionality 研究表明，预训练模型的微调梯度本质上是低秩的——微调只需要在一个很小的子空间内调整，这个子空间的维度远小于参数空间维度。

直觉上：语言模型已经学到了丰富的知识，微调只是"调整方向"，不需要改变整个参数矩阵，只需在关键子空间内做调整。

**参数量对比**（$d=4096, k=4096, r=8$）：
- 全量微调：$4096^2 = 16.7M$ 参数
- LoRA：$2 \times 4096 \times 8 = 65K$ 参数，缩小约 256 倍

---

**Q18：QLoRA 是怎么做的？NF4 量化的原理？**

**QLoRA** 在 LoRA 的基础上用 4-bit 量化存储基础模型：

1. **NF4（Normal Float 4）量化**：将权重量化为 4-bit 整数，但使用信息论最优的量化格点

   假设预训练权重近似服从标准正态分布 $\mathcal{N}(0,1)$，NF4 的 16 个量化级别是标准正态分布的等面积分位点，使得量化误差的期望最小化。

2. **Double Quantization**：对量化常数本身再做 8-bit 量化，进一步压缩内存。

3. **Paged Optimizer**：利用 NVIDIA 统一内存，将优化器状态在 GPU 和 CPU 之间分页，避免 OOM。

**效果**：65B 模型可以在单张 48GB A100 上用 QLoRA 微调，同等参数下效果接近全量微调 BF16 的 16-bit LoRA。

---

**Q19：全量微调 vs LoRA vs QLoRA vs Adapter，各有什么优缺点？**

| 方法 | 训练参数 | 显存需求 | 效果 | 适用场景 |
|------|---------|---------|------|---------|
| 全量微调 | 100% | 最高（模型 + 优化器状态 × 3） | 最好 | 数据充足、资源充足 |
| LoRA | 0.1-1% | 中（冻结模型 + LoRA 参数）| 接近全量 | 数据中等，资源有限 |
| QLoRA | 0.1-1% | 最低（4-bit + LoRA）| 略低于 LoRA | 单卡、资源极限 |
| Adapter | 1-5% | 中 | 接近全量 | 多任务共享 backbone |
| Prefix Tuning | <1% | 低 | 较低 | 极少参数要求 |

**工业界实践**：
- 对话模型微调：通常用 LoRA（$r=8$ 或 $r=16$），效率高效果好
- 特定领域适配（医疗、法律）：若有足够数据和资源，全量微调更彻底
- 端侧/边缘场景：QLoRA 或量化后再 LoRA

---

**Q20：LoRA 训练时 A 和 B 的初始化为什么要一个全零一个随机？**

LoRA 初始化：$A \sim \mathcal{N}(0, \sigma^2)$（随机），$B = 0$（全零）。

**原因**：训练开始时要保证 $\Delta W = BA = 0$，即初始状态下 LoRA 增量不影响原模型的前向传播结果，确保训练开始时与预训练模型行为一致（稳定的初始点）。

若 $A, B$ 都随机初始化，$BA$ 不为零，第一步前向传播的输出就与原模型不同，引入随机噪声，训练不稳定。

---

**Q21：灾难性遗忘是什么？如何缓解？**

**灾难性遗忘（Catastrophic Forgetting）**：模型在新任务上微调时，原来学到的通用能力（如基础语言能力、通用知识）快速退化的现象。

微调语言模型时，若 SFT 数据分布与预训练差异过大、学习率过高、训练步数过多，都会加剧遗忘。

**缓解方法**：

1. **较小的学习率**（$1 \times 10^{-5} \sim 5 \times 10^{-5}$）：减慢参数变化速度
2. **混入通用数据**：SFT 数据中加入 5-10% 的通用指令数据，维持通用能力
3. **LoRA / PEFT**：冻结大部分参数，减少对预训练权重的破坏
4. **EWC（Elastic Weight Consolidation）**：用 Fisher 信息矩阵标记重要参数，对重要参数的更新加大惩罚
5. **早停（Early Stopping）**：用保留的通用任务 benchmark 监控，在遗忘开始前停止

---

## 模块四：对齐（RLHF / DPO / GRPO）

---

**Q22：RLHF 的三个阶段分别做什么？为什么需要 KL 约束？**

**三个阶段**：

1. **SFT**：在高质量 (prompt, response) 演示数据上做监督微调，赋予模型基本指令遵循能力，产生 $\pi_{\text{SFT}}$
2. **奖励建模（RM）**：收集人类偏好数据 $(x, y_w, y_l)$，用 Bradley-Terry 模型训练奖励模型 $r_\phi(x, y)$：$\mathcal{L}_{\text{RM}} = -\mathbb{E}[\log\sigma(r_\phi(x, y_w) - r_\phi(x, y_l))]$
3. **RL 优化（PPO）**：用奖励模型信号，通过 PPO 最大化期望奖励，同时加 KL 约束防止退化

**KL 约束的必要性**：

$$\max_{\pi_\theta} \mathbb{E}[r_\phi(x,y)] - \beta \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$$

若没有 KL 约束，模型会 reward hacking——找到奖励模型的漏洞，生成奖励极高但实际质量很差的输出（如重复词、过度奉承）。KL 将策略锚定在 $\pi_{\text{ref}}$ 附近，保持语言合理性。

---

**Q23：DPO 是怎么推导出来的？它和 RLHF 是什么关系？**

**推导链条**：

1. RLHF 目标的最优解是 Gibbs 分布：$\pi^*(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$

2. 反解出奖励：$r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta\log Z(x)$

3. 代入 Bradley-Terry 偏好模型，$\log Z(x)$ 因为出现在差中而消去：

$$p(y_w \succ y_l|x) = \sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

4. 对偏好数据做最大似然：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

**关系**：DPO 是 RLHF 的隐式求解器，数学上等价于 RLHF 的最优解（在 Bradley-Terry 假设成立时），但不需要显式训练奖励模型，也不需要 RL 训练循环。

---

**Q24：GRPO 为什么能去掉 Critic？优势估计是怎么做的？**

**PPO 需要 Critic 的原因**：PPO 用 GAE 计算 token-level 优势，GAE 需要 Critic 网络估计每个时间步的状态价值 $V(s_t)$。

**GRPO 的做法**：对同一个问题采样 $G$ 条输出，用组内的相对奖励替代绝对价值估计：

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_1,\ldots,R_G\})}{\text{std}(\{R_1,\ldots,R_G\})}$$

**为什么合理**：LLM 的奖励本来就是 sequence-level 的（整条回复一个分），Critic 估计 token-level 价值本身就是一种不自然的强行适配，误差大。同一问题下不同回答的相对排名比跨问题的绝对价值估计更准确、更稳定。

**优势**：去掉 Critic 节省约 50% 显存，适合资源受限的训练场景（如 DeepSeek-R1 的训练）。

---

**Q25：DPO 有哪些已知局限？有哪些改进？**

**局限**：

1. **分布偏移**：DPO 使用离线数据，随着策略偏离参考策略，数据的 log prob 变小，梯度退化，语言能力下降
2. **长度偏置**：序列 log prob 是各 token log prob 之和，更长的序列天然更小，模型可能倾向生成短回答
3. **过拟合倾向**：Bradley-Terry 损失没有上界约束，奖励差 $\Delta \to \infty$ 时会过拟合
4. **单维偏好假设**：Bradley-Terry 假设偏好可由单一标量解释，实际人类偏好是多维的

**改进**：
- **IPO**：把 sigmoid 损失改为回归目标，让奖励差收敛到固定值 $\frac{1}{2\beta}$，防止过拟合
- **SimPO**：去掉参考模型，用长度归一化的 log prob 作为隐式奖励，省 50% 内存
- **KTO**：支持单边标注（只需"好"或"坏"，不需要成对比较），基于前景理论
- **Online DPO**：周期性用当前策略采样新数据，解决分布偏移

---

**Q26：PPO 在 RLHF 中的奖励塑形是怎么做的？**

LLM 的奖励是 sequence-level 的（整条回复一个分），而 PPO/GAE 需要 token-level 的奖励信号，需要做奖励塑形：

$$r_t^{\text{shaped}} = \begin{cases} r_\phi(x, y) - \beta\log\dfrac{\pi_\theta(y_t|s_t)}{\pi_{\text{ref}}(y_t|s_t)} & t = |y|（末端 token）\\ -\beta\log\dfrac{\pi_\theta(y_t|s_t)}{\pi_{\text{ref}}(y_t|s_t)} & t < |y|（中间 token）\end{cases}$$

**每个 token 上的 KL 惩罚**起到了两个作用：
1. 将 sequence-level 的奖励信号分摊到每个 token，让 GAE 有足够的信号可以计算 token-level 优势
2. 实时约束策略偏离参考模型的程度，是 KL 约束的 per-token 形式

---

## 模块五：推理能力与 CoT

---

**Q27：Chain-of-Thought（CoT）为什么有效？有哪些变体？**

**为什么有效**：

1. **计算资源**：逐步推理将复杂问题分解为多步，每步使用更多 token（计算），等价于增加推理时计算量（Test-Time Compute）
2. **中间结果作为 context**：每步的推理结果写入 context，后续步骤可以直接利用，避免在激活值中隐式维护复杂中间状态
3. **训练分布**：预训练数据中大量存在逐步推理的文本（教科书、题解），CoT 激活了这些已学习的模式

**主要变体**：

| 方法 | 描述 |
|------|------|
| Few-shot CoT | prompt 中加入"逐步推理"示例（Brown et al.）|
| Zero-shot CoT | 加"Let's think step by step"（Kojima et al.）|
| Self-Consistency | 多次采样取多数票（Wang et al.）|
| Tree of Thought | 树状搜索，探索多条推理路径（Yao et al.）|
| Program of Thought | 生成代码执行，验证中间结果 |

---

**Q28：DeepSeek-R1-Zero 是怎么做的？长链推理是怎么涌现的？**

R1-Zero 直接在 DeepSeek-V3-Base 上用 GRPO 做 RL，不经过任何 SFT。

**奖励设计**（纯 rule-based）：
- **准确率奖励**：数学题答案正确 $r=1$，错误 $r=0$；代码题通过测试用例 $r=1$
- **格式奖励**：回复必须包含 `<think>...</think>` 和 `<answer>...</answer>` 格式

**涌现过程**：

训练初期，模型生成短的简单回答。随着 RL 训练进行，模型发现：更长、更详细的推理过程（在 `<think>` 块中）能更高概率得到正确答案，从而获得更高奖励。模型自发学会：
- 验证自己的计算（"Let me check..."）
- 发现错误后重新尝试（"Wait, that's wrong..."）
- 拆解复杂问题

这些行为**没有被显式奖励**，而是通过准确率奖励隐式涌现的。

**关键发现**：AIME 2024 的 Pass@1 从 15.6% 提升到 71%，完全由 RL 驱动，无需 CoT 示范数据。

---

**Q29：Self-Consistency 是什么原理？为什么比 greedy decoding 好？**

**Self-Consistency**（Wang et al., 2022）：对同一问题用高温采样多次（如 20-40 次），对各次答案取多数票（majority vote）作为最终答案。

**原理**：

Greedy decoding 选择每步概率最高的 token，但最优解的推理路径不一定每步都是局部最优（beam search 同理）。通过多次采样，可以探索多条不同的推理路径，正确的答案往往在多条路径上都能到达（一致性更高），而错误答案更分散。

**数学直觉**：设一条推理路径正确的概率为 $p > 0.5$，取 $n$ 次多数票的正确概率为：

$$P(\text{majority correct}) = \sum_{k=\lceil n/2 \rceil}^{n} \binom{n}{k} p^k (1-p)^{n-k}$$

当 $n$ 增大时，此概率趋向 1（指数级提升）。

**代价**：推理时间和计算量线性增加（采样 $k$ 次就要 $k$ 倍计算）。

---

**Q30：Process Reward Model（PRM）和 Outcome Reward Model（ORM）的区别？**

| | ORM（结果奖励）| PRM（过程奖励）|
|-|-------------|-------------|
| 奖励粒度 | 整条回答末端 | 每个推理步骤 |
| 训练数据 | 最终答案是否正确 | 每步是否正确（需要人工标注）|
| 优势 | 数据易获取，无需标注中间步骤 | 提供更密集的监督信号，更好的过程引导 |
| 劣势 | 稀疏奖励，难以区分推理错在哪里 | 标注成本高，步骤边界难以界定 |
| 代表 | RLHF 标准做法 | OpenAI Math Shepherd, DeepSeek-R1 数据集 |

**PRM 的工程难点**：
1. 什么叫"一步"？步骤的粒度难以统一
2. 人工标注中间步骤极其昂贵
3. 自动标注 PRM（如用 Monte Carlo 估计每步后验正确率）是当前研究热点

---

## 模块六：长上下文与记忆

---

**Q31：如何扩展 LLM 的上下文长度？RoPE 的外推方法有哪些？**

**问题**：标准 RoPE 在训练时使用固定的 $\theta_i = 10000^{-2i/d}$，超出训练长度后，旋转角度到了训练中从未见过的范围，注意力权重分布混乱。

**主要方法**：

1. **Position Interpolation（PI）**：将位置索引缩放到训练范围内，即将位置 $m$ 映射为 $m \times L_{\text{train}} / L_{\text{target}}$。简单有效，但低频成分（长程关系）的分辨率下降。

2. **NTK-aware Scaling**：修改 RoPE 的基底 $\theta$，从 10000 增大到更大的值：

   $$\theta' = \theta \times \left(\frac{L_{\text{target}}}{L_{\text{train}}}\right)^{d/(d-2)}$$

   保持高频成分不变，让低频成分自然外推。无需微调即可有一定效果。

3. **YaRN（Yet Another RoPE extensioN）**：在 NTK 基础上，对不同频率成分做差异化缩放，同时在训练中加入少量长序列数据（约 0.1%），效果最好。LLaMA-3 等使用此方法扩展到 128K context。

4. **LongRoPE**：搜索更优的非均匀缩放策略，通过进化算法找最优的每维缩放因子。

---

**Q32：RAG（检索增强生成）的工作流程是什么？和长上下文各有什么优劣？**

**RAG 工作流程**：

```
用户问题
  ↓ Embedding 模型编码（如 BGE、E5）
  ↓ 向量数据库检索（如 FAISS、Milvus）
  ↓ 取 Top-K 相关文档
  ↓ 拼接到 prompt 中
  ↓ LLM 生成回答
```

**RAG vs 长上下文对比**：

| 维度 | RAG | 长上下文 |
|------|-----|---------|
| 知识更新 | 实时更新向量库即可 | 需重新训练/微调 |
| 推理成本 | 低（只塞入相关文档）| 高（注意力复杂度 $O(T^2)$）|
| 精确检索 | 取决于检索质量 | 完整信息在 context 中 |
| 长文档理解 | 分块可能割断语义 | 完整理解，无信息损失 |
| 实现复杂度 | 高（需要向量库+检索管道）| 低（直接塞文档）|

**工业界实践**：两者结合，先用 RAG 检索相关段落，再在合理的上下文窗口内生成（如 8K-32K）。

---

**Q33：Lost in the Middle 是什么问题？如何缓解？**

Liu et al.（2023）发现：当相关信息放在长上下文的中间时，LLM 的利用效果显著差于放在开头或结尾——性能呈 U 形曲线。

**原因**：Attention 机制对上下文的首尾有天然优势：
- 开头：位置 0 附近的 token 作为 KV 被大量 Q 关注（"primacy effect"）
- 结尾：最近的 token 在注意力中权重更大（局部性先验）
- 中间：既不是最近的也不是最早的，被相对忽视

**缓解方法**：
1. 重要文档放在开头或结尾
2. 多次重复重要信息
3. 在 Attention 中加入显式的位置偏置（如 ALIBI）
4. 训练数据中加入强制利用中间信息的样本

---

## 模块七：推理加速与部署

---

**Q34：大模型推理的瓶颈在哪？Memory-Bound 是什么意思？**

大模型推理（Prefill 阶段除外）的核心瓶颈是**显存带宽（Memory Bandwidth）**而非计算能力（FLOPS）。

**原因**：自回归生成时，每步只生成 1 个 token（batch size 效应为 1），矩阵乘法退化为矩阵-向量乘（matrix-vector multiply），计算量极少，但需要从显存读取全部模型权重（如 70B 模型 fp16 约 140GB）。

$$\text{Arithmetic Intensity} = \frac{\text{FLOPS}}{\text{Memory Access}} \ll \text{GPU 的 Ops:Byte 比}$$

运算密度太低，GPU 大部分时间在等显存读取，计算单元空闲，称为 **Memory-Bound**。

**加速方向**：
1. **量化**：减少权重大小（FP16→INT8→INT4），减少显存读取量
2. **Speculative Decoding**：小模型快速生成草稿，大模型并行验证
3. **Continuous Batching**：动态合并多个请求的 decode 阶段，提高并行度
4. **Tensor Parallelism**：跨多 GPU 切分 Attention/FFN，减少单 GPU 显存压力

---

**Q35：Speculative Decoding（投机解码）的原理是什么？**

**问题**：大模型每步只生成 1 个 token，每步都要完整前向传播，GPU 利用率低。

**核心思想**：用小（快）模型生成多步草稿，再用大（慢）模型并行验证，接受正确的 token。

**流程**：

```
1. 用草稿模型（Draft Model）自回归生成 K 个 token：t_1, t_2, ..., t_K
2. 把原始 context + K 个草稿 token 一次性送入目标模型（Target Model）并行前向
   → 得到 K+1 个位置的 logits（一次前向！）
3. 用拒绝采样（Rejection Sampling）逐个验证每个草稿 token：
   - 若 t_i 被接受，继续验证 t_{i+1}
   - 若 t_i 被拒绝，重新从目标模型的分布采样，停止
4. 平均接受 α·K 个 token（α 是接受率）
```

**加速比**：若接受率 $\alpha$ 高，每次目标模型的前向传播获得 $\alpha K$ 个 token（而非 1 个），吞吐量提升 $\alpha K$ 倍（实际 2-4x）。

**关键**：目标模型的输出分布不变（拒绝采样保证了这一点），生成质量与直接用目标模型完全相同，这是 Speculative Decoding 的核心保证。

---

**Q36：INT8 量化和 INT4 量化的原理？量化误差怎么减小？**

**量化原理**（以 INT8 为例）：

$$W_{\text{int8}} = \text{round}\!\left(\frac{W_{\text{fp16}}}{s}\right), \quad s = \frac{\max(|W|)}{127}$$

反量化：$W_{\text{fp16}} \approx W_{\text{int8}} \times s$

计算时用 INT8 矩阵乘（速度快、内存少），输出前乘以缩放因子 $s$ 恢复精度。

**主要量化方法**：

| 方法 | 粒度 | 说明 |
|------|------|------|
| Per-tensor | 整个矩阵一个 scale | 最粗，误差大 |
| Per-channel | 每列/行一个 scale | 主流选择，平衡效率与精度 |
| Group Quantization | 每 N 个元素一个 scale（N=128/64）| 最细，误差最小，GPTQ/AWQ 使用 |

**减小量化误差的方法**：

1. **GPTQ**：用二阶信息（Hessian 矩阵）指导量化，逐列量化并补偿后续列的误差，4-bit 几乎无损
2. **AWQ（Activation-aware Weight Quantization）**：发现权重中约 1% 的"显著权重"对输出影响极大，对这些权重用更高精度（FP16 保留），其余量化为 INT4
3. **SmoothQuant**：将激活的量化难度"迁移"到权重上（激活有离群值，权重分布平滑），使得激活和权重都容易量化

---

**Q37：vLLM 的 PagedAttention 是什么原理？**

**问题**：传统推理框架为每个请求的 KV Cache 预分配连续显存（按最大序列长度），导致严重的内存碎片和浪费（平均利用率 20-40%）。

**PagedAttention**：借鉴操作系统虚拟内存的分页机制，将 KV Cache 切分为固定大小的 **Block（页）**，不同请求的 Block 可以不连续存储，由逻辑-物理地址映射表（Block Table）管理。

**优势**：
1. **消除内存碎片**：按需分配 Block，不浪费
2. **内存共享**：Prefix Sharing（相同 system prompt 的 KV 只存一份，多请求共享）
3. **显存利用率** 从 20-40% 提升到 ~90%+
4. **吞吐量** 提升 2-4x（同样显存能 serve 更多并发请求）

vLLM 结合 Continuous Batching + PagedAttention，是目前工业界 LLM 推理的事实标准。

---

**Q38：Tensor Parallelism、Pipeline Parallelism、Data Parallelism 各是什么？**

**Data Parallelism（DP）**：每个 GPU 存一份完整模型，不同 GPU 处理不同 batch 数据，梯度同步后更新。适合模型能放入单卡的情况。

**Tensor Parallelism（TP，Megatron-LM）**：将单个矩阵切分到多个 GPU 上并行计算。如 MLP 的 $W_1 \in \mathbb{R}^{d \times 4d}$，按列切成 $n$ 份，每 GPU 处理一列分片，最后 AllReduce 聚合。适合大矩阵，通信量大但延迟低（NVLink 场景）。

**Pipeline Parallelism（PP）**：将模型的不同层分配给不同 GPU，数据如流水线穿过各 GPU。通信量少，但存在 pipeline bubble（GPU 等待前一阶段完成）。适合跨节点（InfiniBand 网络，带宽低）。

**3D 并行（Megatron-DeepSpeed）**：DP + TP + PP 三者结合，是训练超大模型（如 GPT-3 175B、LLaMA-3 405B）的标准方案：
- PP：跨节点，减少通信量
- TP：节点内（NVLink），利用高带宽
- DP：数据并行扩展 batch

---

## 模块八：多模态与其他

---

**Q39：视觉语言模型（VLM）是如何将图像和文本对齐的？LLaVA 的方法是什么？**

**核心问题**：图像特征（来自 Vision Encoder，如 CLIP ViT）和文本 token 处于不同的特征空间，需要对齐。

**LLaVA 的方法（两阶段）**：

1. **阶段一（特征对齐预训练）**：
   - 用一个简单的线性层（Projection Layer）将 CLIP 的图像特征 $\mathbb{R}^{N_{\text{patches}} \times d_{\text{vision}}}$ 投影到 LLM 的 embedding 空间 $\mathbb{R}^{N_{\text{patches}} \times d_{\text{llm}}}$
   - 用图文对数据（CC-595K）训练，只更新 Projection Layer，冻结 CLIP 和 LLM
   - 目标：让 visual tokens 与 word tokens 处于同一语义空间

2. **阶段二（指令微调）**：
   - 解冻 LLM，与 Projection Layer 一起微调
   - 数据：LLaVA-Instruct（GPT-4 生成的多模态对话数据）

**LLaVA-1.5** 改进：将线性 Projection 换为 2 层 MLP，效果大幅提升。

**现代方案（InternVL、Qwen-VL 等）**：
- 更强的 Vision Encoder（ViT-22B、InternViT）
- 动态分辨率（将高分辨率图像切片分别编码）
- Cross-attention（Flamingo 风格）或直接 concat（LLaVA 风格）

---

**Q40：混合专家模型（MoE）的负载均衡问题是怎么解决的？**

**负载均衡问题**：如果路由器（Router）自由选择，可能导致"专家崩溃"——大多数 token 总是路由到少数几个专家，其他专家几乎不被激活，MoE 退化为小的密集模型。

**主要解决方案**：

1. **辅助负载均衡损失（Auxiliary Loss）**：

$$\mathcal{L}_{\text{balance}} = \alpha \sum_{i=1}^{N} f_i \cdot P_i$$

其中 $f_i$ 是专家 $i$ 被路由的 token 比例，$P_i$ 是路由概率的均值。目标是让 $f_i$ 尽量均匀，通常 $\alpha = 0.01$。

2. **Expert Capacity**：为每个专家设置容量上限（capacity factor），超过容量的 token 被丢弃或路由到第二选择。

3. **DeepSeek 的改进（DeepSeekMoE）**：
   - 将 $N$ 个专家中的部分设为**共享专家（Shared Experts）**，所有 token 都会激活，确保基础能力
   - 其余为**路由专家（Routed Experts）**，按需激活

4. **Token Choice vs Expert Choice**：传统 Token Choice（每个 token 选 Top-K 专家），Expert Choice（每个专家选 Top-K token，天然负载均衡但需要 batch 级别同步）。

---

## 模块九：训练工程与稳定性

---

**Q41：大模型训练中 Loss Spike 是什么？怎么处理？**

**Loss Spike**：训练过程中 loss 突然急剧上升（有时 2-10x），然后缓慢恢复（也可能不恢复）的现象。在数百亿参数的大模型训练中频繁出现。

**常见原因**：
1. 一批特殊数据（极长文本、极高困惑度、低质量文本）引发梯度爆炸
2. 学习率过高，参数更新超出稳定域
3. 数值问题（accumulation over long sequence with BF16）

**处理方法**：

1. **梯度裁剪**（grad norm clipping）：阈值通常 1.0，防止单步更新过大
2. **从 spike 前的 checkpoint 恢复**：回退 100-500 步，跳过这批数据继续训练
3. **数据清洗**：分析 spike 时的训练数据，识别并移除高困惑度样本
4. **动态学习率**：spike 后临时降低学习率
5. **z-loss**（PaLM 提出）：对 logits 的 log-sum-exp 加小惩罚，稳定 softmax 输出

---

**Q42：混合精度训练（Mixed Precision Training）是怎么工作的？**

标准混合精度训练（AMP）流程：

```
FP32 主权重（master weights）
  ↓ 复制为 BF16/FP16
  → BF16 前向传播（计算速度快）
  → BF16 反向传播（计算梯度）
  → 梯度转回 FP32
  → FP32 权重更新（精度高，防止更新丢失）
  → 再复制为 BF16
```

**为什么保留 FP32 主权重**：大模型训练时学习率很小（$10^{-4}$ 以下），每步的参数更新量 $\Delta w = \alpha \nabla w$ 极小（相对于权重本身）。BF16/FP16 的精度不足以表示这么小的增量，若直接在 BF16 上更新，小增量被舍入为 0，权重无法更新。FP32 的精度（约 $10^{-7}$）足够捕捉这些微小更新。

**内存代价**：FP32 主权重 + BF16 工作权重 = 约 1.5x 权重大小的额外内存。

---

**Q43：ZeRO（Zero Redundancy Optimizer）的原理是什么？**

**问题**：数据并行训练中，每个 GPU 都存一份完整的模型参数、梯度、优化器状态（Adam 的 $m, v$，FP32 主权重），严重冗余。

ZeRO（DeepSpeed）将这些状态分片到各 GPU，每个 GPU 只存 $\frac{1}{N}$：

| ZeRO Stage | 分片内容 | 内存节省 |
|------------|---------|---------|
| Stage 1 | 优化器状态 | ~4x |
| Stage 2 | 优化器状态 + 梯度 | ~8x |
| Stage 3 | 优化器状态 + 梯度 + 参数 | ~N x |

**ZeRO-3 的通信代价**：前向传播时需要 AllGather 参数（GPU 间同步所有分片），反向传播后需要 Reduce-Scatter 梯度。通信量与 DDP（Data Parallel）相当，但内存大幅降低。

**ZeRO-Offload**：将 Stage 1/2 的优化器状态下放到 CPU 内存，进一步降低 GPU 内存需求，代价是 CPU-GPU 数据传输延迟。

---

## 模块十：评估与幻觉

---

**Q44：LLM 的幻觉（Hallucination）有哪些类型？根本原因是什么？**

**类型**：

1. **事实性幻觉**：生成错误事实（如错误的历史日期、不存在的论文引用）
2. **上下文幻觉**：回答与输入 context 矛盾
3. **指令幻觉**：没有遵循用户指令，生成了与要求不符的内容

**根本原因分析**：

1. **训练目标的本质**：Next Token Prediction 的目标是最大化训练数据的对数似然，而非"说真话"。模型学会了语言模式，但没有理由区分真实信息和合理但错误的信息。

2. **知识的模糊边界**：模型权重中的知识是隐式的、模糊的，模型无法区分"确定知道"和"不确定但合理猜测"。

3. **暴露偏差（Exposure Bias）**：训练时用真实 token 作为 context（Teacher Forcing），推理时用自己生成的 token，错误会在序列中级联放大。

4. **RLHF 加剧幻觉**：奖励模型倾向于给听起来"自信、流畅"的回答打高分，模型学会了即使不确定也要自信地回答。

**缓解方法**：RAG（引入外部知识源）、不确定性估计、后验校正、训练时加入"我不知道"的样本。

---

**Q45：如何评估 LLM 的能力？常用 benchmark 有哪些？**

**评估维度与 benchmark**：

| 维度 | Benchmark | 说明 |
|------|-----------|------|
| 通用推理 | MMLU（57 个学科选择题） | 知识广度 |
| 数学 | MATH、AIME、GSM8K | 数学推理，AIME 最难 |
| 代码 | HumanEval、MBPP、LiveCodeBench | 代码生成 |
| 常识推理 | HellaSwag、ARC、WinoGrande | 常识理解 |
| 中文能力 | C-Eval、CMMLU | 中文知识 |
| 长文本 | RULER、Needle-in-a-Haystack | 长上下文能力 |
| 对话 | MT-Bench、AlpacaEval | 指令遵循 |
| 安全 | AdvBench、TruthfulQA | 安全性和真实性 |

**评估的注意事项**：
- **数据污染（Data Contamination）**：benchmark 数据可能出现在训练集中，导致虚高分数。GPT-4 等已有污染检测机制
- **LLM-as-Judge**：用 GPT-4 对模型输出打分（如 MT-Bench），更接近人类偏好但受评判模型自身偏见影响
- **动态 benchmark**：LiveCodeBench 等实时更新题目，防止污染

---

**Q46：为什么 benchmark 分数高但实际使用体验差？怎么更好地评估？**

**原因**：

1. **分布不匹配**：benchmark 题目有固定格式，真实用户问题千变万化
2. **单一维度**：MMLU 等只测知识广度，不测指令遵循、上下文理解、风格一致性
3. **数据污染**：模型在训练集中见过 benchmark 答案
4. **形式评估 vs 实质评估**：选择题得分高不等于开放生成质量好

**更好的评估方式**：

1. **人工评估 + 偏好排序**：A/B test，人类标注者比较两个模型的回答
2. **LLM-as-Judge + 盲评**：GPT-4 评判，隐藏模型名称（避免位置偏差、名称偏差）
3. **领域专项测试**：根据实际业务场景构建私有评测集
4. **在线 A/B 测试**：真实用户的隐式反馈（停留时长、满意度、重复使用率）

---

## 模块十一：经典面试手撕题

---

**Q47：手写 Scaled Dot-Product Attention（PyTorch）**

```python
import torch
import torch.nn.functional as F
import math

def attention(q, k, v, mask=None):
    # q, k: (B, H, T, d_k)
    # v: (B, H, T, d_v)
    d_k = q.size(-1)
    # (B, H, T, T)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
```

---

**Q48：手写 LoRA 前向传播**

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features), requires_grad=False
        )  # 冻结原始权重
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # B 初始化为 0
        self.scale = alpha / r

    def forward(self, x):
        # 原始线性层（不更新梯度）
        base_out = F.linear(x, self.weight)
        # LoRA 增量：x @ A^T @ B^T * scale
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base_out + lora_out
```

---

**Q49：手写 RMSNorm**

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # x: (..., d_model)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.gamma
```

---

**Q50：手写 DPO Loss**

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    policy_*_logps: log π_θ(y|x)，形状 (B,)
    ref_*_logps:    log π_ref(y|x)，形状 (B,)
    """
    chosen_logratios  = policy_chosen_logps  - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()
    # 监控指标
    reward_acc = (chosen_logratios > rejected_logratios).float().mean()
    return loss, reward_acc
```

---

*以上题目覆盖了字节跳动（Doubao/豆包）、阿里巴巴（通义千问/Qwen）、腾讯（混元）、百度（文心）、华为（盘古）、DeepSeek、智谱 AI（GLM）、商汤（日日新）、minimax、月之暗面（Kimi）等大厂的高频考点。建议结合对应原论文深入理解每个问题背后的数学推导。*
