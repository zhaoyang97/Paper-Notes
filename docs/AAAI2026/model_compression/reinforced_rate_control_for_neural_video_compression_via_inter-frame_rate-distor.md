---
title: >-
  [论文解读] Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness
description: >-
  [AAAI 2026][模型压缩][神经视频压缩] 提出首个基于约束马尔可夫决策过程（CMDP）的强化学习速率控制框架，通过时空状态建模联合捕获帧内容特征与帧间率-失真耦合依赖，直接映射到逐帧编码参数，在多种神经视频编解码器上将平均比特率误差降至1.20%，BD-Rate节省最高达13.98%。 问题背景 神经视频压缩（NV…
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "神经视频压缩"
  - "速率控制"
  - "强化学习"
  - "帧间依赖"
  - "Actor-Critic"
  - "率失真优化"
---

# Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness

**会议**: AAAI 2026  
**arXiv**: [2601.19293](https://arxiv.org/abs/2601.19293)  
**作者**: Wuyang Cong, Junqi Shi, Lizhong Wang, Weijing Shi, Ming Lu, Hao Chen, Zhan Ma (南京大学)  
**代码**: 待公开  
**领域**: 模型压缩  
**关键词**: 神经视频压缩, 速率控制, 强化学习, 帧间依赖, Actor-Critic, 率失真优化  

## 一句话总结

提出首个基于约束马尔可夫决策过程（CMDP）的强化学习速率控制框架，通过时空状态建模联合捕获帧内容特征与帧间率-失真耦合依赖，直接映射到逐帧编码参数，在多种神经视频编解码器上将平均比特率误差降至1.20%，BD-Rate节省最高达13.98%。

## 研究背景与动机

### 问题背景
神经视频压缩（NVC）利用深度神经网络的非线性建模能力和端到端优化，已在压缩效率上超越传统视频编码标准（如VVC/H.266）。然而，速率控制（rate control）——即在满足目标比特率约束下最大化重建质量——在NVC中仍是一个被严重忽视的关键实用性问题。速率控制本质上需要学习一个策略，为每帧分配目标比特率并映射到最优编码参数（如Lagrange乘子$\lambda$和分辨率缩放因子$m$）。

### 已有工作的不足
现有NVC速率控制方法沿袭传统编解码器的思路，采用基于窗口（如GOP）的方案：先将目标比特率均匀分配到窗口，然后在窗口内用规则或启发式策略分配帧级比特率。这些方法**仅建模帧间失真依赖**，假设帧间比特率依赖可忽略。然而，NVC因其联合优化的像素级、特征级和上下文参考信息，表现出复杂且紧密耦合的率-失真依赖关系。一旦引入速率控制，即使参考帧编码参数的微小变化也会导致后续帧率-失真行为的显著偏移，使得基于静态R-D假设的方法产生次优的比特率分配和编码参数选择。

### 核心动机
NVC中的帧间依赖不仅涉及失真传播，还包含**比特率依赖**——参考帧的编码参数直接影响时间上下文建模，改变后续帧的概率分布估计，从而影响实际比特率。传统方法忽略这种耦合依赖会导致级联的参数决策失误。需要一种能同时建模帧内容与参考帧信息的动态策略，从全局视角优化逐帧编码决策。

## 核心问题

给定视频序列 $\mathcal{X} = \{x_1, x_2, \ldots, x_N\}$ 和目标比特率 $R_{tar}$，速率控制需求解最优编码参数集 $\Pi^{(\mathcal{X})} = \{\boldsymbol{a}_1, \ldots, \boldsymbol{a}_N\}$：

$$\Pi^{(x_t)} = \arg\min_{\Pi^{(\mathcal{X})}} \sum_{t=1}^{N} D_t, \quad \text{s.t.} \quad \frac{1}{N}\sum_{t=1}^{N} R_t \leq R_{tar}$$

引入全局Lagrange乘子 $\Lambda$ 后的必要条件为：

$$\sum_{i=t}^{N} \left( \frac{\Lambda}{N} \frac{\partial R_i}{\partial \mathbf{a}_t} - \frac{\partial D_i}{\partial \mathbf{a}_t} \right) = \sum_{i=t}^{N} \left( \left(\frac{\Lambda}{N} - \lambda_i\right) \frac{\partial R_i}{\partial \mathbf{a}_t} \right) = 0$$

这表明最优编码参数 $\mathbf{a}_t$ 不仅需反映当前帧的R-D行为，还需考虑对未来帧的率-失真传播影响——这是一个NP-hard的全局优化问题。

## 方法详解

### 整体框架：增强型Actor-Critic

将速率控制建模为CMDP，设计包含三个核心组件的Actor-Critic框架：

### 1. 时空状态建模（State Modeling）

状态表示需同时编码历史编码参考信息和当前帧特征。具体地：
- 当前帧 $x_t$ 和参考帧 $x_{t-1}$ 拼接后送入级联残差网络提取时空特征
- 融合编解码器在多分辨率下提取的 $x_{t-1}$ 中间特征以增强时间上下文
- 辅助信息（目标比特率、历史编码参数）经归一化、扩展后通过全连接层嵌入
- 视觉嵌入与辅助嵌入组合形成完整的可学习状态表示

### 2. 动作决策（Action Decision）

动作定义为连续编码参数对 $\{\lambda_t, m_t\}$：
- $\lambda_t \in [\lambda_{min}, \lambda_{max}]$：Lagrange乘子，控制R-D行为
- $m_t \in [0.5, 1.0]$：下采样因子，调整当前帧和参考帧的空间分辨率

策略 $\pi_\phi$ 建模为高斯分布，均值和方差由Actor网络预测。引入策略熵正则化鼓励探索，Actor梯度为：

$$J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{S}, a_t \sim \pi_\phi} \left[ \epsilon \log \pi_\phi(a_t | s_t) - Q_\theta(s_t, a_t) \right]$$

推理时采用贪心策略选择最高似然动作。当 $m_t < 1.0$ 时，对参考帧重采样至当前分辨率以保持帧间预测一致性，输出再双三次上采样回原始分辨率。

### 3. 奖励塑形（Reward Shaping）

速率控制中有意义的指标（总失真、比特率偏差）仅在编码完整序列后可用，导致奖励稀疏。设计加权内积形式的奖励：

$$r_t = -\mathbf{w}_t^\top \mathbf{f}_t, \quad \mathbf{f}_t = \begin{pmatrix} D_t \\ \frac{|R_{\text{rem}}|}{R_{\text{tar}}} \end{pmatrix}, \quad \mathbf{w}_t = \begin{pmatrix} \delta_t \\ \eta_t \end{pmatrix}$$

其中 $R_{\text{rem}}$ 为剩余比特率预算，$\mathbf{w}_t = (\delta_t, \eta_t)^\top$ 平衡失真和比特率精度，每 $\mathcal{K}$ 步根据验证反馈自适应更新。最后一帧施加较大 $\eta_t$ 以强制严格速率控制。采用Twin-Critic架构估计两个独立Q值并取最小值以缓解过估计偏差，同时建模完整回报分布以增强鲁棒性。

## 实验关键数据

### 实验设置
- **编解码器**：DVC、DCVC、DCVC-DC、DCVC-RT
- **数据集**：UVG、MCL-JCV、HEVC Class B/C/D/E
- **GOP大小**：32和100
- **对比方法**：Zhang et al. (2023)、Chen et al. (2023)

### 实验1：速率控制性能对比（GOP=32，$\Delta R$ ↓ / BD-Rate(%) ↓）

| 编解码器 | 方法 | UVG | HEVC B | HEVC E | 平均 |
|---------|------|-----|--------|--------|------|
| DCVC | Chen et al. | 1.85 / -14.28 | 1.96 / -12.23 | 1.18 / -15.25 | 1.76 / -12.18 |
| DCVC | **Ours** | **1.80 / -18.24** | **1.15 / -14.83** | **0.99 / -18.76** | **1.48 / -16.49** |
| DCVC-DC | Chen et al. | 1.66 / -10.33 | 1.71 / -11.02 | 1.28 / -13.00 | 1.61 / -10.63 |
| DCVC-DC | **Ours** | **1.45 / -13.84** | **0.98 / -14.82** | **0.85 / -16.70** | **1.13 / -13.98** |
| DCVC-RT | Chen et al. | 1.49 / -5.12 | 1.44 / -5.26 | 1.50 / -4.98 | 1.45 / -4.81 |
| DCVC-RT | **Ours** | **1.18 / -5.84** | **1.16 / -6.00** | **0.96 / -6.17** | **1.15 / -5.50** |

### 实验2：长GOP性能对比（GOP=100）

| 编解码器 | 方法 | 平均 $\Delta R$ ↓ | 平均 BD-Rate ↓ |
|---------|------|-------------------|----------------|
| DCVC-DC | Zhang et al. | 1.85% | -5.41% |
| DCVC-DC | Chen et al. | 1.62% | -10.42% |
| DCVC-DC | **Ours** | **1.09%** | **-13.93%** |
| DCVC-RT | Chen et al. | 1.32% | -5.39% |
| DCVC-RT | **Ours** | **0.98%** | **-6.03%** |

随着底层编解码器提升（DVC→DCVC-RT），Zhang et al.的BD-Rate增益急剧下降至-5.41%，而本文方法保持稳定的-13.93%增益。

### 实验3：计算复杂度对比（基于DCVC-RT）

| 方法 | 参数量 | KMACs/pxl | 显存(GB) | 编码FPS | 解码FPS |
|------|--------|-----------|----------|---------|---------|
| Baseline | 66.33M | 421.31 | 2.27 | 102 | 95 |
| Zhang et al. | +2.12M | +6.40 | +1.22 | 68 | — |
| Chen et al. | — | — | — | 54 | 108 |
| **Ours** | **+0.57M** | **+1.60** | **+0.33** | **111** | **109** |

本方法额外开销最小（仅+0.57M参数），且通过下采样操作反而**提升**了编解码吞吐量。

### 实验4：训练帧数消融

| 训练帧数 | 4 | 8 | 16 | 32 | 64 |
|---------|---|---|----|----|-----|
| BD-Rate(%) | -8.84 | -11.15 | -15.03 | -16.49 | -16.90 |
| $\Delta R$(%) | 2.48 | 1.82 | 1.67 | 1.48 | 1.43 |

增加训练帧数稳定提升R-D性能和速率精度，验证了方法对帧间依赖的有效建模；同时训练复杂度随序列长度**线性增长**。

## 亮点

- **首个CMDP建模的NVC速率控制**：将速率控制形式化为CMDP，通过RL框架直接优化逐帧编码参数，避免传统两步式（先分配比特率再映射参数）的次优决策
- **帧间率-失真耦合依赖的深刻分析**：通过理论推导和实验验证，揭示NVC中参考帧编码参数变化导致的R-D行为偏移——这是传统方法忽略的核心问题
- **极低计算开销**：仅+0.57M参数和+0.33GB显存，甚至通过下采样提升了编解码吞吐量，高度实用
- **强泛化能力**：在360°视频等未见内容上仍保持3.9%的低比特率偏差，显著优于Zhang et al.的7.6%
- **跨编解码器一致性**：在DVC、DCVC、DCVC-DC、DCVC-RT四种架构上均取得一致的性能提升

## 局限与展望

- **未考虑网络传输条件**：当前仅优化编码端的率-失真性能，未整合丢包、拥塞等网络传输因素
- **分辨率缩放的信息损失**：下采样因子$m_t < 1.0$时通过双三次上采样恢复分辨率，可能引入模糊等伪影
- **RL训练成本**：需要预训练50个epoch（4帧）+ 250个epoch（32帧），训练过程较长
- **GOP结构依赖性**：虽然声称独立于GOP结构，但实验仅在LDP配置下评估，未验证RA（随机访问）等场景
- **Action空间有限**：仅包含 $\lambda_t$ 和 $m_t$，未探索量化步长、滤波强度等更多编码参数

## 与相关工作的对比

- **Zhang et al. (2023)**：使用神经网络预测比特率分配和R-λ映射，但不考虑帧间比特率依赖，在先进编解码器上BD-Rate增益急剧下降
- **Chen et al. (2023)**：用双曲函数建模R-λ-m和D-λ-m关系并迭代更新，但需预编码等距帧初始化，增加编码时间
- **Li et al. (2022)**：首个NVC速率控制方法，采用固定R-D-λ模型，灵活性不足
- **传统编解码器RL方法** (Zhou et al. 2021; Ho et al. 2021)：基于传统规则工具用RL探索更优规则，或依赖多次预编码的启发式搜索，不适用于NVC
- **DCVC-RT内置速率控制** (Jia et al. 2025)：通过分层质量训练隐式分配比特率，但缺乏显式速率控制能力

## 启发与关联

- RL在视频编码速率控制中的应用可推广到其他序列决策场景，如自适应流媒体传输
- 帧间率-失真耦合依赖的发现可能也适用于其他端到端视频处理任务（如视频超分辨率、视频增强）
- 时空状态建模的思路可用于其他需要同时考虑当前输入和历史上下文的RL任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统化分析NVC中帧间率-失真耦合依赖并用CMDP建模，视角新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 四种编解码器、六个数据集、两种GOP设置、完整消融，评估全面
- 写作质量: ⭐⭐⭐⭐ — 问题建模严谨，从理论推导到实验验证逻辑清晰
- 价值: ⭐⭐⭐⭐ — 为NVC提供了实用的速率控制解决方案，计算开销极低适合实际部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] RADIO: Rate-Distortion Optimization for Large Language Model Compression](../../ICML2025/model_compression/radio_rate-distortion_optimization_for_large_language_model_compression.md)
- [\[CVPR 2026\] Real-Time Neural Video Compression with Unified Intra and Inter Coding](../../CVPR2026/model_compression/real-time_neural_video_compression_with_unified_intra_and_inter_coding.md)
- [\[ICLR 2026\] Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport](../../ICLR2026/model_compression/cross_domain_lossy_compression_optimal_transport.md)
- [\[CVPR 2026\] Perceptual Neural Video Compression with Color Separation and Rank Chain](../../CVPR2026/model_compression/perceptual_neural_video_compression_with_color_separation_and_rank_chain.md)
- [\[ICML 2026\] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding](../../ICML2026/model_compression/lk_losses_direct_acceptance_rate_optimization_for_speculative_decoding.md)

</div>

<!-- RELATED:END -->
