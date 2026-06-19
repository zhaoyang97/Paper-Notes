---
title: >-
  [论文解读] Inner Speech as Behavior Guides: Steerable Imitation of Diverse Behaviors for Human-AI Coordination
description: >-
  [NeurIPS 2025 Spotlight][机器人][模仿学习] 受维果茨基内心语言理论启发，提出 MIMIC 框架，利用语言作为感知与动作之间的中介表征，通过 VLM 提供语言脚手架训练 CVAE 生成内心语言，再以扩散策略在条件化于内心语言的情况下生成多样且可控的行为。 领域现状：模仿学习（IL）是构建类人 AI…
tags:
  - "NeurIPS 2025 Spotlight"
  - "机器人"
  - "模仿学习"
  - "内心语言"
  - "行为多样性"
  - "人机协作"
  - "扩散策略"
---

# Inner Speech as Behavior Guides: Steerable Imitation of Diverse Behaviors for Human-AI Coordination

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2602.20517](https://arxiv.org/abs/2602.20517)  
**代码**: [项目主页](https://mimic-research.github.io)  
**领域**: 强化学习 / 模仿学习  
**关键词**: 模仿学习, 内心语言, 行为多样性, 人机协作, 扩散策略

## 一句话总结

受维果茨基内心语言理论启发，提出 MIMIC 框架，利用语言作为感知与动作之间的中介表征，通过 VLM 提供语言脚手架训练 CVAE 生成内心语言，再以扩散策略在条件化于内心语言的情况下生成多样且可控的行为。

## 研究背景与动机

**领域现状**：模仿学习（IL）是构建类人 AI agent 的有前途范式，行为克隆（BC）通过监督学习从演示中学习策略，近期扩散策略和 Transformer 架构大幅提升了性能。

**现有痛点**：现有 BC 方法 ① 难以捕获人类行为的多模态分布（不同动机和技能水平导致的行为多样性）；② 缺乏推理时的行为可控性——要么完全不可控，要么仅支持目标条件生成；③ 忽略了行为的非马尔可夫特性（人类决策受历史和内在动机影响）。

**核心矛盾**：传统 BC 直接学习 $s_t \mapsto a_t$ 的映射，但认知科学研究表明人类决策通过内心语言中介 $s_t \to m_t \to a_t$——相同环境刺激可因不同内心独白产生不同行为响应。

**本文目标**：构建能生成多样行为、支持细粒度行为引导且无需额外演示的模仿学习框架。

**切入角度**：将维果茨基的内心语言理论形式化为计算框架——语言不仅是通信工具，更是组织行为的内部认知机制。

**核心 idea**：$P_\mathcal{H}(a|s) = \int p_\mathcal{H}(a|s,m) p_\mathcal{H}(m|s) dm$——将动作视为状态和内心语言的联合条件分布，内心语言由 CVAE 自主生成，且可被设计者用文本描述覆盖以实现可控生成。

## 方法详解

### 整体框架

MIMIC 由三个组件构成：① VLM 语言脚手架（训练时生成行为描述）→ ② CVAE 内心语言生成器（从行为历史自主生成内心语言嵌入）→ ③ 扩散策略行为克隆器（条件化于观测+内心语言生成动作）。推理时 agent 周期性地自主生成内心语言，同时支持设计者通过文本注入初始内心语言实现行为引导。

### 关键设计

1. **内心语言条件化行为克隆器（Inner Speech-conditioned Behavior Cloner）**：

    - 功能：以观测 $s$ 和内心语言 $m$ 为条件，通过扩散过程生成动作
    - 为什么：条件化于内心语言使相同状态可产生不同行为模式，实现多样性
    - 怎么做：采用 DDPM-T（扩散策略+Transformer 架构），训练去噪网络 $\hat{\epsilon}_\theta(\hat{\mathbf{a}}_\tau, s, m, \tau)$，损失函数为：
    $\mathcal{L}_{\text{diff}}(\mathcal{D}_M) = \mathbb{E}_{(s,\mathbf{a}_0,m)\sim\mathcal{D}_M, \tau\sim[1,T_D]} \|\hat{\epsilon}(\mathbf{a}_\tau, s, m, \tau) - \epsilon\|_2^2$
    - 使用 classifier-free guidance：训练时以概率 $p$ 随机将 $m$ 置零
    - 输入编码：状态用领域特定编码器（CNN/特征），$m$ 用 2 层 MLP，动作用线性编码，时间步用余弦编码，拼接后送入 Transformer
    - 区别：不同于直接 $s \to a$ 的 BC，引入语言中介变量

2. **行为条件化内心语言生成器（Behavior-Conditioned Inner Speech Generator）**：

    - 功能：从行为历史自主生成内心语言嵌入
    - 为什么：推理时无法访问 VLM，agent 需要自行生成行为引导
    - 怎么做：使用 CVAE，编码器和解码器都条件化于过去图像序列（通过卷积表征池化）。CVAE 生成 CLIP 编码的内心语言，训练目标：
    $\mathcal{L}_{\text{is}}(\mathcal{D}_M) = \sum_{i=1}^{n}\sum_{t=W}^{T} \|m^{(i)} - \Psi_{\text{dec}}(\Psi_{\text{enc}}(m^{(i)}, \mathbf{I}_{t-W:t}^{(i)}), \mathbf{I}_{t-W:t}^{(i)})\|_2^2 + \beta \Delta_{KL}(\Psi_{\text{enc}}(m^{(i)}), \mathcal{N}(\mathbf{0}, \mathbf{I}))$
    - $\beta$ 在训练过程中退火，模拟从外部详细语言到压缩自主内心语言的过渡
    - 周期更新：每 $W$ 步生成新内心语言，$m_t = \Psi_{\text{dec}}(\Psi_{\text{enc}}(\mathcal{H}_t), \mathcal{H}_t)$ if $t \bmod W = 0$，否则 $m_t = m_{t-1}$

3. **VLM 语言脚手架（Vision-Language Model Scaffolding）**：

    - 功能：为每条演示轨迹生成行为描述作为内心语言的训练目标
    - 为什么：演示数据中不包含人类思维注释，需要外部模型补充语言标签
    - 怎么做：将演示图像序列转为 GIF，发送给 VLM（GPT-4o），用精心设计的 prompt 让 VLM 生成区分不同行为的描述，然后用 CLIP 编码为嵌入 $m^{(i)}$
    - 区别：不需要 Thought Cloning 那样的逐步人类思维注释，仅需轨迹级描述，更可扩展

4. **可控行为引导**：

    - 推理时通过两种机制控制行为：① 设计者提供文本描述作为初始内心语言 $m \leftarrow \mathcal{B}$；② 设置轮询窗口 $W$ 控制内心语言更新频率
    - 可在不训练额外演示的情况下实现行为风格切换

### 理论基础

基于维果茨基内心语言理论的三个结构属性：
- **述谓性（Predicativity）**：内心语言强调关系和动作而非实体
- **语义浓缩（Semantic Condensation）**：复杂策略压缩为紧凑表征（对应 CVAE 信息瓶颈）
- **时间调节动态（Temporal Regulatory Dynamics）**：内心语言在扩展时间尺度上运作，条件化于行为历史

### 损失函数 / 训练策略

- 行为克隆器和内心语言生成器解耦训练
- Adam 优化器，学习率按数据集调整
- 初始更新步 $t_0 \in \{0, 1, W/2, W-1\}$，窗口 $W \in \{1, 10, 20, 50, 100\}$（低时间步任务）或 $W \in \{100, 200, 300\}$（高时间步）
- 随机 dropout 概率 $p \in \{0, 0.1\}$，CVAE $\beta = 0.1$

## 实验关键数据

### 主实验：D3IL 基准

| 环境 | 方法 | 成功率↑ | 距离↓ | 熵↑ | State-wass↓ |
|------|------|---------|-------|-----|-------------|
| Aligning | BC | 0.665 | 0.111 | 0.474 | 0.696 |
| Aligning | MIMIC-S | **0.802** | **0.066** | 0.418 | **0.046** |
| Aligning | MIMIC-E | 0.723 | 0.085 | **0.615** | 0.049 |
| Sorting-Vision | BC | 0.797 | - | 0.360 | - |
| Sorting-Vision | MIMIC-S | **0.842** | - | 0.372 | - |
| Sorting-Vision | MIMIC-E | 0.808 | - | **0.449** | - |
| Stacking (1/2box) | BC | 0.803/0.488 | - | 0.206/0.150 | - |
| Stacking (1/2box) | MIMIC-E | **0.821/0.533** | - | **0.212/0.156** | - |

MIMIC 在成功率和行为多样性（熵）上全面超越 SOTA BC 基线。

### Overcooked 人机协作

| 环境 | BC | MIMIC | 提升 |
|------|-----|-------|------|
| Cramped room | 115.8 ± 3.86 | **151.8 ± 2.45** | +36 |
| Cramped room-Vision | 73.6 ± 6.18 | **108.8 ± 4.84** | +35 |
| Coordination ring | 113.0 ± 2.21 | **121.0 ± 1.93** | +8 |
| Asymmetric advantages | 215.8 ± 3.04 | **227.6 ± 2.69** | +12 |

集体奖励显著提升，最高涨 36 分。

### 消融实验

| 内心语言形式 | 成功率 | 熵 |
|-------------|--------|-----|
| MIMIC (语言) | **最佳** | **最佳** |
| 随机向量 | 优于BC | 中等 |
| K-means聚类 | 最差 | 最低 |
| 无内心语言 (BC) | 基线 | 基线 |

- 语言形式显著优于随机/聚类替代
- CLIP 优于纯文本 MPNET 编码（共享视觉-语言空间有帮助）
- GPT-4o 生成的描述最优，开源 Qwen 紧随其后
- 可控生成：GPT-4o 评判得分 3.83~4.23/5.0，验证行为可控性

### 关键发现

- 内心语言在视觉观测环境中同样有效，说明语言提供了状态信息之外的补充
- Wasserstein 距离大幅降低，说明 MIMIC 的行为分布更接近真实人类演示
- 周期性更新内心语言有助于纠正行为偏移（BC 的固有脆弱性）

## 亮点与洞察

- **认知科学与 ML 的创新结合**：将维果茨基内心语言理论形式化为计算框架，理论基础扎实
- **三重能力达标**：高保真模仿 + 行为多样性 + 推理时可控性，不需要额外演示
- **VLM 脚手架巧妙**：避免了逐步人类思维注释的昂贵需求，仅需轨迹级 VLM 描述
- **信息瓶颈 ↔ 语义浓缩**：CVAE 的变分压缩自然对应内心语言从详细外部描述到压缩自主表征的过渡
- **实用的行为纠错机制**：周期性内心语言更新可作为行为克隆脆弱性的缓解策略

## 局限与展望

- VLM 生成描述的质量和多样性影响上限（o4-mini 描述过于细腻反而导致 CLIP 无法区分）
- 目前仅在相对简单的操控和 Overcooked 环境验证，复杂真实世界任务有待探索
- CLIP 嵌入空间可能是瓶颈——更强的视觉-语言对齐模型可能进一步提升
- 理论的"生物合理性"声明需谨慎——是功能启发而非神经机制模拟
- 未来方向：多 agent 场景中的内心语言交互、与 RLHF 结合、更长时间跨度任务

## 相关工作与启发

- Thought Cloning：需要逐步人类思维注释，去掉任务条件性能下降 ~40%
- Diffusion Policy (DDPM-T)：本文的行为克隆骨干
- D3IL Benchmark：专为行为多样性设计的模仿学习基准
- Autotelic AI：同样利用语言内化但聚焦自主学习而非行为多样性
- 启发：语言不仅是 agent 的输出接口，更可以作为内部认知中介来组织行为——这为 LLM agent 设计提供了新视角

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 认知科学理论的形式化 + 语言作为内部行为中介的思路高度原创
- 实验充分度: ⭐⭐⭐⭐ 多环境多指标+消融+VLM/编码器对比，但环境复杂度有限
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，架构对应关系清晰，认知科学-ML桥接出色
- 价值: ⭐⭐⭐⭐⭐ 为模仿学习引入全新范式，对人机协作和AI安全评估有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)
- [\[NeurIPS 2025\] COOPERA: Continual Open-Ended Human-Robot Assistance](coopera_continual_open_ended_human_robot_assistance.md)
- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](real-world_reinforcement_learning_of_active_perception_behaviors.md)
- [\[ICML 2025\] Action-Constrained Imitation Learning](../../ICML2025/robotics/action-constrained_imitation_learning.md)
- [\[ICML 2026\] Moving Out: Physically-grounded Human-AI Collaboration](../../ICML2026/robotics/moving_out_physically-grounded_human-ai_collaboration.md)

</div>

<!-- RELATED:END -->
