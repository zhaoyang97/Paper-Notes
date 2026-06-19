---
title: >-
  [论文解读] GoalLadder: Incremental Goal Discovery with Vision-Language Models
description: >-
  [NeurIPS 2025][多模态VLM][视觉语言模型] 提出 GoalLadder，利用 VLM 渐进式发现并排序候选目标状态，结合 ELO 评分系统抵抗噪声反馈，在学习的嵌入空间中定义距离奖励，仅凭单条语言指令就能训练 RL 智能体达到约 95% 的成功率。 自然语言指令（如"打开抽屉"）为 RL 任务提供了简洁的规…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "视觉语言模型"
  - "强化学习"
  - "目标发现"
  - "ELO评分"
  - "奖励函数"
---

# GoalLadder: Incremental Goal Discovery with Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.16396](https://arxiv.org/abs/2506.16396)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 视觉语言模型, 强化学习, 目标发现, ELO评分, 奖励函数

## 一句话总结

提出 GoalLadder，利用 VLM 渐进式发现并排序候选目标状态，结合 ELO 评分系统抵抗噪声反馈，在学习的嵌入空间中定义距离奖励，仅凭单条语言指令就能训练 RL 智能体达到约 95% 的成功率。

## 研究背景与动机

自然语言指令（如"打开抽屉"）为 RL 任务提供了简洁的规范方式，但从语言指令中提取有效奖励函数仍是核心挑战。现有方法存在两类问题：

**嵌入方法**（如 VLM-RM 使用 CLIP）：将任务描述和观察嵌入同一空间，用余弦相似度作奖励。但 CLIP 训练数据与目标环境不匹配，导致奖励函数噪声大

**偏好方法**（如 RL-VLM-F）：用 VLM 比较轨迹片段生成偏好标签，再训练奖励函数。虽比嵌入方法更准确，但 VLM 的错误判断会污染偏好数据集，且需大量 VLM 查询

作者指出，实用的 VLM 反馈方法必须同时解决两个关键问题：**（a）对噪声反馈的鲁棒性**和**（b）VLM 查询效率**。

## 方法详解

### 整体框架

GoalLadder 的核心思想是"渐进式目标发现"：在 RL 训练过程中，利用 VLM 逐步发现更接近任务目标的环境状态。整个流程包含四个循环阶段：

1. **收集**（Collection）：RL 智能体按当前 SAC 策略与环境交互，收集新 episode
2. **发现**（Discovery）：查询 VLM 判断新观察是否比当前最佳候选目标更优
3. **排序**（Ranking）：从缓冲区采样候选目标对，用 VLM 成对比较并更新 ELO 评分
4. **训练**（Training）：智能体以最小化到 top 候选目标的嵌入空间距离为奖励进行训练

### 关键设计

#### 1. 候选目标发现

维护候选目标缓冲区 $\mathcal{B}_g$，每个候选目标 $g_i = (o_i, e_i)$ 包含图像和 ELO 评分。从新收集的轨迹中随机采样观察 $o_j$，与当前最高评分目标 $g^*$ 比较：

- 查询 VLM：$y = \text{VLM}(o^*, o_j, l)$，$y \in \{-1, 0, 1\}$
- 若 $y=1$（新观察更优），将 $o_j$ 加入缓冲区
- 若 $y=0$ 或 $y=-1$，丢弃，保持缓冲区聚焦高质量候选

此过滤机制避免在无关状态上浪费 VLM 查询配额。

#### 2. ELO 评分系统

借鉴国际象棋 ELO 评分机制处理 VLM 噪声反馈。对候选目标对 $(g_i, g_j)$：

- 期望得分：$E_i = \frac{1}{1 + 10^{(e_j - e_i)/C}}$，$C=400$
- 评分更新：$e_i \leftarrow e_i + T(S_i - E_i)$，$T=32$

ELO 系统的优势：增量式吸收噪声比较结果，自适应调整评分，不会因单次 VLM 错误判断而严重偏离。

#### 3. 嵌入空间奖励定义

用 VAE 学习视觉特征提取器 $\psi(\cdot)$，将观察映射到紧凑潜在表示：

$$\mathcal{L} = -\mathbb{E}_{\psi(z_t|o_t)} \log p_\theta(o_t|z_t) + D_{KL}(\psi(z_t|o_t) \| p(z_t))$$

奖励定义为到最佳候选目标的欧氏距离：$R(s_{t-1}, a_{t-1}) = -d(z_t, z^*)$

实际使用时进行 max-min 归一化到 $[0,1]$，并施加非线性变换 $\hat{r} = r^{20}$，使智能体越接近目标获得越大比例的奖励。

### 损失函数 / 训练策略

- **RL 骨干**：Soft Actor-Critic（SAC），每个环境步后做一次梯度更新
- **奖励更新**：每 $L=5000$ 步更新目标状态和奖励函数，并重新标注所有存储的转移
- **VLM 反馈**：OpenAI Gym 环境每 $K=2000$ 步查询 $M=5$ 次，Metaworld 环境每 $K=500$ 步查询 $M=5$ 次
- **缓冲区管理**：目标缓冲区上限 $|\mathcal{B}_g|=10$，每 $L$ 步移除最低评分候选
- **VLM 骨干**：Gemini 2.0 Flash
- **VAE 架构**：6 层卷积编码器 + 6 层反卷积解码器，潜在维度 $|z|=16$

## 实验关键数据

### 主实验

在 2 个经典控制任务 + 5 个 Metaworld 机器人操作任务上评估：

| 方法 | 平均最终成功率 | 特点 |
|------|--------------|------|
| Oracle (真实奖励) | ~97% | 上界 |
| **GoalLadder** | **~95%** | 仅需语言指令 |
| RL-VLM-F | ~45% | 最好的竞争对手 |
| VLM-RM | ~15% | CLIP 嵌入奖励 |
| RoboCLIP | ~10% | 视频-语言相似度 |

GoalLadder 在所有 7 个任务上接近 Oracle 性能，在 Drawer Open 任务上甚至超越 Oracle。

### 消融实验

- **VLM 查询效率**：GoalLadder 平均仅需约 4500 次 VLM 查询即可解决 Metaworld 任务，而 PEBBLE（使用真实奖励偏好）需约 15000 次
- **目标发现过程**：训练过程中 top 候选目标稳步改进，缓冲区自然按任务进度排序
- **ELO 收敛**：一旦发现明显最优目标（约 50K 步），ELO 系统迅速将其推至首位

### 关键发现

1. GoalLadder 比最好的竞争对手（RL-VLM-F）高出约 50 个百分点
2. 在 Drawer Open 上超越 Oracle 说明手工设计奖励函数本身可能是次优的
3. RL-VLM-F 在简单任务上还行，但在复杂 Metaworld 任务上崩溃——说明从噪声偏好学奖励函数的困难
4. 不需要对环境做任何修改（如移除机器人），而 RL-VLM-F 需要移除以帮助 VLM 判断

## 亮点与洞察

1. **ELO + VLM 的结合非常巧妙**：用成熟的评分系统处理 VLM 固有的噪声问题，无需假设反馈准确
2. **范式创新**：从"学习奖励函数"转变为"发现目标状态+距离奖励"，绕开了从噪声标签训练奖励模型的困难
3. **极高的查询效率**：利用无标签数据训练嵌入空间，仅用少量 VLM 查询识别目标，奖励自动泛化到未见状态
4. **实验结果令人惊喜**：接近 Oracle 性能，在一个任务上甚至超越，充分验证了方法的有效性

## 局限与展望

1. 假设任务目标可由单张图像表示（静态目标），无法处理需要序列判断的动态任务
2. 依赖视觉特征相似度作为状态距离的代理，在某些环境中可能不够准确
3. 仅在模拟环境中验证，未涉及真实机器人场景
4. VLM 调用成本仍然不低（单个智能体训练约 45 小时，V100 GPU）
5. 可扩展到视频理解 VLM 以处理动态目标

## 相关工作与启发

- **与 RL-VLM-F 的本质区别**：RL-VLM-F 训练参数化奖励函数，GoalLadder 直接维护目标状态排名
- **与 PEBBLE 的对比**：PEBBLE 需要 3 倍的查询量（且用真实奖励偏好），GoalLadder 用噪声 VLM 反馈就能达到相同效果
- **启发**：ELO 评分的思路可推广到其他需要从噪声反馈中提取信号的场景，如 RLHF 中的奖励模型训练

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (ELO+VLM目标发现+嵌入空间距离奖励的组合非常新颖)
- 实验充分度: ⭐⭐⭐⭐ (7个任务全面验证，但缺少真实机器人实验)
- 写作质量: ⭐⭐⭐⭐⭐ (动机清晰，方法自然流畅，图示直观)
- 价值: ⭐⭐⭐⭐⭐ (显著优于竞争方法，实用价值高)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Unified Reinforcement and Imitation Learning for Vision-Language Models](unified_reinforcement_and_imitation_learning_for_vision-language_models.md)
- [\[CVPR 2026\] Boosting Vision-Language Models Towards Cross-Domain Incremental Object Detection](../../CVPR2026/multimodal_vlm/boosting_vision-language_models_towards_cross-domain_incremental_object_detectio.md)
- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[NeurIPS 2025\] Adapting Vision-Language Models for Evaluating World Models](adapting_visionlanguage_models_for_evaluating_world_models.md)
- [\[NeurIPS 2025\] Context Informs Pragmatic Interpretation in Vision-Language Models](context_informs_pragmatic_interpretation_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
