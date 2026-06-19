---
title: >-
  [论文解读] Object-Centric Latent Action Learning
description: >-
  [AAAI 2026 Oral][机器人][以物体为中心的表示] 提出以物体为中心的潜在动作学习框架，利用自监督的物体分解（VideoSAUR）将场景中任务相关实体与视觉干扰（动态背景等）分离，使潜在动作模型（LAPO）在有干扰的视频中性能退化减少约50%，并通过线性动作探针自动选择控制相关的 slot。
tags:
  - "AAAI 2026 Oral"
  - "机器人"
  - "以物体为中心的表示"
  - "潜在动作学习"
  - "视觉干扰"
  - "模仿学习"
  - "自监督学习"
---

# Object-Centric Latent Action Learning

**会议**: AAAI 2026 Oral  
**arXiv**: [2502.09680](https://arxiv.org/abs/2502.09680)  
**代码**: [https://github.com/dunnolab/object-centric-lapo](https://github.com/dunnolab/object-centric-lapo)  
**领域**: 强化学习  
**关键词**: 以物体为中心的表示, 潜在动作学习, 视觉干扰, 模仿学习, 自监督学习

## 一句话总结

提出以物体为中心的潜在动作学习框架，利用自监督的物体分解（VideoSAUR）将场景中任务相关实体与视觉干扰（动态背景等）分离，使潜在动作模型（LAPO）在有干扰的视频中性能退化减少约50%，并通过线性动作探针自动选择控制相关的 slot。

## 研究背景与动机

具身 AI 领域在通用能力上落后于 NLP 和计算机视觉，主要瓶颈在于缺乏高质量的预训练数据。互联网上海量的视频数据（覆盖各种人类活动）是潜在的训练数据来源，但存在两大障碍：

**缺少动作标签**：互联网视频不包含显式的动作标注，无法直接用于模仿学习或强化学习。潜在动作模型（LAM）通过从观察序列推断潜在动作来解决这个问题，但引入了新问题。

**动作相关的视觉干扰**：真实世界视频本身包含与动作相关的干扰——如动态背景、相机抖动、颜色变化等。这些干扰与智能体动作产生虚假的相关性，导致 LAM 过拟合到非因果模式上。现有方法（如 LAPA）通常假设数据集是干净的或依赖昂贵的标注，严重限制了可扩展性。

**现有 LAM 的脆弱性**：Nikulin et al. (2025) 经验性地证明了 LAPO 方法在包含干扰的数据上会显著性能退化，并建议在 LAM 预训练中重用已有的动作标签提供监督。但在某些领域（如 YouTube 视频），动作标签根本不存在。

**核心假设**：以物体为中心的表示提供了必要的结构先验，可以将因果性的智能体-物体交互与非因果的视觉关联分离。在 slot 特征（而非原始像素）上操作，使 LAM 能够聚焦于相关物体的动态，同时过滤掉背景运动等干扰。

## 方法详解

### 整体框架

方法流水线分为三个阶段：
1. **以物体为中心的预训练**：用 VideoSAUR 将视频帧分解为可解释的物体 slot
2. **潜在动作学习**：在 slot 空间训练基于 LAPO 的逆/正向动力学模型
3. **行为克隆与微调**：用推断的潜在动作训练 BC 策略，然后用少量真实动作标签微调（≤2.5% 数据量）

### 关键设计

1. **以物体为中心的表示学习（VideoSAUR）**：采用 VideoSAUR 模型将输入视频帧分解为时空物体 slot。VideoSAUR 的自监督架构能隔离场景中的个体实体，提供对背景噪声和偶然运动不敏感的结构化表示。对每个观测 $o_t$，编码器产生 $K$ 个 slot 向量 $s_t^{(k)} \in \mathbb{R}^d$。由于其 Transformer 解码器，每个 slot 可通过注意力图作为 alpha mask 投影回原始图像空间，生成物体掩码 $m_t^{(k)}$ 用于可视化。

   关键实现细节：采用**固定初始化**的 slot，确保相同的 slot 索引在不同 episode 间对应相似的语义物体，减少 slot 排列问题。作者也尝试了 STEVE 模型，但发现其无法可靠地隔离 hopper 等实体。

2. **通过线性动作探针进行 Slot 选择**：以物体为中心的模型虽然能分解场景，但自动识别哪些 slot 对应任务相关实体仍是挑战。方法采用线性探测（inspired by Alain & Bengio 2016）：
   
    - 对少量带标签轨迹的 slot 编码进行 PCA 降维
    - 训练线性回归器预测真实动作
    - 用 5 折交叉验证评估平均测试 MSE（Linear Action Probe score）
    - 选择 MSE 最低（即最能预测动作）的 slot 作为相关 slot 集合 $\mathcal{S}^\star = \{s^{(k)} | k \in \mathcal{K}^\star\}$
   
   此选择在物体中心预训练后仅执行一次，利用固定 slot 初始化确保跨 episode 的一致解释。

3. **潜在动作建模的两种变体**：

    - **LAPO-slots**：完全在潜在空间操作。逆动力学模型 $z_t \sim f_{IDM}^s(\cdot | s_t, s_{t+1})$ 和正向动力学模型 $\hat{s}_{t+1} \sim f_{FDM}^s(\cdot | s_t, z_t)$ 在 slot 嵌入空间训练，最小化 $\|\hat{s}_{t+1} - s_{t+1}\|^2$。
   
    - **LAPO-masks**：在像素空间操作。先将选定 slot 的物体掩码应用于输入帧，创建过滤后的图像（只保留任务相关物体），然后在这些过滤图像上训练动力学模型。

### 损失函数 / 训练策略

- VideoSAUR 预训练：自监督，不需要动作标签
- 潜在动作模型训练损失：$\mathcal{L}_{MSE} = \mathbb{E}_t[\|f_{FDM}(f_{IDM}(o_t, o_{t+1}), o_t) - o_{t+1}\|^2]$（对 slots 或 masks 变体有相应调整）
- BC 训练：在推断的潜在动作上进行监督学习
- 微调：使用极少量真实动作标签（0.1%-2.5%）

## 实验关键数据

### 主实验

在 Distracting Control Suite（DCS）和 Distracting MetaWorld（DMW）共8个任务上评估，性能归一化为用全部真实动作标签训练的 BC 智能体。

| 环境 | 任务 | LAPO（基线） | LAPO-clean（上界） | LAPO-masks | LAPO-slots | 恢复比% |
|------|------|-------------|------------------|------------|------------|---------|
| DCS-Hard | cheetah-run | 0.24±0.02 | 0.76±0.04 | 0.41±0.03 (+32%) | **0.55±0.04** (+58%) | 58% |
| DCS-Hard | hopper-hop | 0.03±0.01 | 0.27±0.03 | 0.08±0.01 (+20%) | **0.15±0.02** (+50%) | 50% |
| DCS-Hard | humanoid-walk | 0.02±0.01 | 0.06±0.01 | 0.04±0.02 (+47%) | **0.06±0.01** (+105%) | 105% |
| DCS-Hard | 平均 | 0.08±0.01 | 0.35±0.04 | 0.15±0.02 (+26%) | **0.22±0.02** (+52%) | 52% |
| DCS | 平均 | 0.13±0.02 | 0.35±0.04 | **0.24±0.04** (+50%) | 0.24±0.03 (+50%) | 50% |
| DMW | hammer | 0.75±0.07 | 0.98±0.01 | 0.96±0.01 (+91%) | **0.99±0.02** (+102%) | 102% |
| DMW | bin-picking | 0.18±0.08 | 0.74±0.10 | **0.49±0.10** (+56%) | 0.33±0.08 (+27%) | 56% |
| DMW | 平均 | 0.31±0.06 | 0.65±0.06 | 0.50±0.07 (+55%) | 0.48±0.06 (+50%) | 50% |

定义：恢复比 = $\frac{\text{LAPO-slots/masks} - \text{LAPO}}{\text{LAPO-clean} - \text{LAPO}} \times 100\%$，衡量方法恢复了干扰造成的多少性能差距。

### 消融实验

| 配置 | 关键发现 | 详细说明 |
|------|----------|----------|
| Slot 数量 K=2-15 | 对 K 鲁棒 | 所有 K 值下 LAPO-slots 均优于 LAPO 基线 |
| K=2 | 表示坍缩 | 太少的 slot 导致多个实体合并为一个 |
| K≤8 vs K>8 | 单 slot vs 双 slot | K≤8 时选1个 slot 最佳；K>8 时拼接 top-2 效果更好 |
| DCS vs DCS-Hard | 鲁棒性差异 | LAPO-slots 在更强干扰下保持一致改进（50%→52%），LAPO-masks 退化（50%→26%） |
| 探针准确度 vs BC 性能 | 强正相关 | Linear Action Probe 分数与下游 BC 成功率高度相关 |

### 关键发现

1. **50%性能恢复**：在 DCS 和 DMW 上，物体中心表示平均恢复了基线（有干扰）与上界（无干扰）之间 50% 的性能差距
2. **LAPO-slots 在强干扰下更鲁棒**：相比 LAPO-masks，因为其使用 DINOv2 编码器捕获高层语义特征，对外观变化更鲁棒
3. **功能性 vs 几何性分解**：在 DMW 中，VideoSAUR 将行为耦合的实体（如机械臂和目标物体）合并到一个 slot，而非按几何形状分离——这反映了物体中心模型倾向于功能性分解
4. **极少量标签即可有效**：仅用 0.1%-2.5% 的真实动作标签进行微调即可获得强性能
5. **humanoid-walk 上改进最大**：+174%（DCS）和 +105%（DCS-Hard），说明在最复杂任务上物体中心表示的优势更显著

## 亮点与洞察

- **物体中心表示作为视觉干扰的天然防御**：这是本文最核心的贡献——通过将场景分解为离散 slot，本质上提供了一种无监督的因果/非因果信号分离机制
- **线性动作探针的简洁实用**：不需要复杂的注意力机制或学习过程，简单的线性回归 + PCA 就能可靠地选择控制相关 slot，且与下游性能高度相关
- **两种变体的互补性**：LAPO-slots 更鲁棒（利用 DINOv2 的语义特征），LAPO-masks 在某些任务上更强（保留空间细节）
- **对 K 的鲁棒性**：实际应用中无需精确知道场景中的物体数量

## 局限与展望

1. **依赖 OCL 模型的质量**：物体中心模型的分解质量直接决定方法效果。当前 OCL 模型在严重遮挡、多视角场景中仍面临挑战
2. **固定 slot 数量 K**：大多数 OCL 框架需要预先指定 K，限制了对不同复杂度场景的灵活性
3. **缺乏记忆机制**：VideoSAUR 无法处理物体进出场景的情况
4. **数据多样性不足时的坍缩分解**：在 DMW 中观察到行为耦合的实体被合并，源于轨迹多样性有限
5. **仅在模拟环境验证**：DCS 和 DMW 都是模拟器，向真实世界视频（如 YouTube）的推广有待验证

## 相关工作与启发

- **LAPO**（Schmidt & Jiang 2024）：本文的直接基线——从视觉观测序列推断潜在动作，在干净数据上有效但在干扰下失败
- **VideoSAUR**（Zadaianchuk et al. 2023）：本文使用的物体中心模型——自监督视频分解，基于 DINOv2 编码器
- **Nikulin et al. (2025)**：与本文正交的方法——提出无重建框架，需要少量标签轨迹在预训练时提供监督
- **Genie**（Bruce et al. 2024）：生成式交互环境，大规模 VLA 模型中也使用 LAM
- **LAPA**（Ye et al. 2024）：潜在动作预训练用于 VLA 模型，但假设干净数据集

## 评分

- 新颖性: ⭐⭐⭐⭐ （物体中心表示 + 潜在动作学习的结合新颖且直觉上合理）
- 实验充分度: ⭐⭐⭐⭐ （8个任务，多维度分析，但仅限模拟环境）
- 写作质量: ⭐⭐⭐⭐⭐ （结构清晰，图表直观，slot 选择分析深入）
- 价值: ⭐⭐⭐⭐ （为从真实世界视频中学习具身策略提供了重要的鲁棒性改进思路）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Learning a Unified Latent Action Space from Videos with Action-centric Cycle Consistency](../../CVPR2026/robotics/learning_a_unified_latent_action_space_from_videos_with_action-centric_cycle_con.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](../../NeurIPS2025/robotics/learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)
- [\[ICML 2026\] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](../../ICML2026/robotics/latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)
- [\[CVPR 2026\] Motus: A Unified Latent Action World Model](../../CVPR2026/robotics/motus_a_unified_latent_action_world_model.md)

</div>

<!-- RELATED:END -->
