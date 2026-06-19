---
title: >-
  [论文解读] DreamRelation: Relation-Centric Video Customization
description: >-
  [ICCV 2025][视频生成][关系视频定制] 提出 DreamRelation，首个关系中心的视频定制方法，通过 Relation LoRA Triplet + Hybrid Mask Training 实现关系与外观的解耦，并通过时空关系对比损失增强关系动态学习，使动物能模仿人类交互。 现有视频定制方法可以个性化主体…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "关系视频定制"
  - "MM-DiT"
  - "LoRA"
  - "对比学习"
  - "解耦学习"
---

# DreamRelation: Relation-Centric Video Customization

**会议**: ICCV 2025  
**arXiv**: [2503.07602](https://arxiv.org/abs/2503.07602)  
**代码**: 有 (即将公开)  
**领域**: 视频生成 / 视频定制 (Video Generation / Video Customization)  
**关键词**: 关系视频定制, MM-DiT, LoRA, 对比学习, 解耦学习

## 一句话总结

提出 DreamRelation，首个关系中心的视频定制方法，通过 Relation LoRA Triplet + Hybrid Mask Training 实现关系与外观的解耦，并通过时空关系对比损失增强关系动态学习，使动物能模仿人类交互。

## 研究背景与动机

现有视频定制方法可以个性化主体外观和单物体运动，但无法定制两个主体间的高阶交互关系（如握手、拥抱）。关系中的核心挑战在于：

**空间复杂**：关系涉及复杂的空间布局和位置变化

**时间复杂**：关系包含微妙的时序动态

**外观纠缠**：模型往往过度关注外观细节而忽视交互本身的语义

直接使用文本描述生成反常关系（如动物握手）也行不通——通用 T2V 模型如 Mochi 无法生成此类非常规交互。更精确的关系动态（如"两人从预定位置相向走来"）也超出文本描述能力。

本文定义了**关系视频定制任务**：给定展示某种关系模式 ⟨subject, relation, subject⟩ 的示例视频，生成在新主体上表现该关系的视频。

## 方法详解

### 整体框架

DreamRelation 基于 Mochi（MM-DiT 架构）构建，包含两个并行过程：
1. **Relational Decoupling Learning**：通过 Relation LoRA Triplet 和 Hybrid Mask Training 解耦关系与外观
2. **Relational Dynamics Enhancement**：通过时空关系对比损失强化关系动态学习

视频扩散 Transformer 的训练目标为：$\mathcal{L}(\theta) = \mathbb{E}_{\bm{z}, \epsilon, \bm{c}, t}[\|\epsilon - \epsilon_\theta(\bm{z}_t, \bm{c}, t)\|_2^2]$，其中 $\bm{z}_0 \in \mathbb{R}^{f \times h \times w \times c}$ 为 3D VAE 编码的视频潜码。

### 关键设计

**1. Relation LoRA Triplet**

将关系模式分解为三元组 ⟨$S_1$, $R$, $S_2$⟩，设计复合 LoRA 集合：
- **Relation LoRAs**：注入 MM-DiT full attention 的 **Query** 和 **Key** 矩阵 → 捕获关系信息
- **Subject LoRA 1 & 2**：注入 **Value** 矩阵 → 捕获两个主体的外观信息
- **FFN LoRA**：注入 attention 线性层 → 精炼 Relation 和 Subject LoRA 的输出

注意 MM-DiT 中 text 和 vision token 由不同的 LoRA 集合处理。

**LoRA 位置选择的理论依据——Q/K/V 角色分析**：

这是本文最关键的洞察。通过可视化分析和奇异值分解（SVD）揭示 MM-DiT 中 Q、K、V 的不同角色：

- **Value 特征**：包含丰富的外观信息（如蓝色眼镜、生日帽、衣着），且关系信息（如握手区域）与外观信息纠缠
- **Query/Key 特征**：表现出高度抽象和同质的模式，跨不同视频呈现相似性，与 Value 特征截然不同
- **子空间相似性量化**：对 Q、K、V 权重矩阵做 SVD，取 top-r 左奇异向量计算归一化子空间相似度，结果显示 Q-K 高度相似而 Q/K 与 V 相似度极低

因此，Relation LoRA 放在 Q/K（信息与外观分离，有利于解耦），Subject LoRA 放在 V（与外观信息对齐）。这使 DreamRelation 成为**首个具有可解释组件的关系视频生成框架**。

**2. Hybrid Mask Training Strategy (HMT)**

使用 Grounding DINO + SAM 为示例视频中的两个主体生成分割掩码 $M_{S_1}$, $M_{S_2}$。关系掩码 $M_R$ 定义为两个主体掩码的并集（最小外接矩形，借鉴了关系检测中的经典做法）。由于 3D VAE 压缩时间维度 $T_c$ 倍，掩码在每 $T_c$ 帧内取平均。

训练时包含两个关键策略：

**LoRA 选择策略**：每次迭代随机选择只更新 Relation LoRAs 或某个 Subject LoRA。当选中 Relation LoRAs 时，两个 Subject LoRAs 同时训练提供外观线索，辅助 Relation LoRAs 专注于关系信息。FFN LoRAs 全程参与训练。

**增强扩散损失**：用对应掩码放大目标区域的损失权重：

$$\mathcal{L}_{rec} = \mathbb{E}_{\bm{z}, \epsilon, \bm{c}, t} (\lambda_m \mathbf{M}_l + 1) \cdot \|\epsilon - \epsilon_\theta(\bm{z}_t, \bm{c}, t)\|_2^2$$

其中 $l \in \{S_1, S_2, R\}$ 表示选中的掩码类型，$\lambda_m = 50$ 为掩码权重。

**推理时**：排除 Subject LoRAs（防止引入示例视频中的外观），仅使用 Relation LoRAs + FFN LoRAs，增强泛化能力。

**3. Space-Time Relational Contrastive Loss (RCL)**

显式增强关系动态学习，减少对外观细节的依赖。核心思想：关系动态体现在帧间变化中，而外观信息是单帧静态的。

- **锚点特征 $A \in \mathbb{R}^{(f-1) \times c}$**：对模型输出沿帧维度计算逐帧差分 $\bar{\epsilon}$，再空间维度取平均
- **正样本 $P$**：从展示相同关系的其他视频中采样 $n_{pos}$ 个关系动态特征
- **负样本 $N$**：从单帧模型输出中采样 $n_{neg}$ 个外观特征

对比损失基于 InfoNCE：

$$\mathcal{L}_{RCL} = \log \sum_{i=1}^{f-1} \frac{-\sum_{j=1}^{n_{pos}} \exp(A_i^\top P_{ij} / \tau)}{\sum_{j} \exp(A_i^\top P_{ij} / \tau) + \sum_{k} \exp(A_i^\top N_{ik} / \tau)}$$

使用 FIFO 记忆库（大小 64）存储并动态更新正负样本，扩大对比学习效果和训练稳定性。

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{RCL}$

训练配置：
- 基模型：Mochi，AdamW 优化器，lr=2e-4，weight decay=0.01
- 训练 2400 迭代，LoRA rank=16，$\lambda_m=50$，$\lambda_1=0.01$
- $n_{pos}=4$，$n_{neg}=10$，$\tau=0.07$
- 生成分辨率 61×480×848，batch size=1
- 推理使用 Euler Discrete，64 步，CFG scale=6.0

数据集：NTU RGB+D 中的 26 种人类交互关系，每种约 20 个视频训练，40 个设计 prompt 评估。

## 实验关键数据

### 主实验 — 关系视频定制

| 方法 | Relation Accuracy↑ | CLIP-T↑ | Temporal Consistency↑ | FVD↓ |
|------|:---:|:---:|:---:|:---:|
| Mochi (base) | 0.2623 | 0.3237 | 0.9888 | 2047 |
| Direct LoRA finetuning | 0.3258 | 0.2966 | 0.9945 | 2229 |
| ReVersion | 0.2690 | 0.3013 | 0.9921 | 2683 |
| MotionInversion | 0.3151 | 0.3217 | 0.9855 | 2085 |
| **DreamRelation** | **0.4452** | **0.3248** | **0.9954** | **2080** |

### 消融实验 — 各组件效果

| 配置 | Relation Accuracy↑ | CLIP-T↑ | FVD↓ |
|------|:---:|:---:|:---:|
| w/o HMT | 0.3574 | 0.3244 | 2249 |
| w/o RCL | 0.3416 | 0.3185 | 2137 |
| w/o Relation LoRAs | 0.3626 | 0.3035 | 2318 |
| w/o Subject LoRAs | 0.3769 | 0.3147 | 2409 |
| w/o FFN LoRAs | 0.4021 | 0.3241 | 2370 |
| **Full Model** | **0.4452** | **0.3248** | **2080** |

LoRA 位置消融：

| Relation LoRA | Subject LoRA | Relation Accuracy↑ |
|:---:|:---:|:---:|
| V | Q, K | 0.3444 |
| Q | K, V | 0.3921 |
| K, V | Q | 0.3937 |
| **Q, K** | **V** | **0.4452** |

### 关键发现

1. **Relation Accuracy 大幅领先**：0.4452 vs 次优 0.3258（Direct LoRA），提升 36.6%
2. **所有组件都不可或缺**：移除任何一个显著降低性能，HMT 和 RCL 各贡献约 0.08-0.10 的准确率
3. **Q/K/V 角色分析得到验证**：Relation LoRA 放在 Q/K 最优（0.4452），放在 V 最差（0.3444）
4. **注意力可视化**：本方法对 "shaking hands" 的注意力精确聚焦在手部交互区域
5. **用户研究一致肯定**：15 名评注者在关系对齐、文本对齐、整体质量三方面均最偏好本方法
6. **RCL 可迁移**：加到 MotionInversion 上也能提升 Relation Accuracy（0.3151→0.3633）

## 亮点与洞察

1. **开创性任务定义**：首次系统定义关系视频定制任务并提出完整解决方案
2. **可解释的架构设计**：通过 SVD 分析为 LoRA 位置选择提供理论支撑，而非盲目试错
3. **帧差分 = 关系动态**：用逐帧差分捕获时序变化、空间平均滤除外观，设计简洁优雅
4. **域迁移效果惊艳**：人类关系（握手、拥抱）可迁移到猫狗等动物上

## 局限与展望

1. Relation Accuracy 指标依赖 VLM（Qwen-VL-Max），受其能力限制
2. 训练数据仅 26 种人类交互关系，更多关系类型有待扩展
3. 目前限于双主体关系，多主体场景需要进一步设计
4. 计算开销较大（Mochi 基模型参数量大）
5. 推理时排除 Subject LoRA 的做法虽然增强泛化但不一定总是最优

## 相关工作与启发

- **MotionInversion**：单物体运动定制的代表方法，依赖时间注意力层（MM-DiT 中不存在），不适用于关系建模
- **ReVersion**：图像关系定制先驱，通过文本反转捕获关系，但静态图无法表示动态交互
- **MM-DiT (Mochi)**：视频 DiT 架构基础，本文首次深入分析其 Q/K/V 在定制任务中的功能差异
- **启发**：理解 Transformer 内部组件的功能差异是设计高效 LoRA 策略的关键前提

## 评分

| 维度 | 分数 |
|------|:---:|
| 创新性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization](dualreal_adaptive_joint_training_for_lossless_identity-motion_fusion_in_video_cu.md)
- [\[ICCV 2025\] OCK: Unsupervised Dynamic Video Prediction with Object-Centric Kinematics](ock_unsupervised_dynamic_video_prediction_with_object-centric_kinematics.md)
- [\[CVPR 2025\] TokenMotion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation](../../CVPR2025/video_generation/tokenmotion_decoupled_motion_control_via_token_disentanglement_for_human-centric.md)
- [\[CVPR 2026\] HandWorld: Hand-Centric Unified Video Action Generation](../../CVPR2026/video_generation/handworld_hand-centric_unified_video_action_generation.md)
- [\[AAAI 2026\] Mask2IV: Interaction-Centric Video Generation via Mask Trajectories](../../AAAI2026/video_generation/mask2iv_interaction-centric_video_generation_via_mask_trajectories.md)

</div>

<!-- RELATED:END -->
