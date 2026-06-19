---
title: >-
  [论文解读] Temporally Consistent Object-Centric Learning by Contrasting Slots
description: >-
  [CVPR 2025][视频理解][物体中心表示] Slot Contrast 提出了一种新颖的对象级时序对比损失，通过在批次内跨视频对比 slot 表示，显著提升了视频物体中心模型的时序一致性，在合成和真实世界数据集上的物体发现任务中超越了甚至使用运动掩码的弱监督方法，并有效支持了下游的无监督物体动态预测。
tags:
  - "CVPR 2025"
  - "视频理解"
  - "物体中心表示"
  - "时序一致性"
  - "对比学习"
  - "注意力机制"
  - "无监督物体发现"
---

# Temporally Consistent Object-Centric Learning by Contrasting Slots

**会议**: CVPR 2025  
**arXiv**: [2412.14295](https://arxiv.org/abs/2412.14295)  
**代码**: [https://slotcontrast.github.io/](https://slotcontrast.github.io/)  
**领域**: 视频理解 / 物体中心学习  
**关键词**: 物体中心表示, 时序一致性, 对比学习, Slot Attention, 无监督物体发现

## 一句话总结

Slot Contrast 提出了一种新颖的对象级时序对比损失，通过在批次内跨视频对比 slot 表示，显著提升了视频物体中心模型的时序一致性，在合成和真实世界数据集上的物体发现任务中超越了甚至使用运动掩码的弱监督方法，并有效支持了下游的无监督物体动态预测。

## 研究背景与动机

**领域现状**：物体中心学习（Object-Centric Learning, OCL）是一种将高维视觉数据分解为独立物体表示（通常称为 slots）的无监督学习范式。基于视频的 OCL 方法（如 SAVi、STEVE、VideoSAUR）通过将前一帧的 slot 初始化当前帧的 slot，建立帧间物体对应关系。近期方法借助自监督预训练特征（如 DINOv2）和多样化训练数据（如 YouTube-VIS），已能扩展到真实世界视频。

**现有痛点**：现有方法虽然能在短视频中分解场景，但在长时序上维持一致的物体表示仍然困难。核心问题在于：训练目标（通常是特征重建）并没有显式地鼓励时序一致性。当发生物体遮挡、重新出现、复杂交互时，slot 容易在不同帧中"跳转"到不同物体上（即同一个 slot 在不同帧中表示不同的物体）。

**核心矛盾**：特征重建损失只要求 slot "覆盖" 当前帧的所有内容即可，不关心哪个 slot 覆盖哪个物体，因此无法保证 slot-物体的对应关系跨时间保持稳定。而 slot 的随机初始化策略（从同一高斯分布采样）更加剧了这个问题。

**本文目标**：设计一种显式的时序一致性约束，使每个 slot 在整个视频序列中稳定地跟踪同一个物体，同时不牺牲甚至提升物体发现的能力。

**切入角度**：作者观察到，对比学习天然适合定义"什么应该相似、什么应该不同"的结构——如果将同一 slot 在相邻帧的表示定义为正例对，其余 slot 定义为负例对，InfoNCE 损失就能推动 slot 学习时序一致的表示。

**核心 idea**：提出 slot-slot 对比损失，将对比的负例集从单个视频内扩展到整个 batch 的所有视频，配合可学习的 slot 初始化，构成 Slot Contrast 架构，实现强时序一致性和优秀的物体发现性能。

## 方法详解

### 整体框架

Slot Contrast 基于编码器-解码器的物体中心架构。编码器使用冻结的 DINOv2 ViT 提取 patch 特征，经过可学习的 MLP 适配后送入循环 Slot Attention 模块进行分组。模型处理视频的每一帧时，使用前一帧预测的 slot 初始化当前帧的 slot。训练通过两个损失联合优化：特征重建损失（确保 slot 信息量充足）和 slot-slot 对比损失（确保时序一致性）。

### 关键设计

1. **批次级 Slot-Slot 对比损失（Batch Video Slot-Slot Contrastive Loss）**:

    - 功能：显式强制每个 slot 在时间上保持一致，同时与其他 slot（包括同视频和不同视频中的）保持区分性
    - 核心思路：给定相邻两帧的 slot 集合 $S_{t-1}$ 和 $S_t$，将第 $j$ 个视频中第 $i$ 个 slot 在 $t-1$ 帧的表示 $s_{t-1}^{i,j}$ 与 $t$ 帧的同位置 slot $s_t^{i,j}$ 作为正例对，batch 中所有其他 slot $s_t^{k,b}$（$k \neq i$ 或 $b \neq j$）作为负例。采用 InfoNCE 损失：$\ell_{i,j}^{\text{ssc}} = -\log \frac{\exp(\text{sim}(s_{t-1}^{i,j}, s_t^{i,j}) / \tau)}{\sum_{b,k} \mathbb{1}_{[k,b \neq i,j]} \exp(\text{sim}(s_{t-1}^{i,j}, s_t^{k,b}) / \tau)}$。
    - 设计动机：仅在单个视频内做对比（intra-video）的问题是模型可能通过放大 slot 初始化之间的差异来"作弊"，而不是真正学习物体的区分性特征。扩展到整个 batch 增大了对比集的规模和多样性，因为同一 batch 中所有视频共享相同的初始 slot $S_0$，所以无法依赖初始化差异来区分，被迫基于视觉内容学习区分。实验验证了 batch 级对比比 intra-video 对比提升了 +14.5 FG-ARI（MOVi-C 上）。

2. **可学习的 Slot 初始化（Learned Initialization）**:

    - 功能：为 slot 提供良好的初始表示结构，促进对比学习和物体发现
    - 核心思路：抛弃原始 Slot Attention 中从同一高斯分布随机采样 slot 的初始化方式，改为为整个数据集学习一组固定的初始 slot 向量 $S_0$。这些 slot 在训练中自然学习到互不相同的初始查询，能稳定地关注不同类型的物体。
    - 设计动机：随机初始化对 slot 的对比学习不利——从同一分布采样的 slot 初始值相似，难以在 slot 空间中建立良好的结构。可学习初始化让每个 slot 有独特的"偏好"，配合对比损失形成协同效应。实验中，可学习初始化在 MOVi-E 上带来了 +7.6 FG-ARI 的提升（从 75.3 到 82.9）。

3. **语义循环 Slot Attention 模块（Semantic Recurrent Slot Attention）**:

    - 功能：在 DINOv2 语义特征空间中进行时序物体分组
    - 核心思路：冻结的 DINOv2 ViT 提取 patch 特征 $g_t$，通过可学习的 MLP $g_\psi$ 适配特征 $h_t = g_\psi(g_t)$，再送入循环 Slot Attention。模块包含分组组件 $C_\theta$（标准 Slot Attention 更新 slot）和预测器 $P_\omega$（捕获时空 slot 交互），分别输出 $S_t^c$ 和 $S_t^p$，其中分组输出用于解码，预测输出传递到下一帧。
    - 设计动机：DINOv2 特征虽然语义丰富但主要为图像级训练，通过可学习 MLP 适配可以使特征更适合时序物体分组任务。循环结构允许 slot 在帧间传递信息。

### 损失函数 / 训练策略

总损失为特征重建损失和对比损失的加权和：$\mathcal{L} = \sum_{t=1}^{T-1} \mathcal{L}_{\text{rec}}(h_t, \hat{h}_t) + \alpha \mathcal{L}_{\text{ssc}}(S_{t-1}, S_t)$。解码器使用 MLP 从 slot 重建 DINOv2 特征。DINOv2 编码器冻结不训练。温度参数 $\tau$ 控制对比损失的锐度。MOVi 数据集使用分辨率 336×336，YouTube-VIS 使用 518×518。

## 实验关键数据

### 主实验

时序一致的物体发现（Video FG-ARI / mBO，在整个视频序列上计算）：

| 方法 | MOVi-C FG-ARI↑ | MOVi-C mBO↑ | MOVi-E FG-ARI↑ | MOVi-E mBO↑ | YTVIS FG-ARI↑ | YTVIS mBO↑ |
|------|----------------|-------------|----------------|-------------|---------------|------------|
| SAVi | 22.2 | 13.6 | 42.8 | 16.0 | - | - |
| STEVE | 36.1 | 26.5 | 50.6 | 26.6 | 15.0 | 19.1 |
| VideoSAUR | 64.8 | 38.9 | 73.9 | 35.6 | 28.9 | 26.3 |
| VideoSAURv2 | - | - | 77.1 | 34.4 | 31.2 | 29.7 |
| **Slot Contrast** | **69.3** | 32.7 | **82.9** | 29.2 | **38.0** | **33.7** |

单帧物体发现（Image FG-ARI，MOVi-E）：

| 方法 | 监督类型 | Image FG-ARI↑ |
|------|---------|---------------|
| DINOSAUR | 仅图像 | 65.1 |
| DIOD | +运动分割掩码 | 82.2 |
| SOLV | 仅视频 | 80.8 |
| VideoSAUR | 仅视频 | 78.4 |
| **Slot Contrast** | **仅视频** | **84.8** |

### 消融实验

损失组件消融（MOVi-C / MOVi-E / YouTube-VIS）：

| 特征重建 | Intra对比 | Batch对比 | MOVi-C FG-ARI | MOVi-E FG-ARI | YTVIS FG-ARI |
|---------|----------|----------|--------------|--------------|-------------|
| ✓ | | | 49.7 | 79.8 | 35.3 |
| ✓ | ✓ | | 54.8 | 78.7 | 35.7 |
| ✓ | | ✓ | **69.3** | **82.9** | **38.0** |

初始化策略消融：

| 配置 | MOVi-C FG-ARI | MOVi-E FG-ARI | YTVIS FG-ARI |
|------|--------------|--------------|-------------|
| 特征重建+随机初始化 | 45.3 | 71.1 | 35.2 |
| 特征重建+学习初始化 | 49.4 | 79.8 | 35.3 |
| Slot Contrast+随机初始化 | 62.9 | 75.3 | 36.1 |
| **Slot Contrast+学习初始化** | **69.3** | **82.9** | **38.0** |

### 关键发现

- Batch 级对比远优于 Intra-video 对比：MOVi-C 上 FG-ARI 从 54.8 跃升到 69.3，因为大对比集防止了依赖初始化差异的退化解法
- 对比损失不仅提升了时序一致性，还显著提升了单帧物体发现能力（84.8 Image FG-ARI 超越了使用运动掩码的弱监督方法 DIOD 的 82.2），说明时序约束迫使特征学习更具区分性的物体表示
- 在完全遮挡的场景中，Slot Contrast 的 mBO 从基线的 16% 提升到 21%，说明对比损失帮助 slot 在物体重新出现后恢复对应关系
- 可学习初始化和对比损失有显著的协同效应：两者单独使用各有提升，但组合使用的提升远大于两者之和
- 在 YouTube-VIS 真实数据集上的优势最为明显（FG-ARI +6.8, mBO +4.0 vs VideoSAURv2），证明方法在复杂真实场景中的价值

## 亮点与洞察

- **对比学习促进物体发现的"意外"发现**：时序一致性损失作为副产品带来了更好的单帧物体分割能力，超越了使用额外运动监督的方法。这揭示了时序信号比运动分割掩码更有效的 inductive bias
- **batch 对比防退化的洞察**：因为同一 batch 共享相同的 slot 初始化 $S_0$，模型无法通过差异化初始化来"走捷径"，被迫真正学习物体内容。这是一个关于对比学习设计中避免 shortcut solution 的优秀案例
- **slot 初始化与对比损失的协同**：可学习初始化提供了良好的结构先验，对比损失在此基础上进一步强化差异性，两者配合的效果 > 简单相加

## 局限与展望

- 在合成数据集上 mBO 指标低于 VideoSAUR（如 MOVi-C 32.7 vs 38.9），说明对比约束可能导致 slot 的空间掩码不够锐利，有时一个 slot 会覆盖多个物体的部分
- slot 数量 K 需要预先设定且对所有视频相同，无法自适应地匹配场景中的实际物体数量
- YouTube-VIS 视频较短（最多 76 帧），方法在真正的长视频（数百帧）上的行为尚未验证
- 完全无监督设置意味着无法保证 slot 与语义物体类别的对应，同一 slot 可能在不同视频中表示不同类别的物体
- 下游物体动态预测任务中在 MOVi-E 上没有显著提升，说明对于有相机运动的场景，纯外观一致性不足以捕获完整的动态

## 相关工作与启发

- **vs VideoSAUR**: VideoSAUR 通过预测 DINOv2 特征的时序相似度来隐式建模时序信息；Slot Contrast 通过显式的 slot 级对比损失更直接地强制时序一致性，且扩展到 batch 级别提供了更强的约束
- **vs SOLV**: SOLV 通过聚合聚类和预测中间帧特征来获取时序一致性；Slot Contrast 的对比方法更简洁且效果更好（84.8 vs 80.8 Image FG-ARI）
- **vs SAVi/SAVi++**: SAVi 系列使用循环处理但缺乏显式一致性约束；SAVi++ 引入深度作为额外监督。Slot Contrast 不使用任何标注就超越了这些方法

## 评分

- 新颖性: ⭐⭐⭐⭐ slot 级对比损失的想法自然但有效，batch 扩展负例集合的设计巧妙解决了退化问题
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多个下游任务、详尽的消融实验、与弱监督方法的比较
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构清晰、损失函数的推导层层递进、实验分析深入
- 价值: ⭐⭐⭐⭐ 为视频物体中心学习提供了简单有效的时序一致性方案，同时揭示了时序对比损失促进物体发现的重要洞察

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning](../../ICLR2026/video_understanding/from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv.md)
- [\[CVPR 2025\] H-MoRe: Learning Human-centric Motion Representation for Action Analysis](h-more_learning_human-centric_motion_representation_for_action_analysis.md)
- [\[CVPR 2026\] Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning](../../CVPR2026/video_understanding/reconstruction-guided_slot_curriculum_addressing_object_over-fragmentation_in_vi.md)
- [\[ICCV 2025\] Factorized Learning for Temporally Grounded Video-Language Models](../../ICCV2025/video_understanding/factorized_learning_for_temporally_grounded_video-language_models.md)
- [\[CVPR 2025\] EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](edcflow_exploring_temporally_dense_difference_maps_for_event-based_optical_flow_.md)

</div>

<!-- RELATED:END -->
