---
title: >-
  [论文解读] ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads
description: >-
  [ICCV 2025][3D视觉][视觉基础模型] 基于"VFM 层可分为低层特征提取器和高层任务适配器"的关键观察，提出 ViT-Split，通过冻结 VFM + task head（复制最后 $K_t$ 层）+ prior head（轻量 CNN 聚合多尺度先验特征）的设计，在 ADE20K 上仅用线性头即达到 58.2 mIoU（DINOv2-L），训练速度提升 4 倍，可训练参数仅为传统适配器的 1/4~1/5。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "视觉基础模型"
  - "高效适配器"
  - "DINOv2"
  - "分层特征"
  - "语义分割"
---

# ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads

**会议**: ICCV 2025  
**arXiv**: [2506.03433](https://arxiv.org/abs/2506.03433)  
**代码**: [jackyfl.github.io/vitsplit.github.io](https://jackyfl.github.io/vitsplit.github.io/)  
**领域**: 3D视觉  
**关键词**: 视觉基础模型, 高效适配器, DINOv2, 分层特征, 语义分割

## 一句话总结

基于"VFM 层可分为低层特征提取器和高层任务适配器"的关键观察，提出 ViT-Split，通过冻结 VFM + task head（复制最后 $K_t$ 层）+ prior head（轻量 CNN 聚合多尺度先验特征）的设计，在 ADE20K 上仅用线性头即达到 58.2 mIoU（DINOv2-L），训练速度提升 4 倍，可训练参数仅为传统适配器的 1/4~1/5。

## 研究背景与动机

### 问题定义

如何高效地将视觉基础模型（VFM，如 DINOv2）的预训练知识迁移到下游任务（分割、检测、深度估计、VQA 等），在保持或超越 SOTA 性能的同时大幅减少训练开销？

### 已有方法的不足

**VFM 适配器（ViT-Adapter、ViT-CoMer）的低效性**：
   - 双分支架构（CNN + ViT）导致梯度需要回传到所有层，计算和内存开销随模型增大而线性增长
   - 所有组件（VFM + CNN + 适配器 + task head）都需要微调，参数量大
   - 改变了 VFM 的先验特征，未充分利用预训练知识

**PEFT 方法的局限**：
   - VPT（prompt tuning）、AdaptFormer（adapter tuning）、LoRA（低秩微调）等仍需在每层插入可学习参数，触发早期层梯度回传
   - 没有引入低层特征（如 VFM 适配器中的 CNN 分支提供的），性能通常等于或略低于全量微调
   - 未充分利用 VFM 的预训练先验特征

**多任务部署效率低**：传统 VFM 适配器为每个任务维护独立的 VFM+CNN+适配器+头的完整管线，内存和计算冗余严重

### 核心动机

**核心观察**：通过 CKA（Centered Kernel Alignment）分析发现，DINOv2 等 VFM 的层可以明确分为两组——**前半层学习低层特征**（纹理、边缘等，跨任务相似）和**后半层学习任务相关特征**（分割关注语义、检测关注边角，跨任务差异大）。这意味着：
1. 不需要额外的 CNN 分支提供低层特征——VFM 自身前半层已具备
2. 只需微调后半层就能适配下游任务——前半层冻结即可
3. VFM 所有层的先验特征都应该被利用，而不是被微调覆盖

## 方法详解

### 整体框架

冻结整个 VFM 骨干。引入两个"分裂头"：(1) Task head：复制 VFM 最后 $K_t$ 层，接收第 $L-K_t$ 层输出，学习任务特定特征；(2) Prior head：轻量 CNN，从冻结 VFM 中均匀采样 $K_p$ 层的先验特征并聚合。最终通过 fusion net 融合两者输出，送入下游任务头。

### 关键设计

#### 1. **VFM 层分组观察与 Task Head**

- **功能**：将 VFM 最后几层复制出来作为可训练的任务特定适配器
- **核心思路**：

  **观察验证**：通过 CKA 分析和特征可视化发现：
  - 前半层（如 DINOv2-S 的 L1-L6）：跨任务（预训练/分割/检测）特征相似，关注纹理和边缘
  - 后半层（L7-L12）：不同任务特征分化——分割偏向语义信息，检测偏向物体边角

  **Task Head 设计**：将 VFM 第 $L-K_t$ 层的输出 $f_{L-K_t}$ 输入到复制的最后 $K_t$ 层：
  $$f_t = g_{\theta_t}(f_{L-K_t})$$
  然后丢弃 class token，reshape 为空间特征图 $f_t' \in \mathbb{R}^{h \times w \times D}$。

  $K_t$ 是关键超参数：控制适配器容量 vs. 训练效率的权衡。对分割任务，$K_t$ 增大的收益递减，可选较小值。

- **设计动机**：既然前半层跨任务共享，就不需要微调它们（避免梯度回传）。复制后半层作为 task head 有天然的良好初始化，同时保留了原始 VFM 的先验特征不被破坏。这也意味着大型分割头（Mask2Former、UperNet）可能是不必要的。

#### 2. **Prior Head：多尺度先验特征聚合**

- **功能**：从冻结 VFM 提取多尺度先验特征，增强任务特定特征
- **核心思路**：

  **层选择策略——均匀采样**：从 $L$ 层中均匀选择 $K_p$ 层，采样间隔 $\delta = \frac{L-b-1}{K_p-1}$：
  $$\mathcal{S} = \{b + \text{round}(i \cdot \delta) | i = 0, ..., K_p-1\}$$
  其中 $b=2$ 或 $b=3$（跳过最初几层的噪声特征）。

  **Prior Head 架构**：将采样层的特征拼接为 $f_p \in \mathbb{R}^{h \times w \times (K_p \cdot D)}$，经两层 CNN 处理：
  $$f_p' = g_{\theta_p}(f_p)$$
  CNN 包含 $1 \times 1$ 卷积（通道压缩）+ $3 \times 3$ 可变形卷积（增强低层特征 + 几何变换建模）。

- **设计动机**：均匀采样减少相邻层高度相似特征的冗余，增加多样性。VFM 的先验特征经大规模数据学习，包含丰富的通用知识——直接利用它们可增强任务特定特征并缓解 task head 的过拟合风险。可变形卷积比普通卷积更适合处理不规则的空间结构。

#### 3. **Fusion Net 与多任务适配**

- **功能**：融合任务特定特征和先验特征，适配不同下游任务
- **核心思路**：

  拼接 prior 和 task 特征图（保留更多信息）：
  $$f_o = g_{\theta_f}([f_p'; f_t'])$$
  Fusion net 与 prior head 结构相同（$1 \times 1$ + $3 \times 3$ 可变形卷积）。

  **任务特定变换**：
  - 分割：$4 \times$ 上采样（两层转置卷积）
  - 检测：生成 4 个尺度（$4\times, 2\times, 1\times, 0.5\times$）匹配 MaskRCNN 输入
  - VQA：reshape 为序列维度 $(h \cdot w) \times D$ 送入 LLM 解码器

- **设计动机**：拼接优于加法或注意力融合（消融实验验证），因为它不丢失信息。同一冻结 VFM 骨干可被多个任务共享，每个任务仅需训练自己的 task head + prior head + fusion net，大幅降低多任务部署开销。

### 损失函数 / 训练策略

- **分割**：标准交叉熵损失，AdamW（lr=2e-4, wd=1e-2），batch size 16，40k iterations
- **检测**：MaskRCNN 标准损失，AdamW（lr=1e-4, wd=5e-2），12 epochs
- Task head 学习率缩小 0.1 倍（防止初始化参数被破坏过快）
- 骨干完全冻结，无梯度回传到 VFM 前半层

## 实验关键数据

### 主实验

**ADE20K 语义分割**（512×512）：

| 方法 | Head | 可训练参数(M) | mIoU |
|------|------|-------------|------|
| ViT-Adapter-S | UperNet | 57.6 | 46.2 |
| ViT-CoMer-S | UperNet | 61.4 | 46.5 |
| DINOv2-S (Linear) | Linear | 22.1 | 49.6 |
| **ViT-Split-S** | **Linear** | **10.2** | **51.6** |
| ViT-Adapter-B | UperNet | 133.9 | 48.8 |
| DINOv2-B (UperNet) | UperNet | 120.7 | 54.8 |
| **ViT-Split-B** | **Linear** | **40.5** | **55.7** |
| ViT-Adapter-L | UperNet | 363.8 | 53.4 |
| DINOv2-L (UperNet) | UperNet | 341.2 | 57.1 |
| **ViT-Split-L** | **Linear** | **88.6** | **58.2** |

**Cityscapes 语义分割**（896×896）：

| 方法 | Head | 可训练参数(M) | mIoU(SS/MS) |
|------|------|-------------|-------------|
| ViT-Adapter-L | Mask2former | 571 | 84.9/85.8 |
| DINOv2-L | Linear | 312.9 | 83.5/84.3 |
| **ViT-Split-L** | **Linear** | **164.1** | **85.8/86.7** |

### 消融实验

**VQA 性能（LLaVA-1.5 + ViT-Split vs. 原始）**：

| 基准测试 | LLaVA-1.5 | + ViT-Split | 变化 |
|---------|-----------|------------|------|
| VQAv2 | 78.5 | 78.2 | -0.3 |
| LLaVA-Wild | 65.4 | 71.1 | **+5.7** |
| SciQA-IMG | 66.8 | 70.4 | **+3.6** |
| POPE (adv) | 84.2 | 86.1 | **+1.9** |
| MMBench | 64.3 | 66.4 | **+2.1** |

**训练效率对比**：

| 方法 | 训练时间(10k iters) | vs. ViT-Split 倍数 |
|------|-------------------|-------------------|
| ViT-Adapter-S | ~38min | ~4× 更慢 |
| ViT-CoMer-S | ~37min | ~3.9× 更慢 |
| **ViT-Split-S** | **~9min25s** | **基准** |

**融合方式消融**：

| 融合方式 | mIoU |
|---------|------|
| 加法 | 较低 |
| 拼接 | **最优** |

### 关键发现

1. **简单线性头即可超越复杂分割头**：ViT-Split-L (Linear, 88.6M params) 超过 ViT-Adapter-L (UperNet, 363.8M params) 4.8 mIoU
2. **参数效率显著**：可训练参数仅为传统适配器的 1/4~1/5，训练迭代仅需 1/4
3. **训练速度提升 4 倍**：通过避免早期层梯度回传和消除 CNN 分支
4. **VFM 先验特征的价值**：prior head 比不使用时平均提升 2% mIoU，证明冻结的多尺度先验特征确实有用
5. **多任务通用性**：在分割、检测、深度估计和 VQA 上均有效，为 ViT-Split 嵌入 LLaVA-1.5 带来 LLaVA-Wild 提升 5.7 分

## 亮点与洞察

1. **核心观察简洁有力**：VFM 层 = 特征提取器（跨任务共享）+ 任务适配器（任务特定），通过 CKA 和特征可视化双重验证
2. **"少即是多"哲学**：冻结大量参数 + 只训练少量参数反而超越全量微调，因为避免了先验知识被覆盖
3. **设计优雅**：整个方法就是"复制最后几层 + 加 CNN 聚合先验"，但效果远超复杂的多分支适配器架构
4. **多任务推理高效**：共享一个冻结 VFM，不同任务仅需各自的轻量头，GPU 内存友好
5. **对 VFM 深层理解**：揭示了 DINOv2 分割特征关注语义、检测特征关注边角的直觉性差异

## 局限与展望

1. **VFM 依赖性**：在 DINOv2 上效果最佳（层分组最清晰），其他 VFM（如 CLIP）的层分组不够明显
2. **检测任务差距**：检测 task 与 DINOv2 预训练任务差异大，ViT-Split 在 COCO 检测上仅与 ViT-CoMer 持平
3. **$K_t$ 和 $K_p$ 需手动调优**：不同任务和模型规模的最优值不同
4. **分辨率限制**：沿用 ViT 的 patch 大小（14×14 或 16×16），高分辨率输入时效率下降
5. **视频/时序扩展未探索**：当前仅支持单帧输入

## 相关工作与启发

- 与 ViT-Adapter 的关系：ViT-Adapter 用 CNN 分支提供局部特征 + 跨注意力交互，ViT-Split 证明 VFM 自身前半层已具备局部特征，无需额外 CNN
- 与 LoRA/VPT 的对比：PEFT 方法修改每一层但不利用低层特征，ViT-Split 仅修改高层但利用所有层先验
- DINOv2 的 encoder-decoder 特性：自监督预训练（重建缺失 patch）自然产生了类似 MAE 的"编码-解码"分层

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 核心观察有洞察力，设计简洁有效，但适配器思路本身不算全新
- **实验充分度**: ⭐⭐⭐⭐⭐ — 分割/检测/深度/VQA 四个任务 + 详尽消融 + 训练效率分析
- **写作质量**: ⭐⭐⭐⭐ — 观察和方法描述清晰，图示信息量大
- **价值**: ⭐⭐⭐⭐⭐ — 为 VFM 适配提供了高效范式，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DAViD: Data-efficient and Accurate Vision Models from Synthetic Data](david_data-efficient_and_accurate_vision_models_from_synthetic_data.md)
- [\[AAAI 2026\] FoundationSLAM: Unleashing the Power of Depth Foundation Models for End-to-End Dense Visual SLAM](../../AAAI2026/3d_vision/foundationslam_unleashing_the_power_of_depth_foundation_models_for.md)
- [\[CVPR 2025\] Gaussian Eigen Models for Human Heads](../../CVPR2025/3d_vision/gaussian_eigen_models_for_human_heads.md)
- [\[ECCV 2024\] Sapiens: Foundation for Human Vision Models](../../ECCV2024/3d_vision/sapiens_foundation_for_human_vision_models.md)
- [\[AAAI 2026\] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models](../../AAAI2026/3d_vision/parameter-free_fine-tuning_via_redundancy_elimination_for_vision_foundation_mode.md)

</div>

<!-- RELATED:END -->
