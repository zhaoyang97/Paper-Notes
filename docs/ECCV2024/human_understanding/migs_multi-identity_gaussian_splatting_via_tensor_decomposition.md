---
title: >-
  [论文解读] MIGS: Multi-Identity Gaussian Splatting via Tensor Decomposition
description: >-
  [ECCV 2024][人体理解][3D Gaussian Splatting] 提出MIGS，通过CP张量分解将多个人体身份的3DGS参数统一到一个低秩张量中，在大幅减少参数量的同时实现了对未见姿态的鲁棒动画。 领域现状： 3D Gaussian Splatting (3DGS) 已成功应用于人体avatar建模…
tags:
  - "ECCV 2024"
  - "人体理解"
  - "3D Gaussian Splatting"
  - "多身份表示"
  - "张量分解"
  - "人体动画"
  - "单目视频"
---

# MIGS: Multi-Identity Gaussian Splatting via Tensor Decomposition

**会议**: ECCV 2024  
**arXiv**: [2407.07284](https://arxiv.org/abs/2407.07284)  
**代码**: [项目页面](https://aggelinacha.github.io/MIGS/)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 多身份表示, 张量分解, 人体动画, 单目视频

## 一句话总结

提出MIGS，通过CP张量分解将多个人体身份的3DGS参数统一到一个低秩张量中，在大幅减少参数量的同时实现了对未见姿态的鲁棒动画。

## 研究背景与动机

**领域现状**: 3D Gaussian Splatting (3DGS) 已成功应用于人体avatar建模，能实现实时渲染且视觉质量高。现有方法如3DGS-Avatar、GauHuman等将3DGS与SMPL人体先验结合，从单目视频学习可动画的人体表示。

**现有痛点**: 当前所有3DGS人体方法都是**单身份（per-identity）优化**——每个人需要独立训练一个模型。这导致：(a) 多人场景时参数量线性增长，$N_i$个人需要$N_i \times N_g \times M$个参数；(b) 单个身份的训练数据有限，面对分布外（OOD）姿态时动画质量急剧下降。

**核心矛盾**: 单身份模型只能学到有限的人体变形模式，但真实应用需要在极端姿态（如高难度舞蹈）下保持动画鲁棒性。多身份联合学习可以共享变形知识，但如何在不爆炸参数量的前提下实现？

**本文目标**: 从单目视频学习**多身份统一的3DGS表示**，既能压缩参数，又能通过跨身份知识共享提升OOD姿态下的动画鲁棒性。

**切入角度**: 借鉴经典TensorFaces的思路，将所有身份的Gaussian参数组织成高阶张量，利用CP分解实现低秩近似。

**核心 idea**: 不同人体共享相似的结构特征，因此多身份Gaussian参数张量具有低秩结构，可以用CP分解高效表示。

## 方法详解

### 整体框架

MIGS的管线分为三步：(1) 为每个身份$i$定义canonical空间下的3D Gaussians，参数包括位置$\boldsymbol{\mu}$、缩放$\boldsymbol{s}$、旋转四元数$\boldsymbol{q}$、特征向量$\boldsymbol{f}$和不透明度$\alpha$；(2) 将所有身份的Gaussian参数堆叠成三阶张量$\boldsymbol{\mathcal{W}} \in \mathbb{R}^{N_i \times N_g \times M}$；(3) 对该张量进行CP分解，只学习分解后的因子矩阵。动画时使用非刚性变形网络$f_d$和基于LBS的刚性变换将canonical Gaussians变换到观察空间。

### 关键设计

1. **高阶张量构建**: 对于$N_i$个身份，每个有$N_g$个Gaussians，每个Gaussian有$M=43$维参数（3位置+3缩放+4旋转+32特征+1不透明度），构建张量：

$$\boldsymbol{\mathcal{W}} \in \mathbb{R}^{N_i \times N_g \times M}, \quad \boldsymbol{w}_{i,g,:} = [\boldsymbol{\mu}^{(i,g)}; \boldsymbol{s}^{(i,g)}; \boldsymbol{q}^{(i,g)}; \boldsymbol{f}^{(i,g)}; \alpha^{(i,g)}]$$

这样能自然地将"身份"、"Gaussian索引"、"参数类型"三个维度解耦。

2. **CP张量分解**: 对张量$\boldsymbol{\mathcal{W}}$进行CANDECOMP/PARAFAC分解。先沿第二维展开得到$\boldsymbol{W}_{(2)} \in \mathbb{R}^{N_g \times (N_i M)}$，然后近似：

$$\boldsymbol{W}_{(2)} \approx \boldsymbol{U}_3 (\boldsymbol{U}_2 \odot \boldsymbol{U}_1)^T$$

其中$\boldsymbol{U}_1 \in \mathbb{R}^{M \times R}$、$\boldsymbol{U}_2 \in \mathbb{R}^{N_i \times R}$、$\boldsymbol{U}_3 \in \mathbb{R}^{N_g \times R}$，$\odot$为Khatri-Rao积。实际只需学习$(M + N_i + N_g)R$个参数，而非$M \cdot N_i \cdot N_g$个。当$R=100, N_g=5 \times 10^4, N_i=30$时，参数量从$6.5 \times 10^7$降至$5 \times 10^6$，**降低一个数量级**。

3. **非刚性-刚性变形**: 非刚性变形网络$f_d$输出位置/缩放/旋转的偏移量：$(\delta\boldsymbol{\mu}, \delta\boldsymbol{s}, \delta\boldsymbol{q}, \boldsymbol{z}) = f_d(\boldsymbol{\mu}_c; \boldsymbol{z}_p)$。刚性变换基于SMPL的LBS：$\boldsymbol{T} = \sum_{b=1}^{B} f_r(\boldsymbol{\mu}_d)_b \boldsymbol{B}_b$。颜色通过MLP $f_c$ 从特征向量和球谐基预测。

4. **初始化策略**: 用第一个身份的SMPL mesh采样初始化$N_g$个点，对其参数矩阵用TensorLy的CPPower算法计算CP分解，得到$\boldsymbol{U}_1, \boldsymbol{U}_3$和$\boldsymbol{U}_2$的第一行，然后将$\boldsymbol{U}_2$第一行复制到所有行。

5. **个性化与新身份**: (a) 个性化：冻结其他参数，仅微调颜色MLP $f_c$来恢复高频细节；(b) 新身份：在$\boldsymbol{U}_2$中添加新行，仅优化新行和$f_c$，不破坏已学到的多身份变形知识。

### 损失函数 / 训练策略

沿用3DGS-Avatar的损失函数：RGB光度损失 + mask损失 + 蒙皮权重正则化 + as-isometric-as-possible正则化。训练时交替从不同身份采样帧进行渲染优化。特别地，**不使用per-frame latent code**以避免对训练帧的过拟合。

## 实验关键数据

### 主实验

**ZJU-MoCap新视角合成 (6个身份训练)**:

| 方法 | 377 PSNR↑ | 386 PSNR↑ | 392 PSNR↑ | 394 PSNR↑ |
|------|-----------|-----------|-----------|-----------|
| HumanNeRF | 30.41 | 33.20 | 31.04 | 30.31 |
| 3DGS-Avatar | 30.64 | 33.63 | 31.66 | 30.54 |
| **MIGS (Ours)** | **32.85** | **34.98** | **33.88** | **32.28** |

**AIST++ 舞蹈数据集 (30个身份训练)**:

| 方法 | Basic PSNR↑ | Basic LPIPS*↓ | Advanced PSNR↑ | Advanced LPIPS*↓ |
|------|-------------|---------------|----------------|------------------|
| HumanNeRF | 24.58 | 29.20 | 22.01 | 39.01 |
| 3DGS-Avatar | 28.89 | 18.20 | 25.51 | 28.86 |
| **MIGS** | **29.82** | **17.73** | **26.54** | **26.02** |

### 消融实验

**CP分解秩R的影响 (AIST++ Advanced Test, LPIPS*↓)**:

| 身份数 | R=10 | R=100 | R=200 |
|--------|------|-------|-------|
| 10 | ~28 | ~26 | ~26 |
| 20 | ~32 | ~27 | ~27 |
| 30 | ~38 | ~28 | ~27 |

R=10不足以捕获多身份的多样性，R=100已经足够，R=200无显著提升。个性化微调后R=100和R=200效果几乎一致。

### 关键发现

- 增加训练身份数量→提升OOD姿态鲁棒性（LPIPS降低），但结果变平滑→个性化微调可以恢复细节
- 在AIST++的高难度舞蹈姿态上，MIGS显著优于所有单身份方法，尤其在四肢交叉等极端姿态下
- 新身份学习仅需10秒短视频 + 优化$\boldsymbol{U}_2$的新行即可

## 亮点与洞察

- **低秩假设的物理直觉**: 不同人体共享骨骼结构和运动模式，因此参数张量的低秩结构有很好的先验支撑
- **参数效率极高**: 30个身份仅需单一身份1/13的参数量
- **可扩展设计**: 新身份只需添加一行，不需要重新训练整个模型
- 将经典的张量分解方法（TensorFaces思想）优雅地迁移到3DGS时代

## 局限与展望

- 身份数很多时结果变平滑，需要依赖个性化微调
- 所有身份共享同样数量的Gaussians $N_g$，无法适应不同体型差异
- 当前仅验证到30个身份，千级规模的可扩展性未验证
- 非刚性变形网络仍然是共享MLP，可能限制极端变形的表达

## 相关工作与启发

- **TensorFaces** (2002): 多线性张量分解表示人脸变化模式的开创性工作，MIGS的直接灵感来源
- **3DGS-Avatar**: MIGS的单身份基线，去掉了per-frame latent code以增强泛化性
- **SNARF**: 可微前向蒙皮方法，用于MIGS的刚性变换模块

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将CP张量分解引入3DGS多身份建模
- **实验充分度**: ⭐⭐⭐⭐ — 两个数据集+详细消融+新身份泛化实验
- **写作质量**: ⭐⭐⭐⭐ — 清晰的数学推导和直觉解释
- **价值**: ⭐⭐⭐⭐ — 多身份高效表示有实际应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation](../../ICCV2025/human_understanding/ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)
- [\[ECCV 2024\] Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)
- [\[ECCV 2024\] Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification](multi-memory_matching_for_unsupervised_visible-infrared_person_re-identification.md)
- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[CVPR 2026\] Open the Motion Door: Atomic Motion Decomposition and Recomposition for Open-Vocabulary Motion Generation](../../CVPR2026/human_understanding/open_the_motion_door_atomic_motion_decomposition_and_recomposition_for_open-voca.md)

</div>

<!-- RELATED:END -->
