---
title: >-
  [论文解读] Motion Mamba: Efficient and Long Sequence Motion Generation
description: >-
  [ECCV 2024][人体理解][人体运动生成] 本文提出 Motion Mamba，首次将选择性状态空间模型（Mamba）引入人体运动生成任务，通过层次化时序 Mamba（HTM）和双向空间 Mamba（BSM）两个核心模块，在 HumanML3D 上实现 FID 降低50%（0.473→0.281），同时推理速度提升4倍（0.217s→0.058s）。
tags:
  - "ECCV 2024"
  - "人体理解"
  - "人体运动生成"
  - "状态空间模型"
  - "Mamba"
  - "潜在扩散模型"
  - "长序列建模"
---

# Motion Mamba: Efficient and Long Sequence Motion Generation

**会议**: ECCV 2024  
**arXiv**: [2403.07487](https://arxiv.org/abs/2403.07487)  
**代码**: [https://github.com/steve-zeyu-zhang/MotionMamba](https://github.com/steve-zeyu-zhang/MotionMamba)  
**领域**: 人体理解  
**关键词**: 人体运动生成, 状态空间模型, Mamba, 潜在扩散模型, 长序列建模

## 一句话总结
本文提出 Motion Mamba，首次将选择性状态空间模型（Mamba）引入人体运动生成任务，通过层次化时序 Mamba（HTM）和双向空间 Mamba（BSM）两个核心模块，在 HumanML3D 上实现 FID 降低50%（0.473→0.281），同时推理速度提升4倍（0.217s→0.058s）。

## 研究背景与动机
- **领域现状**：人体运动生成是生成式计算机视觉中的重要方向，条件化（如文本驱动）的运动生成已有多种方法，包括自编码器、GAN、自回归和扩散模型。其中扩散模型凭借生成质量和多样性的优势成为主流。
- **现有痛点**：
  1. 卷积或 Transformer 架构的扩散方法在长序列运动生成上面临计算量平方增长的瓶颈，Transformer 并非天然为时序建模设计
  2. 即使在潜在空间做扩散（如 MLD），注意力机制的二次复杂度仍限制推理效率
- **核心矛盾**：如何在保持高质量生成的同时，处理长程依赖并维持近线性计算复杂度
- **切入角度**：状态空间模型（SSM），特别是 Mamba，具有高效的硬件感知设计和长序列建模能力，但缺乏针对运动数据的专用架构
- **核心 idea**：设计专门面向运动生成的 SSM 架构，在时序维度用分层扫描、在空间维度用双向扫描，结合潜在扩散模型实现高效高质的运动生成

## 方法详解

### 整体框架
Motion Mamba 基于潜在运动扩散模型（Latent Motion Diffusion），整体采用去噪 U-Net 架构。首先用 Motion VAE 将运动序列压缩到潜在空间，然后在潜在空间中进行扩散过程。去噪器 $\epsilon_\theta$ 包含 N 个编码器块 $E_{1..N}$、N 个解码器块 $D_{1..N}$ 和一个中间的 Transformer 注意力混合块 $M$。每个编解码器块由 HTM 和 BSM 两个核心模块组成。文本条件通过冻结的 CLIP-VIT-L-14 编码器获取嵌入。

### 关键设计
1. **层次化时序 Mamba（HTM）**:

    - **功能**：沿时间维度处理潜在表示，捕获不同深度的时序依赖
    - **为什么**：作者发现低层特征空间中的运动密度更高，需要更多扫描来捕获细节。单一扫描数量无法同时满足效率和质量
    - **怎么做**：在 U-Net 的编解码器中，按层次分配不同数量的 SSM 扫描。外层（靠近输入/输出）使用更多扫描 $S_{2N-1}$，内层使用更少扫描 $S_1$，形成对称的层次结构 $K=\{S_{2N-1}, S_{2(N-1)-1}, ..., S_1\}$。每个扫描包含独立的 SSM 模块（1D卷积→线性投影获取 $B,C,\Delta$→离散化→SSM计算），所有扫描输出聚合后经线性投影得到最终输出
    - **与之前方法的区别**：相比 Transformer 的固定注意力头数设计，HTM 利用 SSM 低参数特性增加扫描数，通过层次化分配实现效率与质量的平衡

2. **双向空间 Mamba（BSM）**:

    - **功能**：沿通道/空间维度处理潜在姿态，增强单帧内的运动生成精度
    - **为什么**：潜在骨架的结构化信息流在正向和反向方向上都包含重要信息，单向扫描会遗漏反方向的依赖
    - **怎么做**：首先将输入维度从 $(T,B,C)$ 重排为 $(C,B,T)$，交换时序和通道维度。然后对通道维度进行前向和后向两个方向的 SSM 扫描，最终通过门控求和（GateAndSum）融合双向输出
    - **与之前方法的区别**：不同于 Vim 的视觉双向 SSM，BSM 专门针对运动潜在空间中的通道维度设计，通过维度重排确保空间信息的双向流动

3. **注意力混合块**:

    - 在 U-Net 底部插入一个 Transformer 注意力块 $M$，用于增强条件融合能力，是时序和条件信息交互的枢纽

### 损失函数 / 训练策略
- 采用标准潜在扩散训练目标：最小化潜在空间中真实噪声和预测噪声之间的 MSE
- 使用 AdamW 优化器，学习率 $10^{-4}$，batch size 512（4 GPU 并行）
- 训练 2000 epochs，扩散步数训练时1000步、推理时50步
- 模型配置：11层，潜在维度 $z \in \mathbb{R}^{2,d}$

## 实验关键数据

### 主实验

| 数据集 | 指标 | Motion Mamba | MLD (前SOTA) | 提升 |
|--------|------|------|----------|------|
| HumanML3D | FID↓ | **0.281** | 0.473 | -40.6% |
| HumanML3D | R-Precision Top3↑ | **0.792** | 0.772 | +2.6% |
| HumanML3D | MM Dist↓ | **3.060** | 3.196 | -4.3% |
| KIT-ML | FID↓ | **0.307** | 0.404 | -24.0% |
| KIT-ML | R-Precision Top3↑ | **0.765** | 0.734 | +4.2% |
| HumanML3D-LS(长序列) | FID↓ | **0.668** | 0.952 | -29.8% |

### 消融实验

| 配置 | FID | 说明 |
|------|---------|------|
| MM $\{S_1,...,S_N\}$（低到高） | 1.278 | 基线扫描顺序，效果最差 |
| MM $\{S_N,...,S_1\}$（高到低） | 0.962 | 反转后提升明显 |
| MM $\{S_{2N_n-1},...,S_1\}$（层次化） | **0.281** | 最优，层次化设计大幅提升 |
| SingleScan | 1.063 | 单向扫描效果有限 |
| BiScan, block | **0.281** | 块级双向扫描最优 |
| Dim=1 | 0.652 | 维度太低限制表达 |
| Dim=2 | **0.281** | 最优维度 |
| 9 layers | 1.080 | 层数不足 |
| 11 layers | **0.281** | 最优层数 |
| 27/37 layers | 0.975/0.809 | 过深反而退化 |

### 关键发现
- 层次化扫描策略是最大贡献因子，FID 从1.278降至0.281
- 最优潜在维度为2（而非 MLD 的1），因为 HTM 多扫描机制需要额外维度承载信息
- 每个 Mamba 层参数约为 Transformer 编码器块的25%，允许在增加层数后仍保持高效
- 在长序列子集 HumanML3D-LS 上优势更明显（FID 0.668 vs MLD 0.952）
- 推理时间仅 0.058s/序列，是 MLD（0.217s）的约4倍加速

## 亮点与洞察
- **首次将 Mamba 应用于运动生成**：开辟了 SSM 在运动领域的新方向，证明 SSM 可替代 Transformer 作为扩散模型的骨干
- **层次化时序扫描设计精巧**：利用了"低层特征运动密度更高"的观察，在 U-Net 对称架构中自然匹配
- **效率-质量双赢**：SSM 的线性复杂度使得增加扫描数量在计算上可行，而 Transformer 做不到类似的"增加头数"同时保持效率
- 用户研究进一步验证了生成质量：在文本-运动对应性和质量评估上分别以62%和59%超过 MLD

## 局限与展望
- 仅在文本到运动任务上验证，未扩展到音乐到舞蹈、动作到运动等其他条件生成任务
- 仍依赖冻结的 Motion VAE 做压缩，VAE 本身的重构误差是生成质量的上界
- 潜在维度为2较小，可能限制了对极复杂运动的表达能力
- 未与最新的 MoMask 等非扩散方法做充分对比

## 相关工作与启发
- **MLD** [Chen et al.] 提出在潜在空间做运动扩散，是本文的直接基线
- **Mamba** [Gu & Dao] 的选择性 SSM 提供了硬件高效的长序列建模基础
- **VMamba / Vim** 探索了 SSM 在2D视觉中的应用，本文首次将其扩展到运动序列这一1D时序+结构化姿态数据
- 启发：SSM 在其他时序生成任务（如语音合成、轨迹预测）中也可能取代 Transformer 获得效率提升

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将 Mamba 引入运动生成，HTM 层次化设计有独到之处
- 实验充分度: ⭐⭐⭐⭐ 消融实验覆盖全面，用户研究也有，但缺少与最新SOTA的全面对比
- 写作质量: ⭐⭐⭐⭐ 整体清晰，算法描述用伪代码辅助理解
- 价值: ⭐⭐⭐⭐ 在效率和质量上同时取得突破，开辟了 SSM in motion 新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)
- [\[ECCV 2024\] HUMOS: Human Motion Model Conditioned on Body Shape](humos_human_motion_model_conditioned_on_body_shape.md)
- [\[CVPR 2026\] Causal Motion Diffusion Models for Autoregressive Motion Generation](../../CVPR2026/human_understanding/causal_motion_diffusion_models_for_autoregressive_motion_generation.md)
- [\[ECCV 2024\] CoMo: Controllable Motion Generation Through Language Guided Pose Code Editing](como_controllable_motion_generation_through_language_guided_pose_code_editing.md)
- [\[CVPR 2026\] Open the Motion Door: Atomic Motion Decomposition and Recomposition for Open-Vocabulary Motion Generation](../../CVPR2026/human_understanding/open_the_motion_door_atomic_motion_decomposition_and_recomposition_for_open-voca.md)

</div>

<!-- RELATED:END -->
