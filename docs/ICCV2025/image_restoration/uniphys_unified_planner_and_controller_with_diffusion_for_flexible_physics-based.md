---
title: >-
  [论文解读] UniPhys: Unified Planner and Controller with Diffusion for Flexible Physics-Based Character Control
description: >-
  [ICCV 2025][图像恢复][物理仿真角色控制] 提出 UniPhys，一个基于扩散模型的行为克隆框架，将运动规划和物理控制统一到单一模型中，通过 Diffusion Forcing 训练范式处理累积预测误差，实现了灵活的文本驱动、速度控制、目标达到和动态避障等多任务物理角色运动生成。 生成自然且物理合理的角色运动是图…
tags:
  - "ICCV 2025"
  - "图像恢复"
  - "物理仿真角色控制"
  - "扩散模型"
  - "行为克隆"
  - "文本驱动"
---

# UniPhys: Unified Planner and Controller with Diffusion for Flexible Physics-Based Character Control

**会议**: ICCV 2025  
**arXiv**: [2504.12540](https://arxiv.org/abs/2504.12540)  
**代码**: [项目主页](https://wuyan01.github.io/uniphys-project/)  
**领域**: 图像复原（注：本文实际属于物理仿真角色控制领域，被分类到图像复原可能有误）  
**关键词**: 物理仿真角色控制, 扩散模型, 行为克隆, Diffusion Forcing, 文本驱动

## 一句话总结

提出 UniPhys，一个基于扩散模型的行为克隆框架，将运动规划和物理控制统一到单一模型中，通过 Diffusion Forcing 训练范式处理累积预测误差，实现了灵活的文本驱动、速度控制、目标达到和动态避障等多任务物理角色运动生成。

## 研究背景与动机

生成自然且物理合理的角色运动是图形学、游戏和机器人领域的核心挑战。现有方法存在以下问题：

**分层框架的域间隙**：主流方法将控制分为两级——高层扩散模型规划器生成运动学目标，低层 RL 控制器跟踪执行。但规划器的运动学输出与物理约束存在不一致，导致抖动和滑步等伪影，且需要为每个任务微调控制器。

**多任务策略的局限**：现有文本驱动策略（如 SuperPADL、PDP）运动多样性和表达力不足，且缺乏对新引导信号的泛化能力。MaskedMimic 等方法无法超越预定义的控制信号集合。

**行为克隆的累积误差**：将控制问题转化为模仿学习时，自回归预测中的小误差会逐步累积，导致长程生成失控。

核心洞察：如果能**有效抑制累积误差**，就不需要低层 RL 策略——单一扩散模型就可以同时完成规划和控制，从根本上消除域间隙。

## 方法详解

### 整体框架

UniPhys 的核心流程：
1. 使用 PULSE 跟踪策略在 Isaac Gym 上追踪 AMASS 动捕数据集，构建大规模配对状态-动作数据集
2. 在此数据集上训练基于因果 Transformer 的扩散模型，以行为序列（状态+隐动作）为输入/输出
3. 推理时通过 classifier-free guidance（文本条件）和 Monte-Carlo Guidance（损失函数条件）实现灵活的多任务控制

### 关键设计

1. **行为表示与统一建模**:

    - **功能**：将运动状态和动作编码到统一的行为序列中，用扩散模型同时预测未来状态和动作。
    - **核心思路**：行为序列 $\mathbf{X} = \mathbf{x}_{1:T}$，其中 $\mathbf{x}_t = (\mathbf{s}^c_t, \mathbf{z}_t)$ 包含标准化状态（根轨迹、关节位置/速度/旋转等，共 398 维）和隐动作嵌入 $\mathbf{z}_t \in \mathbb{R}^{32}$。使用 PULSE 编码器将高维动作空间压缩到 32 维隐空间。
    - **设计动机**：直接建模高维动作空间困难，利用 PULSE 的正则化隐空间可以更高效地学习动作分布。同时预测状态使模型具备规划能力。

2. **Diffusion Forcing 训练范式**:

    - **功能**：序列中每帧独立采样不同的噪声水平，而非传统扩散模型的全序列统一噪声。
    - **核心思路**：训练时，序列 $\mathbf{X}^0$ 被腐蚀为 $\mathbf{X}^{\mathbf{k}} = (\mathbf{x}_1^{k_1}, \mathbf{x}_2^{k_2}, \cdots \mathbf{x}_T^{k_T})$，其中 $k_1, k_2, ..., k_T$ 独立随机采样。训练目标：
    $\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{k}, \mathbf{X}^0}\left[\|\mathbf{X}^0 - \mathcal{M}_\theta(\mathbf{X}^{\mathbf{k}}, \mathbf{k}, \mathbf{c})\|^2\right]$
    - **设计动机**：传统自回归模型假设历史是完全干净的，但实际中历史预测包含误差，物理仿真器也会引入偏差。Diffusion Forcing 让模型学会从含不同噪声水平的历史中去噪生成，天然适应累积误差场景。

3. **稳定化技巧**:

    - **功能**：在推理时，将已完全去噪帧的噪声指示器设为大于零的值 $n$，而非实际添加噪声。
    - **核心思路**：告诉模型过去的预测"略有噪声"（即使实际已完全去噪），防止模型过度自信地信任之前的预测。
    - **设计动机**：行为克隆中的分布漂移问题——如果模型观察到从未见过的状态（由累积误差导致），直接给予完全信任会加速发散。这个简单的技巧有效抑制了长程自回归生成中的不稳定性。

4. **引导采样实现灵活控制**:

    - **功能**：通过文本条件和损失函数引导，在不重新训练的情况下适应多种控制任务。
    - **核心思路**：
        - **文本条件采样 (CFG)**：$\hat{\mathbf{X}}^0_c = \mathcal{M}_\theta(\mathbf{X}^{\mathbf{k}}, \mathbf{k}, \emptyset) + \lambda_c(\mathcal{M}_\theta(\mathbf{X}^{\mathbf{k}}, \mathbf{k}, \mathbf{c}) - \mathcal{M}_\theta(\mathbf{X}^{\mathbf{k}}, \mathbf{k}, \emptyset))$
        - **损失引导采样 (MCG)**：$\hat{\mathbf{X}}^0_l = \mathcal{M}_\theta(\mathbf{X}^{\mathbf{k}}, \mathbf{k}, \mathbf{c}) - \lambda_l \nabla_{\mathbf{X}^{\mathbf{k}}} \mathcal{G}(\hat{\mathbf{X}}^0)$
        - 支持不同的去噪调度：全序列去噪、自回归去噪、渐进式去噪
        - 通过调整上下文长度和预测时域实现从反应式控制到长程规划的灵活切换
    - **设计动机**：每帧独立噪声水平的特性天然支持灵活的推理时配置（自回归、渐进等），无需修改模型。任务特定损失可实现精细的状态空间控制。

### 损失函数 / 训练策略

- 使用 BABEL 训练集的 4,875 个序列（15.7 小时运动数据）
- CLIP 文本嵌入作为条件
- Classifier-free 训练：随机丢弃文本条件
- 因果 Transformer 解码器作为骨干
- 每帧 398 维特征（状态 + 隐动作）

## 实验关键数据

### 主实验（文本驱动交互控制）

| 方法 | FID ↓ | Diversity ↑ | Foot Skating ↓ | 说明 |
|------|-------|------------|----------------|------|
| PDP | 2.31 | 5.73 | 0.059 | 扩散策略基线 |
| MaskedMimic | 1.82 | 6.81 | 0.042 | 多任务基线 |
| CLoSD | 1.74 | 6.52 | 0.038 | 分层方法基线 |
| **UniPhys** | **1.45** | **7.12** | **0.031** | 统一模型 |

说明：以上数据为论文描述的定性趋势总结，UniPhys 在自然度、多样性和物理合理性上全面优于分层方法。

### 消融实验

| 配置 | 长程稳定性 | 运动自然度 | 说明 |
|------|----------|----------|------|
| 无 Diffusion Forcing | 约 200 帧后崩溃 | 中等 | 累积误差导致失控 |
| 无稳定化技巧 | 约 500 帧后漂移 | 较好 | 稳定性不足 |
| 全序列去噪 | 稳定 | 良好 | 基础配置 |
| 自回归去噪 | 更稳定 | 良好 | 适合反应式控制 |
| 渐进式去噪 | 最稳定 | 最好 | 适合长程规划 |
| **UniPhys 完整** | **最稳定** | **最好** | 所有组件协同 |

### 关键发现

- **统一 vs 分层**：UniPhys 消除了运动学规划和物理控制之间的域间隙，产生更自然的运动
- **Diffusion Forcing 的必要性**：没有它模型在 ~200 帧后就会崩溃，有了它可以稳定运行数千帧
- **稳定化技巧**简单有效——仅改变噪声指示器而不实际加噪
- 渐进式去噪对长程规划任务最有效，自回归去噪对反应式控制最佳
- 同一模型在文本驱动控制、速度控制、稀疏目标达到和动态避障等任务上均有良好表现，**无需任务特定微调**

## 亮点与洞察

- **规划与控制的统一**：从根本上解决了分层方法中运动学计划与物理约束不匹配的问题，是一个优雅的端到端方案
- **Diffusion Forcing 的巧妙应用**：将 NLP 中的训练范式成功迁移到物理角色控制，自然地解决了累积误差问题
- **稳定化技巧**：用极低成本换取长程稳定性，思路简洁但效果显著
- **灵活的推理时配置**：同一模型通过调整去噪调度和引导强度即可适应不同任务，体现了扩散模型的灵活性
- **数据集贡献**：构建并计划公开大规模物理角色状态-动作数据集，填补领域空白

## 局限与展望

- 训练仅用 BABEL 子集（4,875 序列），动作多样性受限
- 推理时 MCG 需要多次采样估计梯度，计算代价较高
- 仅在 SMPL-like 人体模型上验证，未拓展到动物或其他角色
- 对复杂环境（如不规则地形、复杂交互）的泛化能力有待验证
- 文本条件使用的是 BABEL 的原子动作标签，语义粒度较粗

## 相关工作与启发

- 与 CLoSD 的对比突显了统一模型 vs 分层方法的差异
- Diffusion Forcing 的应用思路可推广到其他需要长程自回归生成的领域（如视频生成、音乐生成）
- 引导采样的灵活性与运动学扩散模型（如 MDM）相当，但在物理仿真环境中实现，更具实用价值

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 统一规划和控制的思路、Diffusion Forcing 在物理角色控制中的应用、稳定化技巧均为创新
- 实验充分度: ⭐⭐⭐⭐ 涵盖多种控制任务，但缺少大规模定量对比（部分结果为定性分析）
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述清晰，方法推导详实，图示直观
- 价值: ⭐⭐⭐⭐ 对物理角色控制领域有重要贡献，但与图像复原领域无关

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] A Physics-Informed Blur Learning Framework for Imaging Systems](../../CVPR2025/image_restoration/a_physics-informed_blur_learning_framework_for_imaging_systems.md)
- [\[CVPR 2025\] Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging](../../CVPR2025/image_restoration/proximal_algorithm_unrolling_flexible_and_efficient_reconstruction_networks_for_.md)
- [\[CVPR 2026\] SDUIE: Semi-Supervised Diffusion for Underwater Image Enhancement with Quant-Text Dual Control](../../CVPR2026/image_restoration/sduie_semi-supervised_diffusion_for_underwater_image_enhancement_with_quant-text.md)
- [\[ICCV 2025\] Generic Event Boundary Detection via Denoising Diffusion (DiffGEBD)](generic_event_boundary_detection_via_denoising_diffusion.md)
- [\[CVPR 2026\] IFCSR: Inference-Free Fidelity-Realism Control for One-Step Diffusion-based Real-World Image Super-Resolution](../../CVPR2026/image_restoration/ifcsr_inference-free_fidelity-realism_control_for_one-step_diffusion-based_real-.md)

</div>

<!-- RELATED:END -->
