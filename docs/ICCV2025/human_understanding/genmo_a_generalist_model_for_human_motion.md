---
title: >-
  [论文解读] GENMO: A GENeralist Model for Human MOtion
description: >-
  [ICCV 2025][人体理解][人体运动建模] 提出 GENMO，首个统一人体运动估计（从视频/2D 关键点恢复运动）和运动生成（从文本/音乐/关键帧合成运动）的通用模型，通过双模式训练范式（回归+扩散）在单一模型中同时实现精确估计和多样生成。 人体运动建模是计算机视觉与图形学的长期研究课题，在游戏、动画、3D 内容创作…
tags:
  - "ICCV 2025"
  - "人体理解"
  - "人体运动建模"
  - "运动估计"
  - "运动生成"
  - "扩散模型"
  - "多模态条件"
---

# GENMO: A GENeralist Model for Human MOtion

**会议**: ICCV 2025  
**arXiv**: [2505.01425](https://arxiv.org/abs/2505.01425)  
**代码**: [项目页面](https://research.nvidia.com/labs/dair/genmo)  
**领域**: 人体理解  
**关键词**: 人体运动建模, 运动估计, 运动生成, 扩散模型, 多模态条件

## 一句话总结

提出 GENMO，首个统一人体运动估计（从视频/2D 关键点恢复运动）和运动生成（从文本/音乐/关键帧合成运动）的通用模型，通过双模式训练范式（回归+扩散）在单一模型中同时实现精确估计和多样生成。

## 研究背景与动机

人体运动建模是计算机视觉与图形学的长期研究课题，在游戏、动画、3D 内容创作中有广泛应用。考虑一个实际创作场景：用户想生成一段运动序列，先从一个视频片段开始，过渡到按文本描述行动，再同步音乐节拍，最后对齐另一段视频——同时还要精细控制关键帧。这需要模型同时具备精确复现观察到的运动和多样化生成的能力。

**核心矛盾**：运动估计和运动生成有本质不同的目标：
- **估计**要求精确确定性输出，给定同一视频应恢复出唯一的运动序列
- **生成**要求多样性输出，给定同一文本描述应能生成多种合理运动

这种矛盾导致两类任务通常由独立模型处理，限制了跨任务知识迁移。

**关键洞察**：
1. 生成先验可改善估计——在遮挡等困难场景下，生成模型学到的运动分布可以提供约束
2. 多样视频数据可增强生成——大量野外视频（仅有 2D 标注）扩充了生成模型的运动分布
3. 扩散模型为统一两者提供了天然框架——估计可视为"最大似然生成"

## 方法详解

### 整体框架

GENMO 基于扩散模型框架，将运动估计重新定义为受约束的运动生成：给定条件信号（视频、2D 关键点、文本、音乐、3D 关键帧），生成满足约束的运动序列 $x = \{x^i\}_{i=1}^N$。每帧运动包含：
$$x^i = (\Gamma_{\text{gv}}^i, v_{\text{root}}^i, \theta^i, \beta^i, t_{\text{root}}^i, \pi^i, p^i)$$
分别是重力-视角朝向（6D）、根节点速度（3D）、SMPL 关节角度（24×6D）、体型参数（10D）、根节点平移（3D）、相机位姿和接触标签。

### 关键设计

1. **双模式训练范式（Dual-Mode Training）**:

    - 功能：在单一模型中兼顾估计的精确性和生成的多样性
    - 核心思路：
        - **估计模式**：将模型输入设为纯高斯噪声 $z \sim \mathcal{N}(\mathbf{0}, I)$，时间步设为最大值 $T$，直接回归清洁运动：
       $$\mathcal{L}_{\text{est}} = \mathbb{E}_{z \sim \mathcal{N}(\mathbf{0}, I)} [\|x_0 - \mathcal{G}(z, T, \mathcal{C}, \mathcal{M})\|^2]$$
       这等价于最大似然估计，迫使模型从纯噪声一步预测最可能的运动
        - **生成模式**：标准 DDPM 训练，从加噪运动逐步去噪：
       $$\mathcal{L}_{\text{gen}} = \mathbb{E}_{t, x_t} [\|x_0 - \mathcal{G}(x_t, t, \mathcal{C}, \mathcal{M})\|^2]$$
    - 设计动机：作者发现视频条件下的扩散模型表现出高度确定性——第一步预测就非常接近最终结果，而文本条件下方差很大。因此对估计任务，加强"第一步预测"的质量（估计模式）至关重要，同时不能损害多样生成能力（生成模式）

2. **多模态条件处理架构**:

    - 功能：支持任意组合的视频、音乐、2D 关键点、文本等多模态条件输入
    - 核心思路：
        - **帧对齐条件**（视频、音乐、2D 骨架）：通过加法融合块（Additive Fusion Block）将各模态特征经各自 MLP 投影后求和，再与噪声运动融合生成 token 序列
        - **文本条件**：使用创新的**多文本注入块（Multi-Text Injection Block）**，支持多段文本在不同时间窗口的注入：
       $$f_{\text{out}} = \sum_{k=1}^K \text{MaskedMHA}(f_{\text{in}}, c_{\text{text}}^k, \Omega_k)$$
       其中 $\Omega_k(i,j)$ 是二值掩码，限定第 $k$ 段文本仅影响其时间窗口内的运动帧
        - 主干网络使用 RoPE-based Transformer，支持变长序列和推理时的滑窗注意力
    - 设计动机：文本与运动帧没有逐帧对齐关系，不能简单拼接（会引入位置偏差）。多文本注入通过掩码注意力优雅解决了多段文本按时间段控制运动的需求

3. **估计引导的 2D 训练（Estimation-Guided Training）**:

    - 功能：利用仅有 2D 标注的野外视频增强生成模型的多样性
    - 核心思路：分两步利用 2D 数据：
      1. 先用估计模式从 2D 条件生成伪 3D 运动 $\hat{x}_0 = \mathcal{G}(z, T, \mathcal{C})$
      2. 对伪运动加噪后用生成模式训练，损失函数使用 2D 重投影：
       $$\mathcal{L}_{\text{gen-2D}} = \mathbb{E} [\|x_{\text{2d}} - \Pi(\mathcal{G}(\hat{x}_t, t, \mathcal{C}))\|^2]$$
    - 设计动机：3D 运动捕捉数据稀少且多样性有限，而 2D 标注可通过检测器大规模获取。直接将 2D 视频通过估计能力转化为训练数据，既扩充了生成分布又避免了 3D 伪标签的噪声问题

### 损失函数 / 训练策略

- **估计模式**：$\mathcal{L}_{\text{est}} + \mathcal{L}_{\text{geo}}$（含 3D 关节/顶点约束、接触约束等几何正则项）
- **生成模式**：$\mathcal{L}_{\text{gen}} + \mathcal{L}_{\text{geo}}$（3D 数据）或 $\mathcal{L}_{\text{gen-2D}} + \mathcal{L}_{\text{geo}}$（2D 数据）
- **模式选择策略**：强条件（视频/2D骨架）同时使用估计和生成模式；弱条件（文本/音乐）仅使用生成模式
- 推理时支持滑窗注意力（窗口 $W$ 帧），实现超长序列生成

## 实验关键数据

### 主实验

**全局运动估计（EMDB-2 数据集）**：

| 方法 | WA-MPJPE100 | W-MPJPE100 | RTE | Foot Sliding |
|------|------------|------------|-----|-------------|
| WHAM (DPVO) | 135.6 | 354.8 | 6.0 | 4.4 |
| GVHMR (DPVO) | 111.0 | 276.5 | 2.0 | 3.5 |
| TRAM (DROID) | 76.4 | 222.4 | 1.4 | - |
| **GENMO (DROID)** | **74.3** | **202.1** | **1.2** | 8.8 |

**音乐-舞蹈生成（AIST++）**：

| 方法 | FIDk↓ | FIDm↓ | PFC↓ | BAS↑ |
|------|-------|-------|------|------|
| Bailando | 28.16 | 9.62 | 1.754 | 0.2332 |
| EDGE | 42.16 | 22.12 | 1.5363 | 0.2334 |
| **GENMO (music only)** | **16.10** | **13.91** | 0.7340 | 0.2282 |
| **GENMO (generalist)** | 40.91 | 18.51 | **0.3702** | **0.2708** |

### 消融实验

**双模式训练的贡献（运动估计，RICH 数据集）**：

| 配置 | WA-MPJPE100 | W-MPJPE100 | 说明 |
|------|------------|------------|------|
| Diffusion-only | 88.9 | 143.9 | 仅标准扩散训练 |
| Regression-only | 87.0 | 141.0 | 仅回归训练 |
| **Dual-mode** | **75.3** | **118.6** | 双模式最优 |

**2D 训练数据的影响（文本-运动生成，Motion-X）**：

| 配置 | FID↓ | R@3↑ | MM Dist↓ | 说明 |
|------|------|------|---------|------|
| MDM baseline | 2.389 | 0.313 | 6.745 | 基线方法 |
| w/o 2D Training | 0.515 | 0.401 | 5.210 | 无 2D 数据 |
| **w/ 2D Training** | **0.207** | **0.472** | **4.801** | 加入 2D 数据 |

### 关键发现

1. **统一模型优于专用模型**：GENMO 在运动估计上超越专门的估计方法 TRAM（W-MPJPE 202.1 vs 222.4），这得益于生成先验提供了对运动合理性的约束
2. **双模式训练两者都不可缺**：纯扩散或纯回归都不如双模式。扩散提供生成多样性，回归保证估计精确性
3. **2D 数据训练全面提升生成质量**：在 Motion-X 上 FID 从 0.515 降至 0.207，R@3 从 0.401 升至 0.472，验证了从野外视频中提取训练信号的有效性
4. 通用模型在音乐-舞蹈上的 FID 不如专用模型（40.91 vs 16.10），但多样性、物理合理性和音乐节拍对齐更优（PFC 0.37 vs 0.73, BAS 0.27 vs 0.23），体现了多任务训练的互惠效应
5. 运动插帧实验进一步验证了估计+生成协同训练的益处

## 亮点与洞察

- **范式性创新**：首次论证了运动估计和生成可以统一在一个扩散框架中，且互有增益
- **双模式训练的理论基础**令人信服：估计模式对应扩散模型的"最大时间步去噪"，与生成模式完全兼容
- **多文本注入设计**解决了文本与运动帧对齐的实际问题，支持"先跑步5秒，然后跳舞10秒"这样的分段控制
- **变长序列支持**通过 RoPE + 滑窗注意力实现，无需后处理拼接，产生自然连贯的长序列
- NVIDIA 出品，工程实现和数据规模都很扎实

## 局限与展望

- 使用 SMPL 参数表示运动，导致在 HumanML3D 指标上不如使用专用表示的方法（存在表示不匹配问题）
- 通用模型在单个任务上通常不如专用模型（如音乐-舞蹈的 FIDk），虽然综合性能更优
- 训练涉及多个数据集和多种训练模式，实现复杂度较高
- 单人运动模型，不支持多人交互运动
- 仅支持 SMPL 骨架表示，不含面部表情和手部细节

## 相关工作与启发

- GVHMR (NeurIPS 2024) 提出的 gravity-view 坐标系被 GENMO 直接采用，证明了该坐标系的通用价值
- MDM (ICLR 2023) 的扩散运动生成范式是基础，GENMO 在此之上增加了估计模式和多模态架构
- 与 MotionGPT 等大模型方法关注语言理解不同，GENMO 更关注精确的几何约束和物理合理性
- 双模式训练思想可推广到其他估计+生成统一任务（如 3D 场景、音频）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 统一估计与生成是重要范式创新，双模式训练和估计引导 2D 训练都很新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖估计（全局/局部）和生成（文本/音乐/插帧）多个任务，消融完整
- 写作质量: ⭐⭐⭐⭐ 内容丰富但篇幅较长，部分架构细节可更简洁
- 价值: ⭐⭐⭐⭐⭐ 对人体运动建模领域有显著推动，统一框架+双向互惠是有说服力的研究方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)
- [\[ECCV 2024\] Large Motion Model for Unified Multi-Modal Motion Generation](../../ECCV2024/human_understanding/large_motion_model_for_unified_multi-modal_motion_generation.md)
- [\[ECCV 2024\] HUMOS: Human Motion Model Conditioned on Body Shape](../../ECCV2024/human_understanding/humos_human_motion_model_conditioned_on_body_shape.md)
- [\[ICCV 2025\] EgoAgent: A Joint Predictive Agent Model in Egocentric Worlds](egoagent_a_joint_predictive_agent_model_in_egocentric_worlds.md)
- [\[NeurIPS 2025\] MOSPA: Human Motion Generation Driven by Spatial Audio](../../NeurIPS2025/human_understanding/mospa_human_motion_generation_driven_by_spatial_audio.md)

</div>

<!-- RELATED:END -->
