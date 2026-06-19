---
title: >-
  [论文解读] Prior Does Matter: Visual Navigation via Denoising Diffusion Bridge Models
description: >-
  [CVPR 2025][图像恢复][扩散桥模型] NaviBridger 将去噪扩散桥模型（DDBM）引入视觉导航任务，用信息丰富的先验动作替代高斯噪声作为去噪起点，理论证明源分布越接近目标分布误差上界越低，并设计了高斯/规则/学习三种先验策略，在室内外仿真和真实场景中均加速推理并超越基线。 领域现状：视觉导航中的局部路径规…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "扩散桥模型"
  - "视觉导航"
  - "模仿学习"
  - "先验动作"
  - "去噪"
---

# Prior Does Matter: Visual Navigation via Denoising Diffusion Bridge Models

**会议**: CVPR 2025  
**arXiv**: [2504.10041](https://arxiv.org/abs/2504.10041)  
**代码**: [https://github.com/hren20/NaiviBridger](https://github.com/hren20/NaiviBridger)  
**领域**: 图像复原  
**关键词**: 扩散桥模型, 视觉导航, 模仿学习, 先验动作, 去噪

## 一句话总结

NaviBridger 将去噪扩散桥模型（DDBM）引入视觉导航任务，用信息丰富的先验动作替代高斯噪声作为去噪起点，理论证明源分布越接近目标分布误差上界越低，并设计了高斯/规则/学习三种先验策略，在室内外仿真和真实场景中均加速推理并超越基线。

## 研究背景与动机

**领域现状**：视觉导航中的局部路径规划已经开始使用扩散模型进行模仿学习（如 NoMaD），从专家演示数据中学习生成动作序列。这类方法在建模多模态分布和捕获序列相关性方面具有优势。

**现有痛点**：标准扩散模型（如 DDPM）从纯高斯噪声开始去噪，但目标动作分布与高斯噪声差异很大，导致：(1) 大量去噪步骤被浪费在将噪声约束到动作空间中，真正与任务相关的精修只发生在最后几步；(2) 有效动作分布是稀疏的，从混沌随机噪声出发很难生成准确动作，尤其缺乏引导时。这两个问题共同增加了计算成本并降低了性能。

**核心矛盾**：扩散模型的强大生成能力与其固定的高斯初始化之间的矛盾——强大的模型被低效的初始化拖累。

**本文目标**：构造合适的初始分布，将扩散模型从"从噪声生成"变为"从有意义的先验出发的分布变换"，减少去噪步骤并提升动作质量。

**切入角度**：受最优传输理论和扩散桥方法（如 DDBM）在图像翻译/修复中的成功启发，作者认为可以将这一思想迁移到机器人学习中——关键挑战在于视觉导航没有天然的成对源分布。

**核心 idea**：用 DDBM 替代标准扩散模型，设计三种先验动作生成策略（高斯、规则、学习），使去噪过程从接近目标的分布开始，减少所需去噪步骤并提升精度。

## 方法详解

### 整体框架

NaviBridger 的整体架构由三部分组成：(1) 特征提取模块——用 Transformer 编码器处理当前和历史视觉观测序列及目标图像，生成上下文向量 $c_t$；(2) 先验动作生成模块——根据上下文向量生成源分布动作 $a_s$；(3) 去噪扩散桥模块——用 1D 时序 CNN，以 FiLM 调制条件向量，通过 DDBM 的反向过程将先验动作变换为目标动作 $a_0$。

### 关键设计

1. **去噪扩散桥（DDBM）在模仿学习中的适配**:

    - 功能：实现从任意源分布到目标动作分布的变换，而非从固定高斯分布出发
    - 核心思路：DDBM 的反向 SDE 同时包含 score function $s$ 和 Doob's h-transform $h$，前者引导去噪方向，后者确保轨迹在终止时间 $T$ 到达目标分布。采样分布 $q(a_t|a_0, a_T)$ 是条件高斯，其均值混合了目标和源动作，方差随 SNR 比例缩放
    - 设计动机：标准扩散模型只能将复杂数据分布映射到高斯分布，DDBM 通过 Doob's h-transform 突破了这一限制，允许在任意两个分布间建立桥接。这使得当有好的先验时可以大幅减少去噪步骤

2. **三种先验动作生成策略**:

    - 功能：根据不同场景和可用信息，提供从无信息到有信息的先验选择
    - 核心思路：
        - **高斯先验**：与标准扩散相同的白噪声，作为无信息基线，使框架向后兼容
        - **规则先验**：用 FC 层预测路径长度和运动行为分类（直行/左转/右转/U 型转），据此生成抛物线先验路径。先验动作是根据预测的动作类别和距离手工构造的参数化曲线
        - **学习先验**：用轻量级 CVAE 从观测直接生成先验动作，CVAE 编码器接收观测+动作学习后验分布，解码器从先验分布采样生成动作
    - 设计动机：理论分析表明误差上界 $E[||a_t - a_0||^2] \leq C \cdot D_{KL}(\pi_s || \pi)$，源分布越接近目标，误差上界越低。学习先验最接近目标分布，但需要额外模型；规则先验不需要学习但需要领域知识；高斯先验是 fallback

3. **FiLM 条件调制去噪网络**:

    - 功能：将视觉观测信息注入去噪过程，使动作生成条件化于当前场景
    - 核心思路：使用 1D 时序 CNN 作为去噪网络，通过 Feature-wise Linear Modulation (FiLM) 将上下文向量 $c_t$ 应用到中间特征层，进行 $k$ 次迭代去噪
    - 设计动机：1D CNN 适合处理时序动作序列，FiLM 是轻量且有效的条件注入方式，且与 NoMaD 等基线使用相同结构便于公平对比

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \lambda_b \mathcal{L}_b + \lambda_p \mathcal{L}_p + \lambda_d \mathcal{L}_d$：

- **扩散桥损失** $\mathcal{L}_b$：加权 MSE，预测目标动作 $a_0$ 与 ground truth 的距离
- **先验动作损失** $\mathcal{L}_p$：规则先验用交叉熵（分类）+ MSE（回归）；学习先验用 MSE + KL 散度（CVAE）
- **时间距离损失** $\mathcal{L}_d$：预测目标图像与当前图像的时间间隔，用于高层规划

训练 30 epochs，学习率 0.0001，batch size 256，单卡 RTX TITAN 约 30 小时。DDBM 使用 VE 模型，$\sigma_0 = \sigma_T = 0.5$，默认 $k=10$ 步采样。

## 实验关键数据

### 主实验

| 场景 | 方法 | 基础任务成功率 | 适应任务成功率 | 碰撞次数↓ |
|------|------|---------------|---------------|----------|
| 室内 | ViNT | 68% | 28% | 1.02/1.58 |
| 室内 | NoMaD (DDPM) | 86% | 32% | 0.74/1.32 |
| 室内 | NaviBridger-Gaussian | 82% | 64% | 0.72/0.98 |
| 室内 | NaviBridger-Learning | **92%** | **88%** | **0.61/0.41** |
| 室外 | NoMaD (DDPM) | 22% | 52% | 0.58/0.34 |
| 室外 | NaviBridger-Learning | **44%** | **64%** | **0.51/0.30** |

### 消融实验

| 配置 (k=去噪步数) | DDPM Avg Rank | NaviBridger-Gaussian Avg Rank |
|-------------------|---------------|------------------------------|
| k=10 | 7.0 | 2.4 |
| k=7 | 6.0 | 2.6 |
| k=4 | 4.4 | 2.4 |
| k=1 | 7.8 | 3.2 |

### 关键发现

- Learning-based 先验在所有场景和任务上均最优，尤其在适应任务（环境变化）中提升极为显著（室内 32%→88%）
- 规则先验在特定场景有效但泛化差，在某些设置下反而低于基线（室外基础任务 14%）——说明不合适的先验比没有先验更糟
- 高斯先验下的 NaviBridger 仍优于 DDPM-based NoMaD——DDBM 框架本身的去噪效率就更高
- 减少去噪步数时，NaviBridger 的性能几乎不衰减（rank 稳定在 2.4-3.2），而 DDPM 在 k=1 时急剧恶化——证明少量步骤即可收敛
- 可视化显示 DDBM 方法在 2-4 步就达到稳定动作，而 DDPM 的大部分步骤都在约束噪声进入动作空间

## 亮点与洞察

- **理论分析驱动的设计**：不是拍脑袋选择先验，而是从 KL 散度误差上界出发推导为何好的先验更好。$E[||a_t - a_0||^2] \leq C \cdot D_{KL}(\pi_s || \pi)$ 这个 bound 简洁有力，为未来的先验设计提供了理论指导
- **框架的通用性**：NaviBridger 不仅适用于视觉导航，还可以直接迁移到其他模仿学习任务（操作、运动规划），只需替换先验策略
- **少步推理的实用价值**：在机器人实时系统中（部署在 Jetson Orin AGX），少步去噪意味着更低延迟，这对安全关键的导航至关重要

## 局限与展望

- 学习先验需要额外的 CVAE 训练，增加了系统复杂性；如果先验模型本身不好，可能引入偏差
- 规则先验对导航行为的分类过于粗糙（5 类），在复杂环境中适用性有限
- 仅在 point-goal 和 image-goal 导航中验证，未在语义导航或操作任务中测试
- 与更新的视觉基础模型（如 VLA）的结合尚未探索

## 相关工作与启发

- **vs NoMaD**：NoMaD 是第一个将策略扩散应用于视觉导航的方法，但使用标准 DDPM 从高斯噪声去噪。NaviBridger 的关键改进在于用 DDBM 替换了去噪框架，允许从更好的起点出发。在相同训练数据和网络结构下，仅改变去噪框架就带来了显著提升
- **vs ViNT**：ViNT 是基于回归的方法（CNN + self-attention），不使用扩散模型。NaviBridger 在生成质量和多模态处理上更有优势
- **vs DiffusionPolicy**：扩散策略在操作任务中表现优异但也受限于从高斯噪声出发。NaviBridger 的 DDBM 思路可以直接迁移到扩散策略中

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将扩散桥模型应用于视觉导航，理论分析有洞察
- 实验充分度: ⭐⭐⭐⭐ 室内外仿真+真实机器人，三种先验对比+消融
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，但论文有些冗长
- 价值: ⭐⭐⭐⭐ 对所有使用扩散模型做决策的工作都有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Audio Super-Resolution with Latent Bridge Models](../../NeurIPS2025/image_restoration/audio_super-resolution_with_latent_bridge_models.md)
- [\[ICML 2025\] ε-VAE: Denoising as Visual Decoding](../../ICML2025/image_restoration/epsilon-vae_denoising_as_visual_decoding.md)
- [\[CVPR 2025\] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)
- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](efficient_visual_state_space_model_for_image_deblurring.md)
- [\[ICML 2026\] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](../../ICML2026/image_restoration/early_decisions_matter_proximity_bias_and_initial_trajectory_shaping_in_non-auto.md)

</div>

<!-- RELATED:END -->
