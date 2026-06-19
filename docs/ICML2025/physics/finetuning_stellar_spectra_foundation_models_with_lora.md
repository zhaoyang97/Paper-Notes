---
title: >-
  [论文解读] Finetuning Stellar Spectra Foundation Models with LoRA
description: >-
  [ICML 2025][物理/科学计算][LoRA] 首次将 LoRA 应用于恒星光谱基础模型 SpecCLIP，实现以约 100-200 个标注样本将预训练在 LAMOST/Gaia XP 上的模型高效适配到 DESI 巡天数据，证明 LoRA 是跨光谱巡天迁移的轻量而有效策略。 领域现状：大规模光谱巡天（LAMOST、G…
tags:
  - "ICML 2025"
  - "物理/科学计算"
  - "LoRA"
  - "基础模型"
  - "恒星光谱"
  - "跨巡天适配"
  - "小样本学习"
---

# Finetuning Stellar Spectra Foundation Models with LoRA

**会议**: ICML 2025  
**arXiv**: [2507.20972](https://arxiv.org/abs/2507.20972)  
**代码**: 无  
**领域**: AI4Science / 天体物理  
**关键词**: LoRA, 基础模型, 恒星光谱, 跨巡天适配, 小样本学习

## 一句话总结

首次将 LoRA 应用于恒星光谱基础模型 SpecCLIP，实现以约 100-200 个标注样本将预训练在 LAMOST/Gaia XP 上的模型高效适配到 DESI 巡天数据，证明 LoRA 是跨光谱巡天迁移的轻量而有效策略。

## 研究背景与动机

**领域现状**：大规模光谱巡天（LAMOST、Gaia、DESI 等）推动了银河系研究的进步，恒星参数推断方法从传统模板匹配（UlySS、LSP3）发展到机器学习方法（The Cannon、The Payne、TransformerPayne），近期 SpecCLIP 等光谱基础模型开始涌现。

**现有痛点**：不同光谱巡天在波长覆盖、分辨率和信噪比上差异巨大，现有方法多依赖充分监督且绑定特定巡天，很难在异构光谱数据间实现一致的参数估计。基础模型虽有泛化潜力，但如何以最少监督将其适配到新巡天仍是开放问题。

**核心矛盾**：光谱基础模型预训练成本高，全量微调不现实、小样本场景标注稀缺，需要一种参数高效的适配方案在新巡天上快速部署。

**本文目标** (1) LoRA 能否有效适配光谱基础模型到全新巡天？(2) 微调模型的不同模块（基础模型、投影头、下游 MLP）对性能的影响如何？(3) 预训练中嵌入的 Gaia XP 跨模态知识能否帮助 DESI 适配？

**切入角度**：光谱数据具有类似语言的结构化特性（局部特征对应物理信息），LoRA 在 NLP/CV 中已证明有效，作者首次将其引入天文光谱领域。SpecCLIP 的多模态对比预训练提供了丰富的跨巡天知识基准。

**核心 idea**：用 LoRA 以极少参数（不到模型总参数的 3%）和极少样本（约 100 个标注）将光谱基础模型高效迁移到新巡天。

## 方法详解

### 整体框架

输入为 DESI 光谱（归一化并插值到 LAMOST 波长网格 400-560nm），经过预训练的 SpecCLIP 基础模型提取嵌入（768 维），再通过投影网络映射到对比学习共享空间，最终由下游 MLP 预测铁丰度 [Fe/H]。LoRA 模块被选择性地插入到四个位置进行微调。

### 关键设计

1. **SpecCLIP 预训练基础（冻结骨干）**:

    - 功能：为 LAMOST LRS 和 Gaia XP 两种模态各建立基础模型，并通过对比学习对齐
    - 核心思路：LAMOST LRS 基础模型是 6 层 Transformer encoder（42.7M 参数），对 1462 个 flux 点进行 token 化（窗口 20、步长 10 得到 146 个 token），用掩码建模预训练；Gaia XP 模型是 MLP 自编码器处理 343 维光谱。对比训练用 820K 配对光谱通过模态投影网络将嵌入对齐到共享空间
    - 设计动机：对比预训练让模型学到跨巡天共享的物理表示，为后续迁移奠定基础

2. **四模块 LoRA 微调策略**:

    - 功能：对模型的四个不同模块分别或联合插入 LoRA 进行微调
    - 核心思路：LoRA 将权重更新分解为 $\Delta W = AB$（$A \in \mathbb{R}^{m \times r}$, $B \in \mathbb{R}^{r \times n}$, $r \ll \min(m,n)$）。LoRA1 插入 LRS 基础模型所有自注意力层（rank=4, α=8, 129K 参数/0.30%）；LoRA2 插入 LRS 下游 MLP（rank=8, α=16, 31.7K/2.30%）；LoRA3 插入投影网络（rank=16, α=32, 147K/0.29%）；LoRA4 插入投影后下游 MLP（同 LoRA2 配置）
    - 设计动机：不同模块承载不同层次的知识——基础模型编码光谱特征，投影网络编码跨模态对齐，MLP 编码标签映射——逐一测试可揭示知识迁移的关键路径

3. **跨巡天数据适配流程**:

    - 功能：将 DESI 光谱标准化后接入 SpecCLIP 流程
    - 核心思路：DESI 光谱通过 SPARCL 检索，与 LAMOST LRS 相同管线归一化后插值到 400-560nm 波长网格。与 APOGEE DR17 交叉匹配获得 495 颗星的高精度 [Fe/H] 标签，其中 89 用于训练、9 用于验证、396 用于测试。LoRA1/3 微调另用 164 颗无标签 DESI 样本（SNR>50）
    - 设计动机：刻意使用不同子集进行基础模型微调和下游微调，增加适配难度以测试泛化能力

### 损失函数 / 训练策略

下游 MLP 用 [Fe/H] 回归损失训练。评估使用稳健标准差（Tukey Biweight Scale Estimator）和 $R^2$ 指标。每个实验在单张 NVIDIA V100 上 10-180 秒内完成。

## 实验关键数据

### 主实验

| 方法 | 全测试集 σ↓ | 全测试集 R²↑ | 富金属星 σ↓ | 富金属星 R²↑ |
|------|-----------|------------|-----------|------------|
| Zero-shot (MLP1) | 0.2730 | 0.7358 | 0.2479 | 0.0702 |
| LoRA2 | 0.2663 | 0.7156 | 0.2272 | 0.2378 |
| LoRA1+LoRA2 | 0.2227 | 0.7719 | 0.1924 | 0.4173 |
| Zero-shot (MLP2) | 0.2560 | 0.7203 | 0.2371 | 0.0725 |
| **LoRA4** | **0.2023** | **0.7937** | **0.1621** | **0.5106** |
| LoRA1+LoRA3+LoRA4 | 0.2297 | 0.7801 | 0.1851 | 0.4274 |

### 消融实验

| 配置 | 贫金属星 σ↓ | 贫金属星 R²↑ | 说明 |
|------|-----------|------------|------|
| Zero-shot (MLP1) | **0.4444** | **-0.5130** | 零样本在贫金属端反而最好 |
| LoRA2 | 0.5872 | -1.2881 | 仅微调 MLP 在稀疏区域过拟合 |
| LoRA1+LoRA2 | 0.5151 | -0.9143 | 联合微调缓解但不解决 |
| LoRA4 | 0.5803 | -0.8357 | 即使最优配置在贫金属端也退化 |
| LoRA1+LoRA3+LoRA4 | 0.5970 | -0.8159 | 全模块微调同样在此区域失效 |

### 关键发现

- **LoRA4（仅微调 Gaia XP 对齐路径的下游 MLP）表现最优**，说明 SpecCLIP 预训练中引入的 Gaia XP 跨模态知识为 DESI 适配提供了最关键信息，即使 DESI 和 Gaia XP 的分辨率与波段差异很大
- 联合微调基础模型（LoRA1）能带来额外增益，但并非总是最优——对贫金属星反而有害，暗示小样本微调在标签稀疏区域容易过拟合
- 所有方法在贫金属星（[Fe/H] < -1，仅 60 颗测试星）上表现很差，$R^2$ 均为负值

## 亮点与洞察

- **首次将 LoRA 引入恒星光谱学**，证明了 NLP/CV 中的参数高效微调技术在天文领域的可迁移性，为光谱基础模型的跨巡天部署提供了标准范式
- **Gaia XP 知识的间接迁移**特别有趣——LoRA4 的成功说明对比预训练嵌入的跨模态信息（Gaia XP → 共享空间）能通过投影路径间接帮助 DESI，尽管 DESI 本身并未参与预训练
- 极少标注（89 个训练样本）就能实现有意义的性能提升，展示了基础模型 + LoRA 在数据稀缺科学领域的巨大潜力

## 局限与展望

- 贫金属星预测严重退化，需要更好的正则化策略或针对性数据增强
- 训练集仅 89 个标注样本，存在明显的数据稀缺瓶颈，更大的交叉匹配目录可能带来实质性提升
- 仅预测 [Fe/H] 单一参数，未验证对其他恒星参数（有效温度 $T_{\text{eff}}$、表面重力 $\log g$、$[\alpha/\text{Fe}]$）的效果
- 波长范围限制在 400-560nm（LAMOST 网格），未利用 DESI 的完整波长覆盖（360-980nm），可能丢失重要光谱信息

## 相关工作与启发

- **vs The Cannon/The Payne**: 传统方法需要大量标注且绑定特定巡天，LoRA+基础模型方案的标注效率高出数个量级
- **vs 全量微调**: LoRA 仅更新不到 3% 参数且训练秒级完成，全量微调在如此少样本下会严重过拟合
- **vs AstroCLIP**: AstroCLIP 首次将 CLIP 引入天文，SpecCLIP 将其专门化到光谱领域并加入了模态特有信息的保留机制

## 评分

- 新颖性: ⭐⭐⭐ LoRA 本身并非新技术，但首次应用于天文光谱有开创意义
- 实验充分度: ⭐⭐⭐ 系统性对比了不同 LoRA 配置，但仅一个下游任务和一个目标巡天
- 写作质量: ⭐⭐⭐⭐ 方法图清晰，实验设计合理
- 价值: ⭐⭐⭐⭐ 为光谱基础模型的跨巡天部署建立了参数高效微调的标准流程

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] L2D: Large Language Models to Diffusion Finetuning](large_language_models_to_diffusion_finetuning.md)
- [\[ICML 2025\] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)
- [\[NeurIPS 2025\] The Platonic Universe: Do Foundation Models See the Same Sky?](../../NeurIPS2025/physics/the_platonic_universe_do_foundation_models_see_the_same_sky.md)
- [\[ICML 2025\] OmniArch: Building Foundation Model For Scientific Computing](omniarch_building_foundation_model_for_scientific_computing.md)
- [\[ICML 2026\] Foundation Inference Models for Ordinary Differential Equations](../../ICML2026/physics/foundation_inference_models_for_ordinary_differential_equations.md)

</div>

<!-- RELATED:END -->
