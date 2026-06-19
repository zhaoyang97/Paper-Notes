---
title: >-
  [论文解读] Information Density Principle for MLLM Benchmarks
description: >-
  [ICCV 2025][多模态VLM][基准评价] 提出"信息密度"原则从 Fallacy（错误）、Difficulty（难度）、Redundancy（冗余）、Diversity（多样性）四个维度评估 MLLM benchmark 质量，构建了一套 Human-Model-Data 三级自动化评估流水线，对 19 个主流 benchmark 进行了系统性的"benchmark for benchmark"分析。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "基准评价"
  - "信息密度"
  - "MLLM评估"
  - "benchmark质量"
  - "元评估"
---

# Information Density Principle for MLLM Benchmarks

**会议**: ICCV 2025  
**arXiv**: [2503.10079](https://arxiv.org/abs/2503.10079)  
**代码**: [GitHub](https://github.com/lcysyzxdxc/bench4bench)  
**领域**: 多模态VLM  
**关键词**: 基准评价, 信息密度, MLLM评估, benchmark质量, 元评估

## 一句话总结

提出"信息密度"原则从 Fallacy（错误）、Difficulty（难度）、Redundancy（冗余）、Diversity（多样性）四个维度评估 MLLM benchmark 质量，构建了一套 Human-Model-Data 三级自动化评估流水线，对 19 个主流 benchmark 进行了系统性的"benchmark for benchmark"分析。

## 研究背景与动机

随着多模态大语言模型（MLLM）快速发展，目前已有 300+ MLLM 基准测试集，开发者面临两大困境：

**选择困难**：面对海量 benchmark，不知道哪个最能揭示模型的强弱

**评估机制本身不可靠**：许多 benchmark 存在以下缺陷：
   - **Fallacy（谬误）**：题目或标注本身有误，反映的信息不可靠
   - **Difficulty（难度不足）**：题目太简单，几乎所有模型都能答对，无法提供有意义的区分
   - **Redundancy（冗余）**：仅凭部分信息（如纯文本不看图）就能答对，多出的模态是冗余的
   - **Diversity（多样性不足）**：多个样本问的是同类问题，导致信息重叠

核心问题：从未有人系统地评估过这些评估基准本身——benchmark 作为评估机制，自身需要被评估。

## 方法详解

### 整体框架

基于信息论建立"信息密度"的理论基础，将抽象的"洞察力"分解为四个可量化维度的乘积：

$$E(I) \propto (1 - D_{fal}) \cdot D_{dif} \cdot (1 - D_{red}) \cdot D_{div}$$

其中 $D_{fal}$ 是谬误率，$D_{dif}$ 是难度，$D_{red}$ 是冗余度，$D_{div}$ 是多样性。信息密度越高，benchmark 对 MLLM 开发者越有价值。

构建三级评估范式：
- **Human Eval**（成本最高，精度最高）：人类专家标注，作为 ground truth
- **Model Eval**（中等成本）：使用 MLLM 推理结果反映数据质量
- **Data Eval**（成本最低）：直接分析数据本身的特征，无需模型推理

### 关键设计

1. **Difficulty 评估**：

    - **Model Eval**：用 GPT-4o、InternVL-2.5、QwenVL-2.5 三模型投票，定义 Junior（至少一个错）、Extreme（全错）、Ambiguity（最佳和备选答案在模型间交叉）三个子维度
    - $D_{dif} = P(Q_{jun}) + P(Q_{amb})$
    - **Data Eval**：从图像结构复杂度（2D 拉普拉斯算子）、文本语法深度（语法树）、选项语义距离（CLIP 距离）、关注区域大小（语法根节点熵）四个特征拟合 Model Eval 结果

2. **Fallacy 评估**（仅 Human Eval）：

    - 在 Difficulty 筛出的困难样本中，人类专家标注三种谬误：Question（问题本身有误）、Annotation（标注有误但有其他正确选项）、Ambiguity（多选项均合理）
    - $D_{fal} = P((Q_{que} + Q_{ano} + Q_{amb}) | D_{dif}=1)$

3. **Redundancy 评估**：

    - **Model Eval**：分别去掉图像/文本，让模型推理，若仍能答对则说明被去掉的模态冗余
    - $D_{red} = \frac{w_{img} \cdot \mathrm{Acc}(\overline{I_{img}}) + w_{txt} \cdot \mathrm{Acc}(\overline{I_{txt}})}{w_{img} + w_{txt}}$
    - 使用 QwenVL-2.5 推理（其他模型会拒绝回答）

4. **Diversity 评估**：

    - **Model Eval**：用 CLIP 编码器对图像/文本样本做聚类和去重，剩余样本比例即多样性
    - $D_{div} = \frac{w_{img} \cdot \frac{\#(\mathrm{SIM}(I_{img}))}{\#(I_{img})} + w_{txt} \cdot \frac{\#(\mathrm{SIM}(I_{txt}))}{\#(I_{txt})}}{w_{img} + w_{txt}}$
    - **Data Eval**：图像用 5 个低层特征（亮度、对比度、色彩、模糊、纹理）的分布方差；文本用 10 种疑问词类型的覆盖率

### 损失函数 / 训练策略

本文是评估方法论，不涉及模型训练。Data Eval 中使用线性回归拟合 Model Eval 结果。

## 实验关键数据

### 主实验（19 个 Benchmark 的信息密度对比）

| Benchmark | Fallacy↓ | Difficulty↑ | Redundancy↓ | Diversity↑ | 发布时间 |
|-----------|----------|------------|-------------|-----------|---------|
| MMStar | **0.135** | **0.546** | **0.054** | 0.827 | Mar-2024 |
| Q-Bench | 0.280 | 0.373 | 0.175 | **0.951** | Sep-2023 |
| RealWorldQA | 0.247 | 0.379 | 0.113 | 0.756 | Apr-2024 |
| HallusionBench | 0.269 | 0.465 | 0.312 | 0.191 | Oct-2023 |
| POPE | 0.557 | 0.119 | 0.562 | 0.383 | May-2023 |
| MME | 0.526 | 0.206 | 0.133 | 0.842 | Jun-2023 |
| A-okvqa | 0.597 | 0.157 | 0.243 | 0.882 | Jun-2022 |

### Model/Data Eval 与 Human Eval 的相关性

| 维度 | Model Eval Pearson r | Data Eval Pearson r |
|------|---------------------|---------------------|
| Difficulty | >0.7 | >0.7 |
| Redundancy | >0.7 | - |
| Diversity (Image) | >0.8 | >0.7 |
| Diversity (Text) | >0.7 | >0.7 |

### 关键发现

- **MMStar 综合表现最佳**：谬误率最低（0.135）、难度最高（0.546）、冗余度最低（0.054），是当前信息密度最高的 benchmark
- **早期 benchmark 普遍存在问题**：POPE（2023.5）冗余度高达 0.562，多样性仅 0.383；A-okvqa（2022.6）谬误率 0.597
- **新 benchmark 有改善但仍有空间**：2024 年的 benchmark 在各维度上总体优于早期版本，但没有一个在四个维度上都达到最优
- **Model/Data Eval 与 Human Eval 相关系数均超 0.7**，验证了自动化评估流水线的合理性

## 亮点与洞察

- **元评估视角新颖**："评估评估机制本身"是一个被忽视但极其重要的方向，该工作首次系统化
- **信息论基础扎实**：将四个维度统一在信息熵框架下推导，不是简单的 ad-hoc 指标堆叠
- **三级评估的实用设计**：从全人工到全自动逐级降低成本，benchmark 开发者可以按需选用
- **Redundancy 的发现很有价值**：揭示了许多 benchmark 的"图文多模态"是假的——仅凭文本就能答对

## 局限与展望

- Fallacy 维度只能靠人工标注，无法自动化，限制了大规模应用
- 仅评估了 MCQ 格式的 benchmark，VQA 开放式回答的 benchmark 尚未覆盖
- Redundancy 的 Model Eval 仅用了 QwenVL-2.5 一个模型（因为其他模型会拒绝回答不完整输入），可能存在偏差
- 未考虑 benchmark 的时效性和数据污染问题（训练数据泄漏）

## 相关工作与启发

- 信息密度框架可以作为新 benchmark 开发的设计准则，在发布前先自查四个维度
- Redundancy 的检测方法（去掉某模态看能否答对）是一种通用的多模态数据质量检查手段
- 对 MLLM 开发者：优先使用 MMStar、Q-Bench 等高信息密度 benchmark

## 评分

- 新颖性：⭐⭐⭐⭐⭐ （首次系统性地"评估 benchmark"，开辟新方向）
- 技术深度：⭐⭐⭐⭐ （信息论推导+三级自动化流水线设计）
- 实验充分度：⭐⭐⭐⭐ （19 个 benchmark、17912 样本、多维度对比）
- 实用价值：⭐⭐⭐⭐⭐ （直接指导 benchmark 选择和开发）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Redundancy Principles for MLLMs Benchmarks](../../ACL2025/multimodal_vlm/redundancy_principles_for_mllms_benchmarks.md)
- [\[ICCV 2025\] Calibrating MLLM-as-a-Judge via Multimodal Bayesian Prompt Ensembles](calibrating_mllm-as-a-judge_via_multimodal_bayesian_prompt_ensembles.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[ICCV 2025\] Pi-GPS: Enhancing Geometry Problem Solving by Unleashing the Power of Diagrammatic Information](pi-gps_enhancing_geometry_problem_solving_by_unleashing_the_power_of_diagrammati.md)
- [\[ICCV 2025\] PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting](physsplat_efficient_physics_simulation_for_3d_scenes_via_mllm-guided_gaussian_sp.md)

</div>

<!-- RELATED:END -->
