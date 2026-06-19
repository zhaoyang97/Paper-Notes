---
title: >-
  [论文解读] MTL-UE: Learning to Learn Nothing for Multi-Task Learning
description: >-
  [ICML 2025][自监督学习][不可学习样本] MTL-UE是首个针对多任务学习的不可学习样本生成框架，通过编码器-解码器架构注入任务特定的类别先验嵌入来降低虚假特征的类内方差，配合任务内/间嵌入余弦正则化增大类间距离和减少冗余，在CelebA(40任务)上将MTL模型准确率从91%降至59%，在4个数据集、3种基础UE方法、5种backbone和5种MTL策略上一致有效。
tags:
  - "ICML 2025"
  - "自监督学习"
  - "不可学习样本"
  - "多任务学习"
  - "数据投毒"
  - "隐私保护"
  - "类别嵌入正则化"
---

# MTL-UE: Learning to Learn Nothing for Multi-Task Learning

**会议**: ICML 2025  
**arXiv**: [2505.05279](https://arxiv.org/abs/2505.05279)  
**代码**: 无  
**领域**: 自监督  
**关键词**: 不可学习样本, 多任务学习, 数据投毒, 隐私保护, 类别嵌入正则化

## 一句话总结

MTL-UE是首个针对多任务学习的不可学习样本生成框架，通过编码器-解码器架构注入任务特定的类别先验嵌入来降低虚假特征的类内方差，配合任务内/间嵌入余弦正则化增大类间距离和减少冗余，在CelebA(40任务)上将MTL模型准确率从91%降至59%，在4个数据集、3种基础UE方法、5种backbone和5种MTL策略上一致有效。

## 研究背景与动机

**领域现状**：不可学习样本（Unlearnable Examples, UE）通过向训练数据添加不可见扰动（$\|\delta\|_\infty \leq 8/255$），使模型学到虚假特征（shortcut）而非真实特征，从而保护数据免受未授权训练。现有方法分为：代理依赖型（EM、TAP、SEP）需要代理模型逐样本优化扰动；无代理型（LSP、AR）使用预定义类别模式。

**现有痛点**：所有现有UE方法仅针对单任务学习（STL），而实际场景中数据集越来越多地面向多任务学习（MTL）——如 CelebA 的 40 个面部属性分类、NYUv2 的语义分割+深度估计+法线估计。MTL 数据的 UE 面临独特挑战：(1) 需在一个扰动中同时编码多个任务的虚假特征；(2) 任务数增多导致标签组合指数爆炸（$\prod_k C_k$ 种组合），无代理方法的patch面积不断缩小失效。

**核心矛盾**：代理依赖型UE逐样本独立优化→类内方差不可控→虚假特征不一致→模型难学到shortcut。无代理patch-based方法随任务增多→patch变小→表达能力下降→攻击失效。两者在MTL下系统性失败。

**本文目标** (1) 设计统一框架在一个扰动中融合多任务虚假特征；(2) 有效降低类内方差提升攻击效果；(3) 同时对MTL和STL模型有效。

**切入角度**：关键观察——类内方差是UE有效性的决定因素。作者发现patch-based AR的类内标准差最低（20.59），攻击最强；而EM/TAP/SEP的类内标准差高达82-103，接近干净数据，攻击无效。

**核心 idea**：用可学习的任务-类别嵌入注入生成器替代像素级扰动搜索，搜索空间缩减使同类样本的虚假特征自然一致。

## 方法详解

### 整体框架

给定多任务数据集 $\mathcal{T}=\{(\mathbf{x}_i, \{y_i^k\}_{k=1}^K)\}_{i=1}^N$，MTL-UE 通过编码器-解码器网络和可学习类别嵌入生成扰动 $\delta_i$，将干净数据转为不可学习数据 $\mathcal{P}=\{(\mathbf{x}_i+\delta_i, \{y_i^k\}_{k=1}^K)\}$。扰动生成流程：编码器 $E$ 将输入映射为隐变量 $\mathbf{z}$，根据样本的 $K$ 个任务标签查找对应嵌入 $\{e_{y^k}^k\}_{k=1}^K$，将 $[\mathbf{z}, e_{y^1}^1, \ldots, e_{y^K}^K]$ 拼接后送入解码器 $D$ 生成扰动并clip到 $[-\epsilon, \epsilon]$。

### 关键设计

1. **编码器-解码器扰动生成器 + 类别嵌入注入**

    - 功能：生成与输入内容相关且携带任务标签先验信息的扰动
    - 核心思路：编码器 $E(\mathbf{x}; \phi_E)$ 提取输入隐表示 $\mathbf{z}$，为每个任务 $k$ 的每个类别 $c$ 维护可学习嵌入 $e_c^k \in \mathbb{R}^{d_e}$。根据样本标签组合查找嵌入，拼接后送入解码器：$\delta = \text{Clip}(D([\mathbf{z}, e_{y^1}^1, \ldots, e_{y^K}^K]; \phi_D), -\epsilon, \epsilon)$。嵌入总数为 $\sum_k C_k$（线性增长）而非 $\prod_k C_k$（指数增长），避免了组合爆炸
    - 设计动机：搜索空间从像素级 $\|\delta\|_\infty \leq \epsilon$ 缩减到解码器输出空间——同类样本被同一嵌入引导，类内方差自然降低；生成器以全数据集训练捕获全局模式，优于逐样本优化；拼接方式自然融合多任务信息到统一扰动

2. **任务内嵌入正则化 (Intra-task ER)**

    - 功能：增大同任务内不同类嵌入的方向差异
    - 核心思路：最小化同任务内所有类别嵌入对的余弦相似度：$\mathcal{L}_{Intra} = \frac{2}{\sum_k C_k(C_k-1)} \sum_k \sum_{m<n} \cos(e_m^k, e_n^k)$
    - 设计动机：虚假特征的类间距离越大，模型越容易学到shortcut。但直接增大L2距离可能被解码器rescale抵消，余弦相似度衡量方向差异不受缩放影响

3. **任务间嵌入正则化 (Inter-task ER)**

    - 功能：促进不同任务嵌入的几何独立性
    - 核心思路：最小化不同任务嵌入间余弦相似度的绝对值：$\mathcal{L}_{Inter} = \frac{1}{\sum_{k<l} C_k C_l} \sum_{k<l} \sum_m \sum_n |\cos(e_m^k, e_n^l)|$
    - 设计动机：绝对值确保正/负相关都被惩罚，目标是使不同任务嵌入空间正交——减少冗余（每个嵌入携带独特信息）、降低耦合（解码器独立处理每个任务）、提高可解释性

### 稠密预测扩展

对NYUv2等稠密预测数据集，将类别嵌入替换为嵌入模块 $\mathcal{E}^k(\mathbf{y}^k; \phi_{\mathcal{E}^k})$ 将稠密标签映射为嵌入。由于稠密预测虚假特征冗余性小，不使用嵌入正则化。

### 总损失与训练

$$\mathcal{L} = \mathcal{L}_b(F'_{MTL}, \mathbf{x}+\delta, \mathbf{y}) + \lambda_1 \mathcal{L}_{Intra} + \lambda_2 \mathcal{L}_{Inter}$$

其中 $\mathcal{L}_b$ 是基础UE方法的损失。如基础方法需要训练代理模型则交替优化。框架即插即用，可接入EM/TAP/SEP等任意代理依赖型UE方法。

## 实验关键数据

### CelebA（40个二分类任务，ResNet-18）

| 方法 | MTL Avg Acc↓ | STL Avg Acc↓ |
|------|-------------|-------------|
| Clean（无攻击） | 91.11 | 90.35 |
| AR (Patch) | 73.12 | 84.41 |
| EM | 75.66 | 89.91 |
| TAP | 85.24 | 87.00 |
| SEP | 84.25 | 89.91 |
| **MTL-UE + EM** | **74.38** | **74.26** |
| **MTL-UE + TAP** | **59.51** | **68.65** |
| **MTL-UE + SEP** | **63.77** | **72.51** |

### 跨数据集验证

| 数据集 | 任务类型 | Clean | MTL-UE效果 |
|--------|---------|-------|-----------|
| ChestX-ray14 | 14个疾病(AUC-ROC) | 0.7577 | 0.4813 |
| UTKFace | 年龄/种族/性别 | 78.97 | 25.84(MTL) |
| NYUv2 | 分割/深度/法线 | - | 所有任务退化 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| w/o Intra-ER | 攻击退化 | 类间方向差异减小 |
| w/o Inter-ER | 攻击退化 | 嵌入冗余增加 |
| w/o 生成器（直接EM） | 75.66→74.38 | 搜索空间缩减是关键 |
| 跨5种backbone | 一致有效 | 泛化性强 |
| 跨5种MTL策略 | 一致有效 | 策略无关性 |

### 关键发现

- **MTL-UE+TAP在CelebA上将MTL准确率从91.11%降至59.51%**，远优于所有基线方法
- 类内标准差分析（Tab.1）：AR(Patch)最大值20.59 → MTL-UE更低；EM/TAP/SEP最大值82-103 → 接近Clean数据（91.97），解释了其攻击无效
- STL模型更难被攻击（MTL共享表示使其更容易学shortcut），但MTL-UE首次对STL有效攻击（STL Avg从90%+降至68-74%）
- 支持部分保护：可选择性使部分任务不可学习，其余正常可学习

## 亮点与洞察

- **搜索空间缩减的核心洞察**：从像素级 $\epsilon$-ball 到解码器输出空间的缩减同时解决了类内方差和多任务融合两个问题。嵌入作为"类别信号"注入的方式优雅地绕开了标签组合爆炸——$\sum_k C_k$ 而非 $\prod_k C_k$
- **从baseline分析到方法设计的完整研究路径**：先做系统的baseline benchmark揭示问题本质（类内方差），再针对性地设计解决方案，最后大规模验证——提供了MTL数据保护的完整解决方案

## 局限与展望

- 需要代理MTL模型训练，攻击者需假定知道目标模型架构
- 对抗训练（adversarial training）等防御下的鲁棒性未充分评估
- 嵌入维度 $d_e$ 对性能影响需更系统分析
- 编码器-解码器容量可能限制高分辨率图像效果
- 论文分类为AI安全/数据保护更合适

## 相关工作与启发

- **vs EM (Huang et al., 2021)**：EM逐样本优化类内方差不可控，MTL-UE通过生成器+嵌入注入大幅降低
- **vs AR Patch (Sandoval-Segura et al., 2022)**：AR Patch在少任务时有效但随任务增多patch变小失效，MTL-UE的嵌入注入不受任务数限制
- **vs TAP (Fowl et al., 2021)**：TAP类内方差高(Tab.1最大值103)导致攻击弱(85.24%)，MTL-UE+TAP将其提升到59.51%
- **启发**：生成式方法（用网络生成扰动而非直接优化像素值）可推广到更多可控数据扰动场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个MTL-UE框架，搜索空间缩减+嵌入正则化设计精巧，问题定义有开创性
- 实验充分度: ⭐⭐⭐⭐⭐ 4数据集×3方法×5backbone×5策略的超大规模实验矩阵
- 写作质量: ⭐⭐⭐⭐ baseline分析→动机→方法的逻辑链清晰，notation略重
- 价值: ⭐⭐⭐⭐ 大模型数据爬取背景下MTL数据保护是真实需求

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](../../NeurIPS2025/self_supervised/mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)
- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](../../NeurIPS2025/self_supervised/soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [\[CVPR 2025\] Task-Agnostic Guided Feature Expansion for Class-Incremental Learning](../../CVPR2025/self_supervised/task-agnostic_guided_feature_expansion_for_class-incremental_learning.md)
- [\[ECCV 2024\] Adaptive Multi-head Contrastive Learning](../../ECCV2024/self_supervised/adaptive_multihead_contrastive_learning.md)
- [\[ICML 2025\] Update Your Transformer to the Latest Release: Re-Basin of Task Vectors](update_your_transformer_to_the_latest_release_re-basin_of_task_vectors.md)

</div>

<!-- RELATED:END -->
