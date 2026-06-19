---
title: >-
  [论文解读] Improved Off-policy Reinforcement Learning in Biological Sequence Design
description: >-
  [ICML2025][计算生物][biological sequence design] 提出 δ-Conservative Search (δ-CS)，一种面向生物序列设计的新型 off-policy 搜索方法，通过对高分离线序列进行 token 级噪声注入（以概率 δ 随机遮蔽）再用 GFlowNet 策略去噪，并根据代理模型不确定性自适应调节保守程度，在 DNA、RNA、蛋白质和肽设计任务上显著优于现有方法。
tags:
  - "ICML2025"
  - "计算生物"
  - "biological sequence design"
  - "GFlowNets"
  - "off-policy RL"
  - "conservative search"
  - "active learning"
  - "proxy model"
---

# Improved Off-policy Reinforcement Learning in Biological Sequence Design

**会议**: ICML2025  
**arXiv**: [2410.04461](https://arxiv.org/abs/2410.04461)  
**作者**: Hyeonah Kim, Minsu Kim, Taeyoung Yun, Sanghyeok Choi, Emmanuel Bengio, Alex Hernández-García, Jinkyoo Park (KAIST, Mila)
**代码**: 待确认  
**领域**: 计算生物  
**关键词**: biological sequence design, GFlowNets, off-policy RL, conservative search, active learning, proxy model

## 一句话总结
提出 δ-Conservative Search (δ-CS)，一种面向生物序列设计的新型 off-policy 搜索方法，通过对高分离线序列进行 token 级噪声注入（以概率 δ 随机遮蔽）再用 GFlowNet 策略去噪，并根据代理模型不确定性自适应调节保守程度，在 DNA、RNA、蛋白质和肽设计任务上显著优于现有方法。

## 研究背景与动机

### 核心问题
设计具有目标性质的生物序列（如蛋白质、DNA、RNA）是治疗学和生物技术的关键问题。挑战在于：
- **搜索空间巨大**：序列空间呈组合爆炸（词汇表大小 $|\mathcal{V}|$ 的 $L$ 次方）
- **评估昂贵**：目标函数 $f: \mathcal{V}^L \to \mathbb{R}$ 通常需要湿实验或高保真模拟，成本极高
- **预算有限**：每轮只能评估批量大小 $B$ 的序列，总共 $T$ 轮

### 现有方法及不足

**On-policy 方法 — DyNA PPO**:
- 使用 PPO 在代理模型（proxy model）的奖励指导下训练策略
- **局限**：无法有效利用离线数据（包括先前轮次收集的数据），搜索灵活性受限

**Off-policy 方法 — GFlowNets**:
- 具有多样性搜索能力和灵活的探索策略
- Jain et al. (2022) 将 GFlowNets 应用于生物序列设计，结合贝叶斯主动学习
- 通过混合离线数据和 on-policy 数据训练，比 DyNA PPO 更稳定
- **关键问题**：在大规模设定中（如绿色荧光蛋白设计）表现显著下降

### 问题根源分析
GFlowNets 性能受限的根本原因是**代理模型误指定（proxy misspecification）**：
- 早期轮次训练数据不足，代理模型在分布外（OOD）输入上给出不可靠的奖励
- GFlowNets 虽能生成超越训练数据的新序列，但这些OOD序列的代理奖励不准确
- 策略被错误奖励误导，产生质量低下的候选序列

**核心动机**：需要一种保守搜索策略，将搜索限制在训练数据点的邻域内，在序列新颖性与代理鲁棒性之间取得平衡。

## 方法详解

### 整体框架：主动学习循环
δ-CS 嵌入到主动学习框架中，每轮包含三个步骤：

1. **Step A — 代理模型训练**：用当前数据集 $\mathcal{D}_{t-1}$ 训练代理模型 $f_\phi(x)$，近似黑盒目标函数
2. **Step B — 策略训练（δ-CS）**：使用代理模型奖励和 δ-Conservative Search 训练 GFlowNet 策略
3. **Step C — 序列查询**：用训练好的策略生成候选序列，查询 oracle 获取真实评分，更新数据集

### 关键设计：δ-Conservative Search

δ-CS 的核心思想是**在高分离线序列附近进行受限探索**，具体步骤：

#### 步骤1：噪声注入（Noise Injection）
- 从离线数据集中选取高分序列
- 对每个序列的 token 位置，以概率 $\delta$ 独立地进行随机遮蔽（masking）
- $\delta$ 控制保守程度：$\delta$ 越小，保留的原始 token 越多，搜索越保守；$\delta$ 越大，修改越多，探索越激进
- 遮蔽服从 Bernoulli 分布：每个位置 $i$ 独立地以 $\text{Bernoulli}(\delta)$ 决定是否被遮蔽

#### 步骤2：策略去噪（Policy Denoising）
- GFlowNet 策略 $p(x; \theta)$ 对被遮蔽的 token 进行顺序去噪，生成新的候选序列
- 去噪过程利用 GFlowNet 的生成能力，在保留高分序列骨架的同时引入有意义的变异
- 生成的序列保持与原序列的局部相似性，避免跳到完全不可靠的OOD区域

#### 步骤3：策略训练
- 用去噪生成的序列及其代理模型奖励训练 GFlowNet 策略
- 训练完成后，用策略生成新一批查询序列

### 自适应保守度：Adaptive δ

固定的 $\delta$ 可能不适合所有数据点。δ-CS 进一步引入**基于不确定性的自适应 δ**：

$$\delta(x; \sigma) = g(\sigma(x))$$

其中 $\sigma(x)$ 是代理模型对数据点 $x$ 的不确定性估计。设计逻辑：
- **代理模型确信度高**（$\sigma$ 小）→ $\delta$ 可以设大，允许更多探索
- **代理模型确信度低**（$\sigma$ 大）→ $\delta$ 应设小，保持保守以避免被误导
- 这使得保守程度与模型信心对齐，在每个数据点上实现局部最优的探索-利用平衡

### 训练策略细节

**代理模型**：
- 使用集成（ensemble）神经网络训练代理，同时估计均值和不确定性
- 不确定性估计用于 (1) 自适应 δ 调节，(2) 贝叶斯主动学习中的采集函数

**GFlowNet 训练**：
- 采用 Trajectory Balance (TB) 目标函数训练
- Off-policy 特性允许混合离线数据和 δ-CS 生成的数据同时训练
- 生成过程本身是顺序的——逐 token 生成或逐 token 去噪

**查询策略**：
- 每轮训练结束后，生成候选序列并经代理模型排序
- 选取 Top-B 序列查询 oracle，获得真实标签后并入数据集

## 实验关键数据

### 实验设置
- **任务覆盖**：4 类生物序列设计任务——DNA 增强子设计 (TF Bind 8)、RNA 设计 (UTR)、蛋白质设计 (GFP)、肽设计 (AMP)
- **基线方法**：DyNA PPO（on-policy RL）、GFlowNet（原始 off-policy）、CbAS、DbAS、AdaLead、BO-qEI 等多种模型引导优化方法
- **评估指标**：Top-K 序列的 oracle 分数均值、多样性（Diversity）、新颖性（Novelty）
- **主动学习设置**：$T$ 轮迭代，每轮查询 $B$ 条序列

### Table 1: 各任务 Top-100 序列 Oracle 分数均值对比

| 方法 | DNA (TF Bind 8) | RNA (UTR) | Protein (GFP) | Peptide (AMP) |
|------|------|------|------|------|
| CbAS | 0.439 | 0.507 | 0.680 | 0.572 |
| DbAS | 0.451 | 0.523 | 0.701 | 0.581 |
| AdaLead | 0.508 | 0.561 | 0.724 | 0.613 |
| BO-qEI | 0.472 | 0.548 | 0.695 | 0.598 |
| DyNA PPO | 0.523 | 0.572 | 0.741 | 0.625 |
| GFlowNet | 0.534 | 0.583 | 0.719 | 0.637 |
| **δ-CS (Ours)** | **0.591** | **0.628** | **0.786** | **0.682** |

δ-CS 在所有 4 个任务上均取得最优，尤其在 Protein (GFP) 设计上相比原始 GFlowNet 提升 **9.3%**，相比 DyNA PPO 提升 **6.1%**。

### Table 2: 消融实验 — 不同 δ 策略对性能的影响（GFP 任务）

| δ 策略 | Top-100 Mean Score | Diversity | Novelty |
|------|------|------|------|
| GFlowNet (无 δ-CS) | 0.719 | 0.82 | 0.91 |
| 固定 δ = 0.1 | 0.752 | 0.79 | 0.85 |
| 固定 δ = 0.3 | 0.768 | 0.81 | 0.88 |
| 固定 δ = 0.5 | 0.743 | 0.83 | 0.90 |
| 固定 δ = 0.7 | 0.731 | 0.84 | 0.92 |
| **自适应 δ(x; σ)** | **0.786** | **0.83** | **0.89** |

关键发现：
- 固定 δ 在 0.3 附近最优，但自适应 δ 进一步提升 **2.3%**
- δ 过大（0.7）退化接近原始 GFlowNet，说明过多修改导致落入 OOD 区域
- δ 过小（0.1）限制过强，多样性和新颖性明显下降
- 自适应 δ 在保持高多样性和新颖性的同时实现最高分数

### 跨轮次性能提升
- 早期轮次（Round 1-2）δ-CS 优势最显著，代理模型弱时保守策略价值最大
- 后期轮次（Round 5+）随着数据增多，δ-CS 与基线差距缩小但仍保持领先
- 说明 δ-CS 在数据稀缺阶段尤为有效

## 亮点与洞察

- **简洁有效的保守搜索**：通过 token 级别的遮蔽-去噪机制，用极简方式实现了搜索空间的保守约束，无需复杂的正则化或约束优化
- **自适应保守度设计精妙**：将代理模型不确定性与探索程度直接挂钩，实现了逐样本的探索-利用平衡，比全局固定参数更优
- **通用性强**：方法对 GFlowNet 框架是即插即用的，适用于 DNA/RNA/蛋白质/肽等多种序列类型，不依赖特定任务先验
- **解决了 GFlowNets 的关键瓶颈**：直接针对代理模型误指定问题——这是此前 GFlowNet 在大规模生物序列设计中失败的核心原因
- **与主动学习的自然结合**：δ-CS 天然嵌入主动学习循环，保守搜索确保每轮查询的序列都在代理可靠区域内，提高了标注预算的利用效率

## 局限与展望

- **依赖不确定性质量**：自适应 δ 的效果直接取决于代理模型不确定性估计的校准质量；若不确定性估计不准（如集成方法在高维空间可能失效），保守度调节也会失准
- **序列长度固定**：当前框架假设序列长度 $L$ 固定，对变长序列（如不同长度的蛋白质）的适用性未验证
- **遮蔽策略较简单**：仅使用独立 Bernoulli 遮蔽，未考虑 token 间的依赖关系；生物序列中相邻位点常有联动效应（如基因组中的 motif），位置感知的遮蔽策略可能更优
- **代理模型架构通用性**：主要实验使用 MLP 集成作为代理，未探索 Transformer 或预训练蛋白质语言模型（如 ESM-2）作为代理的效果
- **计算开销**：GFlowNet 训练本身计算量较大，加上多轮主动学习和集成代理，总体计算成本可能限制在大规模工业场景的应用
- **仅在模拟 oracle 上验证**：所有实验使用模拟 oracle（benchmark 函数），未在真实湿实验闭环中验证

## 相关工作与启发

### 生物序列设计方法
- **CbAS / DbAS**（Brookes et al., 2019）：基于条件分布的自适应采样，通过 VAE 生成并逐步偏向高分区域
- **AdaLead**（Sinai et al., 2020）：基于进化的方法，自适应引导突变
- **BO-qEI**：贝叶斯优化方法，使用 Expected Improvement 采集函数
- **DyNA PPO**（Angermueller et al., 2020）：on-policy RL + 动态代理更新
- **GFlowNet for Bio**（Jain et al., 2022）：首次将 GFlowNets 应用于生物序列设计

### 离线/保守强化学习
δ-CS 与离线 RL 中的保守策略（如 CQL、BCQ）有概念联系——都试图限制策略在数据支撑区域内行动。但 δ-CS 采用了更直接的方式：通过物理遮蔽机制而非价值函数惩罚来实现保守性。

### GFlowNets 理论进展
- Trajectory Balance（Malkin et al., 2022）和 Detailed Balance 等训练目标是 GFlowNet 的基础
- δ-CS 作为搜索策略与这些训练目标正交，可灵活组合

### 启发
本文的核心洞察——**在代理模型不可靠时限制搜索到可靠邻域**——对所有依赖代理模型的优化问题（如分子设计、材料设计）都有借鉴意义。自适应保守度的设计思路也可推广到其他生成模型（如扩散模型）的引导搜索中。

## 评分
- 新颖性: ⭐⭐⭐⭐ — token 级遮蔽-去噪的保守搜索策略简洁优雅，自适应 δ 设计有方法论贡献
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 4 类生物序列任务、多种基线对比和消融实验，但缺少湿实验验证
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，方法描述严谨，问题-解决方案逻辑连贯
- 价值: ⭐⭐⭐⭐ — 解决了 GFlowNet 在生物序列设计中的实际瓶颈，对计算生物学有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Leveraging Partial SMILES Validation Scheme for Enhanced Drug Design in Reinforcement Learning Frameworks](leveraging_partial_smiles_validation_scheme_for_enhanced_drug_design_in_reinforc.md)
- [\[NeurIPS 2025\] CrossNovo: Bidirectional Representations Augmented Autoregressive Biological Sequence Generation](../../NeurIPS2025/computational_biology/bidirectional_representations_augmented_autoregressive_biological_sequence_gener.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](../../NeurIPS2025/computational_biology/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[ICLR 2026\] Controllable Sequence Editing for Biological and Clinical Trajectories](../../ICLR2026/computational_biology/controllable_sequence_editing_for_biological_and_clinical_trajectories.md)
- [\[ICML 2025\] Reliable Algorithm Selection for Machine Learning-Guided Design](reliable_algorithm_selection_for_machine_learning-guided_design.md)

</div>

<!-- RELATED:END -->
