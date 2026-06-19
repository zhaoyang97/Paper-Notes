---
title: >-
  [论文解读] Leveraging Partial SMILES Validation Scheme for Enhanced Drug Design in Reinforcement Learning Frameworks
description: >-
  [ICML2025][计算生物][SMILES分子生成] 提出 PSV-PPO 算法，在自回归 SMILES 分子生成的每一步引入部分 SMILES 验证（PSV）真值表，实时惩罚无效 token，在保持分子有效性的同时增强化学空间探索能力。 基于 SMILES 的分子生成已成为药物发现中的主流方法。大语言模型（LLM）结合…
tags:
  - "ICML2025"
  - "计算生物"
  - "SMILES分子生成"
  - "强化学习"
  - "PPO"
  - "部分验证"
  - "药物发现"
  - "灾难性遗忘"
---

# Leveraging Partial SMILES Validation Scheme for Enhanced Drug Design in Reinforcement Learning Frameworks

**会议**: ICML2025  
**arXiv**: [2505.00530](https://arxiv.org/abs/2505.00530)  
**代码**: 待确认  
**领域**: 计算生物  
**关键词**: SMILES分子生成, 强化学习, PPO, 部分验证, 药物发现, 灾难性遗忘

## 一句话总结

提出 PSV-PPO 算法，在自回归 SMILES 分子生成的每一步引入部分 SMILES 验证（PSV）真值表，实时惩罚无效 token，在保持分子有效性的同时增强化学空间探索能力。

## 研究背景与动机

基于 SMILES 的分子生成已成为药物发现中的主流方法。大语言模型（LLM）结合强化学习（RL）可以微调生成模型以优化目标分子属性，但面临一个核心难题——**灾难性遗忘**：预训练阶段分子有效率可达 99% 以上，但 RL 微调后有效率急剧下降。

现有方法的不足：

- **REINVENT**：使用先验模型作为锚点保留预训练知识，但限制了探索能力，多样性不足
- **SELFIES / Grammar-VAE**：在表示层面强制有效性约束，但研究表明 SMILES 方法在分子属性优化和多样性方面通常优于它们
- **标准 PPO**：熵驱动的探索机制在 SMILES 生成中表现不稳定——熵过高导致梯度爆炸，熵过低导致模式坍塌（mode collapse）
- **稀疏奖励问题**：分子有效性和属性评分仅在完整 SMILES 串生成后才能评估，模型缺乏中间反馈

## 方法详解

### 核心思路：PSV 真值表

PSV-PPO 的核心创新是**部分 SMILES 验证（Partial SMILES Validation）真值表**。在自回归生成的每一步，PSV 表系统地评估所有候选 token：如果某个 token 会导致当前部分 SMILES 串变为无效结构，则立即标记并惩罚。

PSV 表执行三种检查：
1. **语法合规性**：确保 SMILES 串遵循语法规范
2. **芳香性处理**：检查芳香体系是否可正确 Kekulé 化
3. **价态验证**：确保每个原子的价态在化学合理范围内

### PSV-PPO 损失函数

PSV-PPO 在标准 PPO 的基础上引入四个新损失项，总损失为六项加权组合：

$$Loss = L^{\text{CLIP}}(\theta) + L^{\text{Value}}(\theta) + L^{\text{ENTROPY}}_{PSV}(\theta) + L^{\text{HD}}_{PSV}(\theta) + L^{\text{TPC}}_{PSV}(\theta) + L^{\text{GPS}}_{PSV}(\theta)$$

#### 1. PSV 驱动的熵损失

仅对 PSV 表验证为有效的 token 集合 $D_{PSV}$ 计算熵，并用 $\log(\text{len}(D_{PSV}))$ 归一化，防止模型偏向有效 token 集较大的动作（如非芳香碳 "C"）：

$$L^{\text{ENTROPY}}_{PSV}(\theta) = -\beta \mathbb{E}_t \left[ \sum_{a \in D_{PSV}} \frac{\pi_\theta(a|s_t) \log \pi_\theta(a|s_t)}{\log(\text{len}(D_{PSV}))} \right]$$

#### 2. PSV 驱动的 Hellinger 距离损失

传统 KL 散度无法处理 PSV 过滤后出现的零概率情况，因此替换为 Hellinger 距离来衡量当前策略与 PSV 过滤后先验策略的差距：

$$L^{\text{HD}}_{PSV}(\theta) = \mathbb{E}_t \left[ \text{HD} \left[ \pi_{\theta_{\text{old\_PSV}}}(\cdot|s_t) \| \pi_\theta(\cdot|s_t) \right] \right]$$

#### 3. TPC Loss（Token 概率控制损失）

动态惩罚概率过高的 token，防止模式坍塌。

#### 4. GPS Loss（全局概率稳定损失）

额外的正则化项，在模型无法发现更高分分子时维持生成多样性。

### 训练流程

1. 先验模型生成分子结构及其概率分布
2. 并行计算奖励和 PSV 真值表（最小化计算开销）
3. 将带评分的分子存入经验回放池
4. 从回放池采样，当前模型重新生成概率分布
5. 计算六项损失并反向传播更新参数

## 实验关键数据

### GuacaMol Benchmark

| 任务 | SMILES GA | SMILES LSTM | Reinvent | MolRL-MGPT | **PSV-PPO** |
|------|-----------|-------------|----------|------------|-------------|
| C11H24 | 0.829 | 0.993 | 0.999 | 1.000 | **1.000** |
| C9H10N2O2P2Cl | 0.889 | 0.879 | 0.877 | 0.939 | **1.000** |
| Osimertinib MPO | 0.886 | 0.907 | 0.889 | 0.977 | 0.951 |
| Fexofenadine MPO | 0.931 | 0.959 | 1.000 | 1.000 | **1.000** |
| Perindopril MPO | 0.661 | 0.808 | 0.764 | 0.810 | **0.849** |
| Amlodipine MPO | 0.722 | 0.894 | 0.888 | 0.906 | **0.908** |
| Valsartan SMARTS | 0.552 | 0.978 | 0.095 | 0.997 | **0.999** |

### PMO Benchmark（AUC-Top10）

| 任务 | REINVENT | LSTM HC | LSTM PPO | **LSTM PSV-PPO** |
|------|----------|---------|----------|------------------|
| albuterol_similarity | 0.882 | 0.719 | 0.527 | **0.761** |
| drd2 | 0.945 | 0.919 | 0.883 | **0.959** |
| gsk3b | 0.865 | 0.839 | 0.794 | **0.869** |
| isomers_c9h10n2o2pf2cl | 0.642 | 0.342 | 0.608 | **0.652** |
| celecoxib_rediscovery | 0.713 | 0.539 | 0.532 | **0.612** |
| amlodipine_mpo | 0.635 | 0.593 | 0.587 | **0.647** |

**关键发现**：LSTM PSV-PPO 在所有任务上一致优于 LSTM PPO，证明 PSV 验证机制的有效性。

### 消融实验

- **去掉 PSV 表**（PSV-PPO_WO_PSV）：分子有效率显著下降，验证了 PSV 对维持有效性的必要性
- **去掉 GPS/TPC 损失**（PSV-PPO_WO_PL）：经验回放池和生成阶段的重复率上升，确认了这两个损失项对防止模式坍塌的作用

### 分子对接实验

在 fa7 蛋白靶点上，PSV-PPO 与 HC 和标准 PPO 相比，在 Top-1/10/100 和多样性指标上均表现出竞争力。

## 亮点与洞察

- **逐步验证 vs 生成后验证**：PSV 在每个自回归步骤执行验证，而非等完整分子生成后再检查，实现了即时反馈，大幅减少无效分子的产生
- **Hellinger 距离替代 KL 散度**：优雅地解决了 PSV 过滤引入零概率时 KL 散度不可计算的问题
- **归一化熵损失**：通过 $\log(\text{len}(D_{PSV}))$ 归一化避免了模型偏向大有效集 token 的问题
- **框架可扩展性**：PSV 框架可以扩展到注入其他领域知识（如合成可行性、毒性约束等），不局限于有效性验证
- **兼容性强**：基于标准 LSTM + PPO 架构，可方便地集成到现有 SMILES 生成管线中

## 局限与展望

- PSV 验证只保证部分 SMILES 的局部有效性，**不能保证最终完整分子一定有效**（但显著提高了有效率）
- 实验主要基于 LSTM 预训练模型，未验证在 Transformer 等更强架构上的效果
- PSV 表的计算引入了额外开销，虽然论文声称并行计算可最小化，但对超长序列的可扩展性未深入讨论
- 在 GuacaMol 的部分任务（如 Median molecules 2、Sitagliptin MPO）上未超越最佳基线
- 分子对接实验仅在单一蛋白靶点（fa7）上验证，生物相关性的广泛性需进一步考察

## 相关工作与启发

- **REINVENT**（Blaschke et al., 2020）：用先验模型锚定预训练知识，但限制探索
- **SELFIES**（Krenn et al., 2022）：表示层面强制有效性，但优化能力受限
- **PPO**（Schulman et al., 2017）：基础 RL 算法，PSV-PPO 在此基础上扩展
- **partialsmiles**（O'Boyle, 2024）：提供实时 SMILES 语法验证的工具包，是 PSV 表的基础
- 本文的"逐步验证 + 域知识注入"思路可推广到其他序列生成任务中的约束满足问题

## 评分

- 新颖性: ⭐⭐⭐⭐ — PSV 真值表 + 多损失项设计有创新性
- 实验充分度: ⭐⭐⭐⭐ — PMO、GuacaMol、对接、消融均覆盖，但靶点单一
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式完整，图表丰富
- 价值: ⭐⭐⭐⭐ — 灾难性遗忘是 RL 分子生成的真实痛点，方案实用且可扩展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improved Off-policy Reinforcement Learning in Biological Sequence Design](improved_off-policy_reinforcement_learning_in_biological_sequence_design.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](../../NeurIPS2025/computational_biology/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[ICML 2025\] Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule](piloting_structure-based_drug_design_via_modality-specific_optimal_schedule.md)
- [\[NeurIPS 2025\] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](../../NeurIPS2025/computational_biology/pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)
- [\[NeurIPS 2025\] GFlowNets for Learning Better Drug-Drug Interaction Representations](../../NeurIPS2025/computational_biology/gflownets_for_learning_better_drug-drug_interaction_representations.md)

</div>

<!-- RELATED:END -->
