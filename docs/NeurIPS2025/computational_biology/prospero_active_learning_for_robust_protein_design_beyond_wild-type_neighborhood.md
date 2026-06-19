---
title: >-
  [论文解读] PROSPERO: Active Learning for Robust Protein Design Beyond Wild-Type Neighborhood
description: >-
  [NeurIPS 2025][计算生物][蛋白质设计] 提出 ProSpero，一个主动学习框架，通过冻结的预训练生成模型（EvoDiff）在代理模型引导下的推理时采样、针对性掩码策略和生物约束的 SMC 采样，在代理模型可能失配的条件下仍能发现高适应性且新颖的蛋白质序列。 蛋白质序列设计面临核心矛盾：需要在野生型附近以外的…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "蛋白质设计"
  - "主动学习"
  - "序列蒙特卡洛"
  - "预训练生成模型引导"
  - "代理模型鲁棒性"
---

# PROSPERO: Active Learning for Robust Protein Design Beyond Wild-Type Neighborhood

**会议**: NeurIPS 2025  
**arXiv**: [2505.22494](https://arxiv.org/abs/2505.22494)  
**代码**: [GitHub](https://github.com/szczurek-lab/ProSpero)  
**领域**: 计算生物  
**关键词**: 蛋白质设计, 主动学习, 序列蒙特卡洛, 预训练生成模型引导, 代理模型鲁棒性

## 一句话总结

提出 ProSpero，一个主动学习框架，通过冻结的预训练生成模型（EvoDiff）在代理模型引导下的推理时采样、针对性掩码策略和生物约束的 SMC 采样，在代理模型可能失配的条件下仍能发现高适应性且新颖的蛋白质序列。

## 研究背景与动机

蛋白质序列设计面临核心矛盾：需要在野生型附近以外的新颖区域探索高适应性序列，但离开训练分布后代理模型的预测精度急剧下降（代理失配问题），同时远离野生型的序列容易丧失生物学合理性。现有方法各有缺陷：PEX 局限于局部突变；GFN 等全局探索方法受代理失配影响严重；GFN-AL-δCS 的随机掩码可能破坏保守残基。预训练生成模型天然编码生物先验，但在迭代优化中需要反复微调，不够实用。

## 方法详解

### 整体框架

ProSpero 遵循主动学习循环：每轮 (1) 在当前数据集上训练代理模型 $f_\theta$；(2) 通过针对性掩码识别并掩盖适应性相关残基；(3) 用 EvoDiff 在代理引导下通过受约束的 SMC 采样生成新候选序列；(4) 交由 Oracle 评估并更新数据集。核心创新是生成模型始终冻结，仅通过推理时引导整合代理反馈。

### 关键设计

1. **针对性掩码 (Targeted Masking)**: 受丙氨酸扫描启发，批量生成突变序列（随机位置替换为丙氨酸），用代理模型的 UCB（均值+不确定性）打分。选择 UCB 最高的突变——这些位置虽被突变但未立即降低适应性且具有不确定性，表明是值得探索的功能相关残基。将这些位置替换为 [MASK] 标记，保留其余野生型残基。这避免了随机掩码可能破坏关键保守残基的问题。

2. **生物约束的 SMC 采样 (Biologically-Constrained SMC)**: 为解决代理失配问题，利用氨基酸电荷类别作为显式生物先验约束采样。将掩码位置按野生型残基的电荷分为正电荷（R,K,H）、负电荷（D,E）和中性三组，采样时仅从同类电荷的氨基酸中提议。SMC 流程逐残基执行：(i) 从约束基模型 $\mathcal{P}_{RAA}$ 提议氨基酸；(ii) 用基模型全补全后由代理 UCB 打分，除以无约束模型困惑度作为重要性权重；(iii) 按归一化权重重采样，淘汰低质量序列。

3. **推理时引导的主动学习闭环**: 生成模型（EvoDiff-OADM）始终冻结不动，仅通过 SMC 的权重机制引入代理模型信号。每轮主动学习后代理在更多数据上更新，扩展其有效支撑范围，从而为后续轮次提供更可靠的引导信号。

### 损失函数 / 训练策略

代理模型（三个 1D CNN 集成）使用 MSE 损失训练，所有基线共享相同架构以确保公平比较。序列选择基于 UCB 获取函数 $f_\theta(x) = \mu_\theta(x) + k \cdot \sigma_\theta(x)$，掩码阶段 $k=1$（偏重探索），SMC 采样阶段 $k=0.1$（偏重利用）。每轮 $N=10$，每轮评估 $K=128$ 个序列，SMC 批大小 $B=256$。丛氨酸扫描批数 $S=16$，短序列（$L \approx 100$）突变 3-10 个位点，长序列突变 5-15 个。SMC 的最后 $n_{keep}=10$ 步 rollout 也参与候选选择。

## 实验关键数据

### 主实验

| 方法 | AMIE | TEM | E4B | AAV | GFP | UBE2I | LGK | 第一名次数 |
|------|------|-----|-----|-----|-----|-------|-----|-----------|
| ProSpero | 0.246 | 1.231 | **8.114** | **0.720** | **3.617** | **2.993** | **0.043** | **5/8** |
| PEX | **0.248** | **1.232** | 8.099 | 0.665 | 3.603 | 2.991 | 0.037 | 3/8 |
| AdaLead | 0.235 | 1.228 | 8.034 | 0.683 | 3.581 | 2.985 | 0.038 | 0/8 |
| GFN-AL-δCS | 0.203 | 0.701 | 7.930 | 0.686 | 3.589 | 2.984 | 0.033 | 0/8 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 去掉电荷约束 | 适应性下降 | 无约束采样更易生成失配序列 |
| 随机掩码替代针对性掩码 | 适应性下降 | 可能破坏关键残基 |
| 去掉主动学习（单轮） | 显著下降 | 代理无法扩展有效范围 |
| 增加代理噪声 | ProSpero 最鲁棒 | 电荷约束提供额外安全网 |

### 关键发现

- ProSpero 和 PEX 是唯二在所有 8 个景观上均改善野生型适应性的方法
- 在新颖性（Hamming距离）上，ProSpero 生成的 top-100 序列平均比 PEX 更远离野生型
- ProSpero 在半数任务上比基线更早发现高适应性序列
- 在代理失配（协变量偏移和预测噪声）条件下，ProSpero 始终维持最佳或接近最佳表现
- Top-100 序列的多样性（平均成对 Hamming 距离）保持在合理范围
- 生物合理性分析显示电荷约束有效保持了序列的结构完整性
- 4 个基线方法在所有任务上均未能改善野生型，突显了全局探索的难度

## 亮点与洞察

- "冻结生成模型 + 推理时引导"的设计避免了昂贵的微调，且天然利用了预训练模型的生物先验
- 电荷类别约束采样是一个简洁优雅的生物学先验注入方式，比复杂的结构约束更实用
- SMC 中用困惑度修正约束采样偏差的技巧值得借鉴
- 针对性掩码（受丙氨酸扫描启发）巧妙地将实验生物学技术转化为计算策略

## 局限与展望

- 电荷分组较粗糙，可探索更精细的氨基酸理化性质分组（如体积、极性）
- 单残基逐步解码的 SMC 计算开销较大，可探索并行解码策略
- 仅用 TAPE Oracle 模拟湿实验，未在真实实验中验证
- 未考虑蛋白质结构信息，结合 AlphaFold 等结构预测可能进一步提升

## 相关工作与启发

- PEX 擅长局部优化但无法远程探索；ProSpero 在保持可靠性的同时扩大探索范围
- GFN-AL-δCS 的掩码策略是随机的，ProSpero 的针对性掩码更智能
- 与 Amin et al. 的 twisted SMC 方法相比，ProSpero 不需要在特定 clonal family 上训练
- 预训练+推理引导范式可推广到其他离散序列优化问题（如 DNA、RNA 设计）
- DyNaPPO 和 LatProtRL 等 RL 方法全局探索能力强但在多数任务上不如 ProSpero 稳健
- CbAS 和 GFN-AL 在 8 个任务中多数无法改善野生型，暴露了全局方法的脆弱性
- AdaLead 的贪心突变+重组策略性能稳健，但缺乏预训练模型的生物先验
- MLDE 直接用 ESM-2 提议候选，但不具备主动学习的迭代改进能力
- BO 方法在某些景观上表现出色但缺乏一致性
- EvoDiff 的 order-agnostic 特性使 SMC 的采样顺序设计更灵活

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将推理时引导+主动学习用于蛋白质设计，三个组件协同设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 8个蛋白任务、10个基线、代理失配鲁棒性分析全面
- 写作质量: ⭐⭐⭐⭐ 动机明确、方法描述清晰、消融分析详细
- 价值: ⭐⭐⭐⭐ 对数据高效蛋白质工程有实际推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Self Iterative Label Refinement via Robust Unlabeled Learning](self_iterative_label_refinement_via_robust_unlabeled_learning.md)
- [\[NeurIPS 2025\] Amortized Active Generation of Pareto Sets](amortized_active_generation_of_pareto_sets.md)
- [\[NeurIPS 2025\] Protein Design with Dynamic Protein Vocabulary](protein_design_with_dynamic_protein_vocabulary.md)
- [\[ICML 2025\] MF-LAL: Drug Compound Generation Using Multi-Fidelity Latent Space Active Learning](../../ICML2025/computational_biology/mf-lal_drug_compound_generation_using_multi-fidelity_latent_space_active_learnin.md)
- [\[ACL 2025\] Concept Bottleneck Language Models For Protein Design](../../ACL2025/computational_biology/concept_bottleneck_language_models_for_protein_design.md)

</div>

<!-- RELATED:END -->
