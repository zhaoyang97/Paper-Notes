# The Pareto Frontier of Resilient Jet Tagging

**会议**: NeurIPS 2025
**arXiv**: [2509.19431](https://arxiv.org/abs/2509.19431)
**代码**: [Zenodo](https://zenodo.org/) (数据集公开)
**领域**: Physics / ML for HEP
**关键词**: jet tagging, Pareto frontier, resilience, model dependence, quark/gluon discrimination

## 一句话总结
系统评估LHC射流标记任务中多种架构（DNN/PFN/EFN/ParT）的AUC-鲁棒性权衡，揭示更复杂模型虽AUC更高但对蒙特卡洛模型依赖性更强，构建Pareto前沿并通过案例研究证明低鲁棒性分类器即使校准后仍在下游参数估计中产生偏差。

## 研究背景与动机

1. **领域现状**：射流标记（jet tagging）是LHC数据分析的核心任务，基于Transformer和GNN的SOTA架构（ParT等）在AUC上大幅超越传统方法。
2. **现有痛点**：单一关注AUC会导致选择模型依赖性（model-dependent）强的架构——这些架构可能学到了模拟器的特异性而非真实物理。ATLAS已发现分类器对物理建模不确定性比探测器效应更敏感。
3. **核心矛盾**：当前社区以AUC作为唯一优化目标（"当度量成为目标，它就不再是好度量"），忽视了模型在不同蒙特卡洛生成器间的鲁棒性。
4. **本文要解决什么？** 量化AUC与鲁棒性（resilience）之间的权衡关系，并证明低鲁棒性模型会在实际物理分析中引入偏差。
5. **切入角度**：在Pythia-8训练、Herwig-7测试的框架下度量resilience = AUC百分比差异。
6. **核心idea一句话**：构建AUC vs resilience的Pareto前沿，证明复杂模型占据高AUC-低resilience角，而简单模型如EFN和Expert Features占据低AUC-高resilience角，知识蒸馏无法突破前沿。

## 方法详解

### 整体框架
Pythia-8生成训练数据 → 多种架构在q/g和top tagging任务上训练 → Pythia测试AUC + Herwig测试AUC → 计算resilience（AUC百分比差异）→ 构建Pareto前沿 → 案例研究验证下游影响。

### 关键设计

1. **多架构系统评估**：
   - 覆盖：Expert Features（angularities/multiplicities）、DNN（2-10层，1-300节点）、PFN/EFN（潜空间1-1024维）、Particle Transformer（注意力头2/4/8）
   - 输入统一为粒子级运动学信息（$p_T$, $\eta$, $\phi$），pT 500-550 GeV，无探测器模拟

2. **Resilience度量**：
   - Pythia和Herwig使用不同的parton shower和hadronization模型，代表物理建模的系统不确定性

3. **知识蒸馏实验**：
   - 师模型PFN，生模型各种DNN和EFN，最小化KL散度训练
   - 目标：尝试突破Pareto前沿

4. **案例研究：q/g混合比例估计**：
   - 选取Pareto前沿上两个PFN（小=高resilience，大=高AUC）
   - 从混合样本中通过似然比估计quark jet比例$\kappa$，用Pythia-Herwig重加权校准

## 实验关键数据

### 主实验 — Pareto前沿观察

| 架构类型 | AUC范围 (q/g) | Resilience范围 | 趋势 |
|---------|---------------|----------------|------|
| Expert Features | 较低 | 最低（最鲁棒） | 基于物理原理的特征最稳定 |
| EFN | 中等 | 中低 | IRC安全架构较稳定 |
| PFN | 中高 | 中高 | 潜空间维度驱动权衡 |
| ParT | 最高 | 最高（最不鲁棒） | 复杂注意力机制最model-dependent |

### 案例研究 — q/g混合比例$\kappa$估计

| 分类器 | 真值$\kappa$ | Pythia推断 | 校准后Herwig | 结论 |
|--------|-------------|-----------|-------------|------|
| Large PFN | 0.50 | 0.490±0.005 | 0.529±0.006 | 偏差 ✗ |
| Large PFN | 0.25 | 0.253±0.005 | 0.305±0.006 | 偏差 ✗ |
| Small PFN | 0.50 | 0.504±0.013 | 0.478±0.017 | 无偏 ✓ |
| Small PFN | 0.25 | 0.258±0.013 | 0.268±0.013 | 无偏 ✓ |

### 关键发现
- **Pareto前沿清晰存在**：模型复杂度是沿前沿移动的主要驱动力；top tagging中不同架构形成垂直柱状分布
- **知识蒸馏无法突破前沿**：蒸馏生模型优于线性组合但无法超越Pareto前沿
- **下游分析受偏差影响**：高AUC大PFN校准后仍偏差显著，低AUC小PFN校准后无偏（2σ内）

## 亮点与洞察
- **"度量成为目标就失效"在物理ML中的具体体现**：追逐AUC leaderboard可能导致分析结果系统偏差
- **Pareto前沿作为模型选择工具**：提供比单一AUC更全面的架构评估框架
- **简单模型的价值**：IRC安全的EFN和物理驱动的Expert Features在鲁棒性上远超ParT

## 局限性 / 可改进方向
- 仅用Pythia vs Herwig衡量resilience，真实系统不确定性更复杂
- 无探测器模拟——实际分析中探测器效应也影响resilience
- 知识蒸馏实验较初步，未尝试更先进策略

## 相关工作与启发
- **vs ATLAS accuracy-vs-precision研究**：ATLAS发现物理建模不确定性主导，本文量化并提出Pareto框架
- **vs Butter et al. (2023)**：他们也研究q/g resilience，本文扩展到top tagging并增加下游bias案例

## 评分
- 新颖性: ⭐⭐⭐⭐ Pareto前沿视角新颖，下游偏差案例有说服力
- 实验充分度: ⭐⭐⭐⭐ 多架构系统扫描+知识蒸馏+案例研究
- 写作质量: ⭐⭐⭐⭐⭐ 简洁有力，信息密度高
- 价值: ⭐⭐⭐⭐⭐ 对整个HEP-ML社区的模型选择实践有直接指导意义
