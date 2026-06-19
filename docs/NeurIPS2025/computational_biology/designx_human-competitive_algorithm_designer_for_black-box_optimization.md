---
title: >-
  [论文解读] DesignX: Human-Competitive Algorithm Designer for Black-Box Optimization
description: >-
  [NeurIPS 2025][计算生物][黑盒优化] 提出 DesignX，首个统一学习算法工作流生成和超参数动态控制两个子任务的自动算法设计框架，通过双 Transformer 智能体在 10k 合成问题上大规模预训练，在合成测试集和蛋白质对接/AutoML/UAV 路径规划等真实场景中超越人类手工设计的优化器。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "黑盒优化"
  - "自动算法设计"
  - "双智能体强化学习"
  - "MetaBBO"
  - "Transformer"
---

# DesignX: Human-Competitive Algorithm Designer for Black-Box Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2505.17866](https://arxiv.org/abs/2505.17866)  
**代码**: [GitHub](https://github.com/MetaEvo/DesignX)  
**领域**: 计算生物  
**关键词**: 黑盒优化, 自动算法设计, 双智能体强化学习, MetaBBO, Transformer

## 一句话总结
提出 DesignX，首个统一学习算法工作流生成和超参数动态控制两个子任务的自动算法设计框架，通过双 Transformer 智能体在 10k 合成问题上大规模预训练，在合成测试集和蛋白质对接/AutoML/UAV 路径规划等真实场景中超越人类手工设计的优化器。

## 研究背景与动机

**领域现状**：黑盒优化（BBO）是科学工业核心问题。进化计算（EC）是主流无梯度求解范式，数十年来产生了 GA、DE、PSO、CMA-ES 等大量变体，但每种都需专家手工设计自适应算子和超参控制

**现有痛点**：
   - 为每个新 BBO 问题手动重设计优化器不可扩展
   - MetaBBO（元黑盒优化）虽引入学习范式，但现有方法只学习**单一子任务**——要么做算法选择/工作流生成，要么做超参数控制，二者分离导致次优设计
   - LLM 虽可生成算法代码，但同样只处理单一子任务

**核心矛盾**：算法设计本质包含两个耦合子任务（工作流结构 + 动态超参），分开优化无法联合最优

**切入角度**：构建模块化算法空间 Modular-EC + 双智能体 RL 系统端到端联合学习

**核心 idea**：Agent-1 自回归生成合法优化器工作流 + Agent-2 动态调控超参，通过合作训练目标在 10k 问题分布上元学习

## 方法详解

### 整体框架
输入为一个 BBO 问题实例的特征向量（维度、搜索范围、ELA 统计特征等）。Agent-1（Transformer）根据问题特征自回归生成一个合法的优化器工作流（从 Modular-EC 的 116 个模块中选取）。Agent-2（Transformer）在优化过程中根据实时反馈动态调节所有可控模块的超参数。两个智能体通过合作奖励目标联合训练。

### 关键设计

1. **Modular-EC 模块化算法空间**

    - 功能：将 EC 优化器分解为 10 种模块类型（Initialization, Mutation, Crossover, Selection, Niching, ...），共 116 个模块变体
    - 核心思路：每个模块有唯一 16-bit 编码和拓扑规则（定义合法后继模块），支持自回归生成合法工作流。相比前身 Modular-BBO（主要服务 DE），新增了 ES/GA/PSO 的算子和 Other_Update 模块类型
    - 设计动机：将数十年人类专家设计的算法组件统一编码，为学习代理提供**数百万种可能工作流**的搜索空间

2. **Agent-1: 工作流生成**

    - 功能：给定问题特征 $\mathcal{F}_p$（13维，含 4 基本属性 + 9 ELA 特征），自回归采样模块序列
    - 核心思路：GPT-2 架构，通过 masked Softmax 采样保证拓扑合法性：
      $P(\mathcal{A}_p^{m+1} | \text{start}, \mathcal{A}_p^1, ..., \mathcal{A}_p^m) \sim \text{Softmax}(\text{mask}(\mathcal{A}_p^m) \odot (\mathcal{W}_\text{sample}^T \cdot H^{(m)}))$
      mask 向量根据当前模块的拓扑规则将非法模块概率置零
    - 设计动机：Transformer 的序列建模能力天然适合工作流的有序生成，masked sampling 保证生成的优化器始终合法可执行

3. **Agent-2: 动态超参控制**

    - 功能：在优化过程中每步根据观测 $\mathcal{O}_t$（9维进度特征）为所有可控模块生成超参值
    - 核心思路：将模块 ID 和观测拼接后编码，通过另一组 GPT-2 blocks 输出正态分布参数：
      $\mu = \mathcal{W}_\mu^T \cdot H_{dec}, \quad \Sigma = \mathcal{W}_\Sigma^T \cdot H_{dec}$
      超参从预测分布采样：$C_t^m \sim \mathcal{N}(\mu^{(m)}, \Sigma^{(m)})$
    - 设计动机：EC 优化器的超参直接影响探索/利用权衡，动态控制可根据优化阶段自适应调整

4. **合作训练目标**

    - Agent-1 用 REINFORCE（延迟奖励），Agent-2 用 PPO（即时奖励）
    - 统一目标函数：$\mathcal{J}(\phi, \theta) = \mathbb{E}_{p \sim \mathcal{D}_{train}}[\sum_{t=1}^T r_t]$
    - 每步奖励：$r_t = \frac{f_p^{t-1,*} - f_p^{t,*}}{f_p^{0,*} - f_p^*}$（标准化优化进步）

### 损失函数 / 训练策略
- 12,800 个合成问题实例（9,600 训练 + 3,200 测试），通过 32 个基础函数的 single/composition/hybrid 组合构造
- 训练历时 6 天，主要瓶颈是 BBO 仿真而非神经网络（在 CPU 上运行优化循环）
- 推理时 DesignX 平均 5.5s/问题，与 CMA-ES（5.0s）相当

## 实验关键数据

### 主实验：合成测试集（部分）

| 问题 | before 00 | 00s | 10s | after 20 | MetaBBO | **DesignX** |
|------|-----------|-----|-----|----------|---------|-------------|
| F1 (50D, 30K FEs) | 6.60E+00 | 1.64E+00 | 1.27E+00 | 5.32E+00 | 2.80E+00 | **2.89E-01** |
| F2068 (20D) | 3.79E+01 | 2.32E+00 | 1.46E+01 | 1.65E+01 | 3.72E+01 | **5.16E-01** |
| F2390 (10D) | 3.93E+00 | 2.78E+00 | 6.34E+00 | 1.54E+00 | 2.04E+01 | **1.85E-03** |
| **归一化平均** | 2.94E-01 | 1.96E-01 | 1.54E-01 | 1.46E-01 | 1.32E-01 | **8.26E-02** |

DesignX 在几乎所有测试实例上排名第一，归一化平均比最佳 MetaBBO 低 37%。

### 消融实验

| 配置 | 说明 | 归一化性能 |
|------|------|-----------|
| w/o A1+A2 | 随机双智能体 | 最差 |
| w/o A1 | 仅训练 Agent-2 | 较差 |
| w/o A2 | 仅训练 Agent-1 | 中等 |
| SBS | 静态工作流 | 较差 |
| **DesignX** | **双智能体合作** | **最优** |

### 关键发现
- Agent-1（工作流生成）对性能的贡献大于 Agent-2（超参控制），但两者合作后显著超越单一子任务
- DesignX 学到的设计策略具有可解释性：对多模态问题偏好复合变异策略，对小搜索范围问题偏好种群缩减机制
- 有趣发现：DesignX 认为初始化策略对最终性能影响极小，这与人类直觉未必一致
- DE 相关模块被 DesignX 最频繁选用，说明 DE 算子组合具有最强的通用性
- 在蛋白质对接、AutoML、UAV 路径规划等 OOD 真实任务中同样保持优势

## 亮点与洞察
- **首个端到端学习双子任务的自动算法设计框架**：统一了工作流生成和超参控制，打破了 MetaBBO 领域只学单一子任务的瓶颈
- **Masked Softmax 保证合法性**巧妙地利用拓扑规则和 mask 机制确保自回归生成的优化器工作流始终合法可执行
- **可解释性分析有价值**：通过模块重要性因子和子模块分布分析，揭示了 DesignX 学到的非平凡设计原则，反过来为人类优化器设计提供启示
- 合成数据大规模训练 + 真实任务零样本迁移的范式值得借鉴

## 局限与展望
- Modular-EC 目前仅支持 EC 类优化器（DE/PSO/GA/ES），不覆盖贝叶斯优化等其他 BBO 范式
- 训练需要 6 天 CPU 计算，scaling law 实验受限于计算资源
- 在 rank-based 比较中 DesignX 与 CMA-ES 性能接近，说明仍有提升空间
- 问题特征仅用 13 维 ELA 特征，对高维复杂问题的刻画可能不够
- 目前只在最小的模型配置（1 层 GPT-2）上训练，更大模型+更大训练集的潜力未充分探索

## 相关工作与启发
- **vs ConfigX**: ConfigX 仅做 DE 超参控制（单子任务），DesignX 拓展为双子任务并升级 Modular-BBO 为 Modular-EC
- **vs ALDes**: ALDes 做工作流生成但不做动态超参控制，DesignX 统一了两者
- **vs GLHF**: GLHF 用梯度下降模拟 DE 算子，DesignX 直接通过 RL 学习模块组合
- **vs LLM-based approaches**: LLM 生成算法代码但每次只处理单子任务且推理成本高，DesignX 用小模型实现更高效的端到端设计

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个端到端双智能体自动算法设计框架，理论设计和工程实现均有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 3200 合成测试 + 3 个真实场景 + 消融 + scaling law + 可解释性分析，极为全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，可视化丰富，但符号较多需反复查阅
- 价值: ⭐⭐⭐⭐⭐ 对 MetaBBO 和自动算法设计领域有范式性推进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] g-DPO: Scalable Preference Optimization for Protein Language Models](g-dpo_scalable_preference_optimization_for_protein_language_models.md)
- [\[ICML 2025\] Reliable Algorithm Selection for Machine Learning-Guided Design](../../ICML2025/computational_biology/reliable_algorithm_selection_for_machine_learning-guided_design.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[NeurIPS 2025\] Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning](flow_density_control_generative_optimization_beyond_entropy-regularized_fine-tun.md)
- [\[ICLR 2026\] A Genetic Algorithm for Navigating Synthesizable Molecular Spaces](../../ICLR2026/computational_biology/a_genetic_algorithm_for_navigating_synthesizable_molecular_spaces.md)

</div>

<!-- RELATED:END -->
