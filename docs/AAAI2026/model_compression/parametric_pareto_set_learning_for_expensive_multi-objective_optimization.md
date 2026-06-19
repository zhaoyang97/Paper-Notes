---
title: >-
  [论文解读] Parametric Pareto Set Learning for Expensive Multi-Objective Optimization
description: >-
  [AAAI 2026][模型压缩][Pareto Set Learning] 本文提出 PPSL-MOBO 框架，通过超网络 + LoRA 架构学习从偏好和外在参数到 Pareto 最优解的统一映射，结合高斯过程代理模型和超体积改进采集策略，高效解决昂贵的参数化多目标优化问题。 领域现状 Pareto 集学习（PSL）近年取…
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "Pareto Set Learning"
  - "Multi-Objective Bayesian Optimization"
  - "Hypernetwork"
  - "LoRA"
  - "参数化多目标优化"
---

# Parametric Pareto Set Learning for Expensive Multi-Objective Optimization

**会议**: AAAI 2026  
**arXiv**: [2511.05815](https://arxiv.org/abs/2511.05815)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: Pareto Set Learning, Multi-Objective Bayesian Optimization, Hypernetwork, LoRA, 参数化多目标优化  

## 一句话总结

本文提出 PPSL-MOBO 框架，通过超网络 + LoRA 架构学习从偏好和外在参数到 Pareto 最优解的统一映射，结合高斯过程代理模型和超体积改进采集策略，高效解决昂贵的参数化多目标优化问题。

## 研究背景与动机

### 领域现状

Pareto 集学习（PSL）近年取得显著进展，能学习从偏好向量到 Pareto 最优解的连续映射。多目标贝叶斯优化（MOBO）则用代理模型+采集函数高效解决评估昂贵的优化问题。

### 现有痛点

**参数化多目标优化（PMO）被忽视**：现有 PSL 方法只能处理固定问题实例，无法应对目标函数随参数变化的场景

**传统方法效率低下**：每个新参数值都需要重新从头优化，对于昂贵评估场景代价过高

**无法实时适应**：当参数（如运行条件、患者特征、时间）变化时，缺乏即时推断 Pareto 集的能力

### 核心矛盾

PMO 问题需要在**评估预算有限**的情况下，学习跨越整个参数空间的 Pareto 集，同时要求对未见参数值具有泛化能力。

### 本文目标

设计一个统一框架，仅需一次训练就能即时推断任意参数值下的完整 Pareto 集，大幅减少昂贵评估次数。

### 切入角度

将 PMO 建模为学习映射 $(\boldsymbol{\lambda}, \boldsymbol{t}) \mapsto \boldsymbol{x}^\star$，其中 $\boldsymbol{\lambda}$ 是偏好向量，$\boldsymbol{t}$ 是外在参数。利用超网络生成参数特定的 PS 模型，同时集成贝叶斯优化进行高效数据采集。

### 核心 idea

**用超网络 + LoRA 学习跨参数空间共享的 Pareto 集结构，将"每个参数独立优化"转化为"统一学习+即时推断"**。

## 方法详解

### 整体框架

PPSL-MOBO 包含三个紧密耦合的组件：
1. **超网络-LoRA 架构**：生成参数特定的 PS 模型
2. **高斯过程代理训练**：利用 GP 代理模型进行可扩展优化
3. **智能数据采集**：基于超体积改进的参数空间探索

系统形成闭环：新采集的数据持续精化代理模型和参数化 PS 表示。

### 关键设计一：超网络 + LoRA 架构

**功能**：高效地将 PS 模型适配到不同参数值。

**核心思路**：对 PS 模型的每一层 $l$，将权重分解为共享基础权重和低秩适配：
$$\boldsymbol{\theta}_{\text{ps}}^l(\boldsymbol{t}) = \boldsymbol{\theta}_0^l + \boldsymbol{B}^l(\boldsymbol{t}) \boldsymbol{A}^l(\boldsymbol{t})$$

其中 $\boldsymbol{B}^l(\boldsymbol{t}) \in \mathbb{R}^{d^l \times r}$，$\boldsymbol{A}^l(\boldsymbol{t}) \in \mathbb{R}^{r \times k^l}$，秩 $r \ll \min(d^l, k^l)$。超网络只生成低秩矩阵：
$$\boldsymbol{\theta}_{\text{lora}}(\boldsymbol{t}) = g_{\boldsymbol{\theta}_{\text{hn}}}(\boldsymbol{t})$$

**设计动机**：直接用超网络生成完整权重面临严重的维度不匹配（低维参数 $\to$ 高维权重），难以训练且易过拟合。LoRA 提供了优秀的归纳偏置：不同参数下的 Pareto 集共享大部分结构，差异仅体现在低秩空间中。参数量从 $d^l k^l$ 降至 $r(d^l + k^l)$。

### 关键设计二：增强空间高斯过程代理

**功能**：构建参数感知的代理模型，替代昂贵的目标函数评估。

**核心思路**：定义增强输入空间 $\mathcal{Z} = \mathcal{X} \times \mathcal{T}$，其中 $\boldsymbol{z} = [\boldsymbol{x}, \boldsymbol{t}]$。对每个目标函数建立独立 GP：
$$f_i(\boldsymbol{z}) \sim \mathcal{GP}(\mu_i(\boldsymbol{z}), k_i(\boldsymbol{z}, \boldsymbol{z}'))$$

用下置信界（LCB）作为代理目标：
$$\hat{\boldsymbol{f}}(\boldsymbol{x}; \boldsymbol{t}) = \hat{\boldsymbol{\mu}}(\boldsymbol{z}) - \beta \hat{\boldsymbol{\sigma}}(\boldsymbol{z})$$

**设计动机**：输入增强策略让核函数 $k_i$ 自动学习参数 $\boldsymbol{t}$ 对目标函数的影响及 $\boldsymbol{x}$ 与 $\boldsymbol{t}$ 之间的交互效应。LCB 提供探索-利用的自然平衡。

### 关键设计三：基于平滑 Tchebycheff 的代理训练

**功能**：用 GP 代理模型端到端地训练超网络和基础权重。

**核心思路**：最小化代理 STCH 损失的期望：
$$\hat{\mathcal{L}}(\boldsymbol{\theta}_{\text{hn}}, \boldsymbol{\theta}_0) = \mathbb{E}_{\boldsymbol{t} \sim P_{\boldsymbol{t}}, \boldsymbol{\lambda} \sim P_{\boldsymbol{\lambda}}} \left[ \hat{l}_{\text{stch}}(h_{\boldsymbol{\theta}_{\text{ps}}(\boldsymbol{t})}(\boldsymbol{\lambda}) \mid \boldsymbol{\lambda}, \boldsymbol{t}) \right]$$

其中 STCH 是 Tchebycheff 散化的光滑近似：
$$l_{\text{stch}}(\boldsymbol{x} | \boldsymbol{\lambda}, \nu) = \nu \log\left(\sum_{j=1}^m e^{\lambda_j(f_j(\boldsymbol{x}) - (z_j^\star - \varepsilon))/\nu}\right)$$

**设计动机**：经典 Tchebycheff 可恢复所有（弱）Pareto 最优解（定理 1），但 max 算子不可微。STCH 的光滑化保持了完整 PS 恢复能力（定理 2），同时支持高效梯度反传。

### 关键设计四：超体积改进数据采集

**功能**：在参数空间中智能选择新评估点。

**核心思路**：
1. 从训练好的模型生成候选池 $\mathcal{C} = \{(\boldsymbol{x}_p, \boldsymbol{t}_p)\}$
2. 贪心选择使边际超体积改进最大的点加入评估批次：
$$\text{HVI}(\hat{\mathcal{Y}}_+, \mathcal{Y}) = \text{HV}(\hat{\mathcal{Y}}_+ \cup \mathcal{Y}) - \text{HV}(\mathcal{Y})$$

**设计动机**：超体积是多目标优化中唯一同时衡量收敛性和多样性的标准质量指标，基于 PS 模型生成候选确保采样点位于有希望的参数-决策空间区域。

### 损失函数

全局训练目标为最小化期望代理 STCH 损失（公式 21），通过 Monte Carlo 采样近似期望，梯度反传更新 $\boldsymbol{\theta}_0$ 和 $\boldsymbol{\theta}_{\text{hn}}$。

## 实验关键数据

### 主实验：共享组件多目标优化

在 RE21 问题上（四个决策变量），比较不同共享配置下的超体积：

| 共享变量 | NSGA-II | qParEGO | qEHVI | PSL-MOBO | **PPSL-MOBO** |
|---------|---------|---------|-------|----------|---------------|
| $(x_1)$ | 6.52e-1 | 6.96e-1 | 7.32e-1 | **7.34e-1** | 7.33e-1 |
| $(x_1,x_2)$ | 5.92e-1 | 6.19e-1 | 6.23e-1 | 6.23e-1 | **6.23e-1** |
| $(x_2,x_3,x_4)$ | 5.11e-1 | 5.15e-1 | 5.18e-1 | 5.18e-1 | **5.19e-1** |

**关键**：基线方法每个参数配置需 100 次评估（10 个配置共 1000 次），PPSL-MOBO **总共只需 200 次评估**，且推断新参数只需毫秒级。

### 动态多目标优化

在 DF1/DF2 基准上，PPSL-MOBO 能即时生成近似真实 Pareto 前沿的解集分布，而 DNSGA-II 在两代更新窗口内难以收敛。

### 消融实验

消融研究验证了各组件的贡献：LoRA 适配、GP 代理训练、HVI 采集策略缺一不可。

### 关键发现

1. **5 倍以上的评估效率**：200 次评估 vs 1000 次评估达到相当性能
2. **即时推断**：新参数值下的 PS 推断仅需毫秒，无需重训练
3. LoRA 秩 $r$ 的选择对性能影响显著——过小无法充分表达参数变化，过大则过拟合
4. 在高维共享变量配置下，PPSL-MOBO 的优势更明显（参数空间更大时泛化能力更重要）

## 亮点与洞察

1. **优雅的架构设计**：超网络 + LoRA 将参数化 PS 学习问题转化为高效的低秩适配问题
2. **闭环系统**：代理训练 $\leftrightarrow$ 数据采集 $\leftrightarrow$ 模型更新形成正反馈循环
3. **两个杀手级应用**：共享组件设计和动态优化都是实际工程中的核心需求
4. **统一框架**：一个模型处理整个参数空间，彻底改变了"逐实例优化"的传统范式
5. **理论基础扎实**：基于 STCH 的完整 PS 恢复保证（定理 1, 2）

## 局限与展望

1. GP 代理模型在高维输入空间（$\boldsymbol{x}$ 和 $\boldsymbol{t}$ 拼接）下可能效率下降
2. 目前只处理无约束多目标优化，约束优化扩展有待完成
3. 缺乏样本复杂度的理论保证
4. 超网络 + LoRA 的训练稳定性在实践中可能需要精细调参
5. 实验中的测试问题维度较低（RE21 为 4 维），更高维问题的可扩展性未验证
6. 动态优化实验中，算法与专门设计的 DMOP 方法比较，优势部分来自于"提前看全局"

## 相关工作与启发

1. **PSL** (Lin et al. 2022)：学习偏好到 Pareto 解的映射，本文将其推广到参数化设定
2. **LoRA** (Hu et al. 2022)：大模型高效微调技术，本文创新性地将其用于超网络输出压缩
3. **MOBO** (qEHVI, qParEGO)：单实例多目标贝叶斯优化方法，本文将其思想扩展到参数空间
4. **启发**：LoRA 的"共享基础 + 低秩适配"思想可能推广到更多需要跨实例泛化的优化场景

## 评分

⭐⭐⭐⭐ (4/5)

**优势**：方法设计新颖（PSL + LoRA + MOBO 的巧妙结合），两个应用场景切实有价值，评估效率提升显著。

**不足**：缺乏理论保证，实验问题规模偏小，动态优化的比较不完全公平。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](../../ACL2025/model_compression/bone_soups_multi_objective_gen.md)
- [\[CVPR 2026\] Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](../../CVPR2026/model_compression/frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)
- [\[ICLR 2026\] SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning](../../ICLR2026/model_compression/swireasoning_switch-thinking_in_latent_and_explicit_for_pareto-superior_reasonin.md)
- [\[AAAI 2026\] MetaGDPO: Alleviating Catastrophic Forgetting with Metacognitive Knowledge through Group Direct Preference Optimization](metagdpo_alleviating_catastrophic_forgetting_with_metacognitive_knowledge_throug.md)
- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](../../ACL2025/model_compression/more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)

</div>

<!-- RELATED:END -->
