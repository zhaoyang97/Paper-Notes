---
title: >-
  [论文解读] FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling
description: >-
  [NeurIPS 2025][优化/理论][联邦学习] 将联邦动态剪枝重新建模为组合多臂赌博机(CMAB)问题，提出基于 Thompson Sampling 的拓扑调整机制 TSAdj，通过概率性决策替代确定性决策来获得更鲁棒的稀疏模型拓扑，同时显著降低通信开销。 领域现状：联邦学习(FL)允许分布式设备协作训练模型而不共享…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "联邦学习"
  - "动态剪枝"
  - "Thompson Sampling"
  - "组合多臂赌博机"
  - "稀疏训练"
---

# FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling

**会议**: NeurIPS 2025  
**arXiv**: [2501.19122](https://arxiv.org/abs/2501.19122)  
**代码**: [GitHub](https://github.com/Little0o0/FedRTS)  
**领域**: 优化  
**关键词**: 联邦学习, 动态剪枝, Thompson Sampling, 组合多臂赌博机, 稀疏训练

## 一句话总结
将联邦动态剪枝重新建模为组合多臂赌博机(CMAB)问题，提出基于 Thompson Sampling 的拓扑调整机制 TSAdj，通过概率性决策替代确定性决策来获得更鲁棒的稀疏模型拓扑，同时显著降低通信开销。

## 研究背景与动机

**领域现状**：联邦学习(FL)允许分布式设备协作训练模型而不共享数据，但深度模型的高计算和通信需求限制了资源受限设备的参与。现有联邦剪枝框架采用动态稀疏训练，内循环固定拓扑更新权重，外循环基于聚合信息调整拓扑（剪枝+重激活）。

**现有痛点**：现有方法（PruneFL、FedTiny、FedDST、FedMef等）存在三个关键问题：
   - **贪婪调整**：仅利用当前轮少数参与客户端的近视信息，丢弃所有历史观测
   - **拓扑不稳定**：基于不可靠聚合信息做确定性决策，受异构数据分布影响产生高方差
   - **通信低效**：需上传全尺寸梯度等辅助数据，通信代价接近稠密模型

**核心矛盾**：根源在于"近视观察+确定性决策"范式——仅依赖当前轮小部分客户端数据做确定性选择，缺乏全局视角

**切入角度**：将拓扑调整视为 CMAB 问题，每个参数位置为一个臂，每次外循环选择 $K$ 个臂激活

**核心idea**：用 Beta 后验概率分布替代确定性幅值排序进行拓扑选择，用远视综合信息替代近视聚合信息

## 方法详解

### 整体框架
FedRTS 采用双循环训练过程。内循环固定拓扑 $m$，每 $\Delta T$ 轮标准 FL 更新权重 $W$；外循环通过 TSAdj 调整拓扑——服务器从 Beta 分布采样，选择 Top-$K$ 链接作为新拓扑，并基于综合观测更新分布参数。初始拓扑通过随机剪枝和 Beta 分布采样获得。

### 关键设计

1. **CMAB 问题建模**:

    - 功能：将拓扑调整映射为组合多臂赌博机框架
    - 核心思路：每个参数位置 $m_i$ 视为一个臂，外循环 $t$ 选择动作 $S_t$（含 $K = d\langle m\rangle$ 个臂）。目标是最大化 $\max \sum_{t=1}^T r(S_t, \mu)$。观测 $X_t$ 在动作执行后获得，用于指导下一轮决策
    - 设计动机：CMAB 框架使得可以严格分析现有方法的三大缺陷，并自然引出 Thompson Sampling 解法

2. **Thompson Sampling-based Adjustment (TSAdj)**:

    - 功能：用概率性决策替代确定性决策进行拓扑调整
    - 核心思路：为每个链接 $i$ 维护 Beta 后验分布 $P_i \sim \text{Beta}(\alpha_i, \beta_i)$。调整时采样 $\xi_i \sim P_i$，选择 Top-$K$：$S_t = \{i \in \text{Top}(\xi, K)\}$。分布更新：$(\alpha_i, \beta_i) \leftarrow (\alpha_i + \lambda X_{t,i}, \beta_i + \lambda(1 - X_{t,i}))$，其中 $\lambda$ 为缩放因子
    - 设计动机：概率决策天然平衡探索与利用，方差低于确定性方法，后验分布自然整合历史信息

3. **远视综合观测机制**:

    - 功能：融合内外循环、个体与聚合信息构建观测
    - 核心思路：最终观测为加权融合 $X_t = \gamma X_t^{agg} + (1-\gamma)\sum_{n \in C_t} p_n X_t^n$。激活臂用权重幅值判断（区分核心链接 $\kappa$ 个）；非激活臂用梯度幅值判断（选 $K-\kappa$ 个候选重激活）
    - 设计动机：融合个体和聚合信息消除单源偏差；Beta 分布累积历史信息弥补部分客户端参与的缺陷

4. **通信高效设计**:

    - 功能：客户端仅上传 Top 梯度索引
    - 核心思路：上传 $\mathcal{I}_{t+1}^n = \text{Top}(|G_{t+1}^n|, K-\kappa)$，无需传输全梯度
    - 设计动机：服务器端不使用聚合梯度计算非激活臂的 $X_{t,i}^{agg}$（设为0.5以避免高方差），仅使用个体 Top 索引

### 损失函数 / 训练策略
- 标准 FL 任务损失（图像分类用交叉熵，NLP 用语言建模损失）
- SGD 优化器，学习率 $\eta=0.01$，5 local epochs，batch size 64(CV)/16(NLP)
- $\lambda=10$, $\gamma=0.5$, $\alpha_{adj}=0.4$，层间稀疏度遵循 ERK 分布

### 理论保证
在独立性、$L$-连续性和 Mean-field 近似假设下，TSAdj 遗憾上界为 $Reg(T) = O\Big(\sum_{s \neq s_1^*} \frac{\Delta_{max} L^2 \log T}{\Delta_{s,k}^2} + \Delta_{max}\Big)$，是首次从 CMAB 角度给出的联邦剪枝遗憾界。

## 实验关键数据

### 主实验 — NLP 任务 (TinyStories + GPT-2-32M)

| 密度 | 方法 | PPL ↓ | Avg. Acc ↑ | 通信代价 ↓ |
|------|------|-------|-----------|-----------|
| 100% | FedAVG | 20.56 | 0.4387 | 260.41MB |
| 50% | FedDST | 20.10 | 0.4261 | 138.30MB |
| 50% | FedMef | 20.61 | 0.4352 | 165.96MB |
| 50% | **FedRTS** | **18.54** | **0.4422** | **138.84MB** |
| 20% | FedDST | 26.49 | 0.4219 | 60.22MB |
| 20% | FedMef | 21.53 | 0.4293 | 72.26MB |
| 20% | **FedRTS** | **19.93** | **0.4333** | **60.34MB** |

### 消融实验 (CIFAR-10 + ResNet18)

| 配置 | 表现 | 说明 |
|------|------|------|
| FedRTS (完整) | 最优 | 概率决策 + 远视信息 + 融合机制 |
| FedRTS ($\gamma=0$) | 接近完整 | 不用聚合信息，仅用个体信息 |
| FedRTS ($\gamma=1$) | 明显下降 | 仅用聚合信息，丧失稳定性 |
| FedRTS w/o TS ($\gamma=1$) | 大幅下降 | 动量确定性调整替代概率决策 |
| FedTiny | 最差 | 三项优势均无 |

### 关键发现
- FedRTS 在 20% 密度下 PPL 甚至优于全尺寸 FedAVG (19.93 vs 20.56)
- 概率决策贡献最大：FedRTS w/o TS 远不如 FedRTS ($\gamma=1$)
- 个体信息比聚合信息更可靠：$\gamma=0$ 接近 $\gamma=0.5$，而 $\gamma=1$ 显著变差
- 高异构性 (Dirichlet $a=0.1$) 和低参与率 (10%) 下优势更大

## 亮点与洞察
- **CMAB 视角**：首次将联邦剪枝与组合赌博机理论连接，提供了全新的分析工具和算法设计思路
- **概率决策双重优势**：Thompson Sampling 同时解决探索-利用平衡和低方差决策
- **通信效率的自然获得**：TSAdj 仅需 Top 梯度索引，是方法设计的自然副产物

## 局限与展望
- 理论分析简化了内循环和半观测，与实际实现存在差距
- 独立性假设在实际中不严格成立
- 仅在轻量模型(ResNet18/ShuffleNetV2/GPT-2-32M)上验证，缺乏大模型实验
- Beta 分布是否是最优后验选择值得探索

## 相关工作与启发
- **vs FedDST/FedMef**：确定性幅值剪枝 vs 概率采样，FedRTS 在异构场景优势明显
- **vs PruneFL**：客户端端调整更不稳定，FedRTS 服务器端全局概率调整更优
- **vs FedTiny**：需上传完整梯度 vs 仅需 Top 索引，通信效率差距大

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次 CMAB+TS 用于联邦剪枝，理论视角新颖
- 实验充分度: ⭐⭐⭐⭐ CV+NLP 覆盖全、消融完整，但缺大模型实验
- 写作质量: ⭐⭐⭐⭐ 从 CMAB 建模到 TS 解法推导自然流畅
- 价值: ⭐⭐⭐⭐ 为资源受限联邦学习提供实用高效的稀疏训练方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[NeurIPS 2025\] Oracle-Efficient Combinatorial Semi-Bandits](oracle-efficient_combinatorial_semi-bandits.md)
- [\[NeurIPS 2025\] Stable Coresets via Posterior Sampling: Aligning Induced and Full Loss Landscapes](stable_coresets_via_posterior_sampling_aligning_induced_and_full_loss_landscapes.md)
- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](doubly_robust_alignment_for_large_language_models.md)

</div>

<!-- RELATED:END -->
