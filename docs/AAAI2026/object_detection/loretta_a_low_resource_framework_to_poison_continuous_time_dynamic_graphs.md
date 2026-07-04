---
title: >-
  [论文解读] LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs
description: >-
  [AAAI 2026][目标检测][时序图神经网络] 提出 LoReTTA，一种无需代理模型的两阶段对抗投毒攻击框架：先通过 16 种时序重要性度量稀疏化高影响力边，再用保度数负采样算法替换对抗边，在 4 个数据集 × 4 个 TGNN 模型上平均降低 29.47% 性能，同时逃避 4 种异常检测系统且抵御 4 种防御方法。
tags:
  - "AAAI 2026"
  - "目标检测"
  - "时序图神经网络"
  - "对抗攻击"
  - "数据投毒"
  - "连续时间动态图"
  - "图稀疏化"
  - "时序 PageRank"
---

# LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs

**会议**: AAAI 2026  
**arXiv**: [2511.07379](https://arxiv.org/abs/2511.07379)  
**代码**: [https://github.com/ansh997/LoReTTA](https://github.com/ansh997/LoReTTA)  
**领域**: 时间序列 / 图神经网络安全  
**关键词**: 时序图神经网络, 对抗攻击, 数据投毒, 连续时间动态图, 图稀疏化, 时序 PageRank

## 一句话总结
提出 LoReTTA，一种无需代理模型的两阶段对抗投毒攻击框架：先通过 16 种时序重要性度量稀疏化高影响力边，再用保度数负采样算法替换对抗边，在 4 个数据集 × 4 个 TGNN 模型上平均降低 29.47% 性能，同时逃避 4 种异常检测系统且抵御 4 种防御方法。

## 研究背景与动机

**领域现状**：时序图神经网络（TGNN）被广泛用于金融欺诈检测、推荐系统、社交网络分析等高风险场景。连续时间动态图（CTDG）通过带时间戳的边流建模更精细的时序关系。

**现有痛点**：  
   - 静态图对抗攻击方法无法直接迁移到 CTDG：过去的扰动会被新边稀释，未来边不可观测  
   - 离散时间动态图（DTDG）的攻击仅影响单个快照，缺乏时间传播  
   - 现有 SotA 方法 T-SPEAR 需要训练代理模型（计算昂贵），假设攻击者可访问完整数据集（训练+验证+测试），不现实

**核心矛盾**：CTDG 投毒攻击需要精确的时序协调（何时、何处扰动），但代理模型方法代价过高，且访问全数据集的假设在现实中不成立。

**本文目标** 设计一种无需代理模型、仅需训练集访问、满足不可察觉性约束的高效 CTDG 投毒攻击框架。

**切入角度**：用图论启发式直接评估边的时序重要性，然后精心「删除+替换」而非仅添加边。

**核心 idea**：稀疏化高影响力边 + 用约束感知的负采样填补，无需梯度，利用 Temporal PageRank 驱动量化边的时序影响力。

## 方法详解

### 整体框架
LoReTTA 分两步走：**Step 1 稀疏化**（移除高影响力边）→ **Step 2 对抗负采样**（用满足 4 项不可察觉性约束的对抗边替换）。最终投毒图 $\tilde{G} = (V, (E \setminus E') \cup \tilde{E})$，总扰动量受 $\Delta = \lfloor p \cdot |E| \rfloor$ 限制。

攻击者设定为**严格黑盒**：不知模型架构、损失函数、梯度，仅可观测训练集部分。

4 项不可察觉性约束：
- **C1**：扰动预算 $\Delta$
- **C2**：时间可行性（新边时间戳从原始边时间分布采样）
- **C3**：节点活跃窗口（新边只连接时间窗口 $W$ 内活跃的节点）
- **C4**：度数保持（每个节点的入度/出度统计不变）

### 关键设计 1：边稀疏化策略（16 种启发式）
**边级稀疏化（5 种）**：
- 在每个时间戳 $t_i$ 构建累积静态图 $G^{(i)}$，计算每条边的启发式得分（Degree、PageRank、Jaccard、Preference、Random）
- 按得分降序排列，移除 top-$\Delta$ 条边

**时间戳级稀疏化（11 种，基于 Temporal PageRank 漂移）**：
- 计算每个时间戳的 TPR 向量 $r^{(t_i)}$，度量相邻时间戳间的漂移 $\delta^{(t_i)} = d(r^{(t_i)}, r^{(t_{i-1})})$
- 使用 11 种距离度量（MSS、Cosine、Euclidean、Jaccard、JSD、KL、Chebyshev、Wasserstein 等）
- 高漂移值 → 时序不稳定期，微小扰动可造成最大破坏
- 还包括 TER（Temporal EdgeRank）和 Combined-TER 两种边级时序评分

**实验发现**：相似度度量（Cosine、Jaccard）一致优于距离度量（KL、JSD），因为它们更好地对齐 TGNN 隐空间的敏感方向。

### 关键设计 2：约束感知负采样算法
对每条被移除的边，插入满足 C2–C4 约束的新边：
1. **C2**：用核密度估计（KDE）从原始边时间分布采样候选时间戳
2. **C3**：在时间窗口 $W$ 内构建活跃节点池，确保端点在该窗口内活跃
3. **C4**：跟踪每个节点的删除/插入计数，确保入度出度统计不变
4. 对二部图数据集，端点从不同节点集采样；确保新边的节点对未在图中出现过
5. 当约束冲突导致无候选时，触发恢复步骤重新初始化 KDE

### 关键设计 3：Temporal PageRank 与 EdgeRank
- **TPR**：扩展经典 PageRank 到时序图，用时间衰减惩罚中间交互，偏好短的、时间一致的路径
- **TER**：结合 TPR 节点得分与出度归一化，量化每条边在时序信号传播中的重要性
- 时间复杂度 TPR: $O(|E| + |V|)$，空间 $O(|V| \cdot |T|)$

### 损失函数
无损失函数——LoReTTA 是纯启发式方法，不涉及梯度优化。受害者模型使用标准的链接预测损失训练。

## 实验

### 主实验结果（p=0.3，MRR %）

| 方法 | Wikipedia | UCI | MOOC | Enron | 平均降幅 |
|------|-----------|-----|------|-------|---------|
| Clean（TGN） | 80.5 | 44.2 | 61.8 | 27.9 | — |
| T-SPEAR | 69.8 | 30.9 | 32.1 | 27.2 | — |
| ADD-Random | 69.8 | 30.9 | 31.9 | 27.6 | — |
| REM-Degree | 50.5 | 29.7 | 32.1 | 47.6* | — |
| **LoReTTA-Degree** | **66.2** | **17.5** | **20.1** | **12.3** | — |
| **LoReTTA-Cosine** | **54.1** | **15.4** | **23.3** | **20.3** | — |
| **LoReTTA-Jaccard** | **55.2** | **15.7** | **22.9** | **18.3** | — |

- 跨 4 数据集 × 4 模型平均性能降低 **29.47%**
- 最大降幅：MOOC 上 42.0%、Wikipedia 上 31.5%、UCI 上 28.8%、Enron 上 15.6%
- 比 T-SPEAR 速度快 **3.91×**（最高 10×），无需代理模型

### 防御鲁棒性实验（TGN, p=0.3）

| 防御方法 | Wiki Clean→攻击后 | UCI Clean→攻击后 |
|---------|-------------------|-----------------|
| 无防御（LoReTTA-Degree） | 80.5 → 66.2 | 44.2 → 17.5 |
| SVD | 80.5 → 41.0 | 44.2 → 34.0 |
| Cosine 过滤 | 80.5 → 27.6 | 44.2 → 8.1 |
| T-shield | 80.5 → 10.3 | 44.2 → 6.0 |

- **防御反而加剧了性能下降**：过滤方法误删约 **70%** 的真实边（仅 30% 是实际对抗边）
- 说明 LoReTTA 的不可察觉性极强

### 关键发现
1. 相似度度量（Cosine、Jaccard）在稀疏化阶段一致优于距离度量，因其与模型隐空间漂移更对齐
2. 增加攻击者知识（训练数据比例 0.2→0.8）并不总能提升攻击效果——说明脆弱性集中在少量关键节点
3. 攻击效果随扰动率增加先线性增长后饱和——LoReTTA 的确定性算法优先移除最高影响力的边
4. Wikipedia 上 REM 基线表现接近 LoReTTA，因其高度模块化的语义聚类结构使稀疏化本身很有效
5. 所有 4 种异常检测器（MIDAS、F-FADE、AnoEdge-L/G）均无法同时达到 0.7 以上的精确率和召回率

## 亮点
- **零代理模型、零梯度的黑盒攻击**：纯启发式，计算效率极高
- **完善的约束体系**（C1-C4）：比 T-SPEAR 更严格的不可察觉性约束，且都经实验验证
- **16 种稀疏化策略的系统评估**：提供了时序图脆弱性的全面理解
- **攻防双面评估**：同时验证了攻击有效性、防御鲁棒性、异常检测逃逸性，评估最为全面

## 局限性
1. 在 Wikipedia 数据集上，纯删除（REM）基线在仅满足 C1 约束时可接近 LoReTTA 效果，说明负采样在高模块化图上可能引入隐式正则化
2. 攻击仅针对链接预测任务，未验证在节点分类、图分类等其他下游任务上的效果
3. 攻击严格依赖训练数据的可见性，在完全黑盒（无训练数据）场景下不适用
4. 伦理风险：所提方法可被恶意使用破坏依赖 TGNN 的实际系统

## 相关工作
- **CTDG 投毒**：T-SPEAR（首个 CTDG 投毒攻击，需代理模型）、T-Shield（配套防御）
- **TGNN 模型**：TGN（记忆网络）、JODIE（嵌入更新）、TGAT（时序图注意力）、DySAT（动态自注意力）
- **静态图攻击**：Structack（基于度数）、Nettack（梯度引导）、Metattack（元学习）
- **异常检测**：MIDAS（微聚类）、F-FADE（频域）、AnoEdge（异常边流）

## 评分
⭐⭐⭐⭐ — 在 CTDG 对抗攻击这个新兴方向上做了系统性工作。无代理模型 + 约束感知的设计哲学非常务实，16 种启发式的全面对比提供了有价值的图脆弱性分析。攻防评估的完整度在安全会议中都算出色。不足在于防御端没有提出新方法，且在高模块化图上负采样的反作用值得深究。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection](../../CVPR2026/object_detection/cd-buffer_complementary_dual-buffer_framework_for_test-time_adaptation_in_advers.md)
- [\[AAAI 2026\] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)
- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](../../ICCV2025/object_detection/dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [\[AAAI 2026\] YOLO-IOD: Towards Real Time Incremental Object Detection](yolo-iod_towards_real_time_incremental_object_detection.md)
- [\[CVPR 2026\] EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer](../../CVPR2026/object_detection/ew-detr_evolving_world_object_detection_via_incremental_low-rank_detection_trans.md)

</div>

<!-- RELATED:END -->
