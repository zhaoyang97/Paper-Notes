---
title: >-
  [论文解读] FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion
description: >-
  [AAAI 2026][视频理解][细粒度动作识别] 提出 FineTec 框架，通过上下文感知序列补全、基于生物先验的骨架空间分解、物理驱动的加速度建模三个模块，在时序损坏条件下实现鲁棒的细粒度骨架动作识别。 领域现状 细粒度动作识别（FAR）要求区分时间变化微妙、语义差异细微的动作（如"前空翻伸展体加两次转体"）…
tags:
  - "AAAI 2026"
  - "视频理解"
  - "细粒度动作识别"
  - "时序损坏"
  - "骨架分解"
  - "拉格朗日动力学"
  - "序列补全"
---

# FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion

**会议**: AAAI 2026  
**arXiv**: [2512.25067](https://arxiv.org/abs/2512.25067)  
**代码**: [项目页](https://smartdianlab.github.io/projects-FineTec/)  
**领域**: 视频理解  
**关键词**: 细粒度动作识别, 时序损坏, 骨架分解, 拉格朗日动力学, 序列补全

## 一句话总结

提出 FineTec 框架，通过上下文感知序列补全、基于生物先验的骨架空间分解、物理驱动的加速度建模三个模块，在时序损坏条件下实现鲁棒的细粒度骨架动作识别。

## 研究背景与动机

### 领域现状
细粒度动作识别（FAR）要求区分时间变化微妙、语义差异细微的动作（如"前空翻伸展体加两次转体"），骨架表示因紧凑性和对运动线索的显式关注而成为有效模态。

### 核心痛点
在复杂场景（如体操）中，在线姿态估计存在**严重的帧丢失问题**——快速运动时丢帧率可达 69.6%。这导致骨架序列**时序损坏**，对依赖连续细微运动线索的细粒度动作识别造成致命打击。

### 现有方法的两大局限

**时间恢复不足**：大多数模型在干净的离线标注骨架上训练，缺乏处理在线检测伪影的机制

**时空建模不充分**：忽略人体固有的生物结构，主要关注逐点位置特征，忽视连续运动学约束——只用"位移"信息，缺乏物理可解释性

### 核心 Idea
将问题分解为三步：(1) 先**补全**损坏的骨架序列恢复时序连续性；(2) 利用**生物先验**将骨架分解为动态/静态区域进行差异化增强；(3) 引入**拉格朗日动力学**估计关节加速度作为额外判别性特征。位置序列 + 加速度序列联合输入 GCN 进行分类。

## 方法详解

### 整体框架

FineTec 包含三个核心模块，按序处理时序损坏的骨架序列 $S_{corrupt} \in \mathbb{R}^{T \times K \times 2}$：

1. **Context-aware Sequence Completion**：从损坏输入恢复基础序列 $S_{base}$
2. **Skeleton-based Spatial Decomposition**：将 $S_{base}$ 分解为动态/静态区域，生成增强序列 $S_{dyna}$、$S_{stat}$，融合为 $S_{pred}$
3. **Physics-driven Acceleration Modeling**：利用拉格朗日动力学估计加速度特征 $\mathbf{a}$

最终 $S_{pred}$ 和 $\mathbf{a}_{pred}$ 联合输入 GCN 分类器。

### 关键设计

#### 1. **上下文感知序列补全（Context-aware Sequence Completion）**

- **核心思路**：采用 **In-Context Learning(ICL)** 范式恢复损坏序列
- **实现**：
    - 从 Human3.6M 构建骨架库，取时间平均获得先验序列 $S_{prior}$
    - 从骨架库采样序列，用**五种时间掩码策略**（随机、模式化、前缀/后缀/中间块）进行损坏，构造 prompt pair（原始+掩码）
    - 输入的 $S_{corrupt}$ 与 $S_{prior}$ 组成 query pair
    - 通过轻量级空间和时间 MLP 处理，完成近似恢复
- **设计动机**：五种掩码策略覆盖真实场景中各种数据缺失模式，ICL 范式避免对特定损坏模式的过拟合
- **损失函数**：$\mathcal{L}_{ICL} = \text{MSE}(S_{gt}, S_{base}) + \text{MSE}(S_{context}, S_{mask})$

#### 2. **骨架空间分解（Skeleton-based Spatial Decomposition）**

- **核心思路**：基于人体生物结构先验，将关节按运动强度分为动态/静态区域，进行差异化增强
- **具体步骤**：
    - 将 K=17 个关节分为 5 个语义区域：头部($G_0$)、左臂($G_1$)、右臂($G_2$)、左腿($G_3$)、右腿($G_4$)
    - 计算每个关节的平均帧间位移：$D_{avg}^{(i)} = \frac{1}{T-1}\sum_{t=0}^{T-2}\|S_{base}^{t+1,i} - S_{base}^{t,i}\|_2$
    - 计算区域运动强度：$\bar{D}_j = \frac{1}{|G_j|}\sum_{i \in G_j} D_{avg}^{(i)}$
    - **Top-2 运动最强区域 → 动态组**，其余 3 个 → 静态组
- **差异化增强**：
    - 动态区域：强时空扰动（时间裁剪、随机丢弃、插值）→ $S_{dyna}$
    - 静态区域：弱空间扰动（随机翻转）→ $S_{stat}$
- **融合**：$S_{base}$、$S_{dyna}$、$S_{stat}$ 三者融合为 $S_{pred}$
- **设计动机**：动态区域的细微差异是区分相似动作的关键，强増强放大判别信息；静态区域保持稳定避免引入噪声

#### 3. **物理驱动加速度建模（Physics-driven Acceleration Modeling）**

- **核心思路**：利用拉格朗日动力学方程显式建模关节加速度
- **拉格朗日方程**：$M(S)\ddot{S} + C(S,\dot{S})\dot{S} + g(S) = \tau$
- **求解加速度**：$\ddot{S} = \{M(S)\}^{-1} \cdot \tau - \hat{C}(S,\dot{S})\dot{S} - \hat{g}(S)$
- **实现**：
    - 提取全局+局部的位置和速度特征
    - 用神经网络 $\mathbb{E}$ 估计各物理项（惯性矩阵 $M$、科里奥利力 $C$、重力 $g$、驱动力 $\tau$）
    - 对对称矩阵只估计上三角部分，再对称化
- **伪加速度**：用二阶有限差分计算 $\hat{a}_t = \frac{S_{t+1} - 2S_t + S_{t-1}}{(\Delta t)^2}$
- **融合**：$\mathbf{a}_t = \text{Fusion}(\hat{a}_t, \ddot{S}_t) \in \mathbb{R}^{K \times 2}$
- **设计动机**：加速度能捕捉位移信息无法表达的动态特性，物理约束让特征更具可解释性

### 损失函数 / 训练策略

两阶段训练：
1. **预训练阶段**：在骨架数据集上训练序列补全模块（Adam, lr $1 \times 10^{-5}$, 40K iterations）
2. **微调阶段**：冻结补全模块，训练整体框架（SGD + Nesterov动量, lr 0.05-0.2, 150 epochs）

总体损失：$\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{Ac}$，$\mathcal{L}_{Ac} = \frac{1}{3}\sum_\alpha \text{MSE}(\hat{\mathbf{a}}_\alpha, \mathbf{a}_\alpha)$

## 实验关键数据

### 主实验（细粒度数据集）

| 方法 | Gym288-Sev. Top-1 | Gym288-Sev. Mean | Gym99-Sev. Top-1 | Gym99-Sev. Mean |
|------|-------------------|------------------|-------------------|-----------------|
| ST-GCN | 0.742 | 0.304 | 0.871 | 0.783 |
| PYSKL-J | 0.773 | 0.315 | 0.884 | 0.791 |
| CTR-GCN | 0.760 | 0.271 | 0.884 | 0.803 |
| Sparse(CVPR'25) | 0.683 | 0.237 | 0.808 | 0.725 |
| **FineTec** | **0.781** | **0.356** | **0.891** | **0.805** |

在最具挑战性的 Gym288-Severe 上，FineTec Mean class accuracy 超过最佳基线 **13%**，超过 Sparse **50%**。

### 粗粒度数据集

| 方法 | NTU-60 Sev. | NTU-120 Sev. |
|------|-------------|--------------|
| ST-GCN | 0.879 | 0.781 |
| CTR-GCN | 0.879 | 0.793 |
| Sparse | 0.864 | 0.767 |
| **FineTec** | **0.892** | **0.813** |

严重损坏下 NTU-60 提升 **1.3%**，NTU-120 提升 **1.7%**。

### 消融实验

| 配置 | Minor | Moderate | Severe | 说明 |
|------|-------|----------|--------|------|
| w/o Completion | 0.812 | 0.785 | 0.751 | 去除补全模块 |
| w/o Skeleton Decomp. | 0.787 | 0.780 | 0.770 | 去除骨架分解 |
| w/o Physics | 0.789 | 0.776 | 0.775 | 去除物理建模 |
| **Full FineTec** | **0.815** | **0.797** | **0.781** | 完整模型 |

| $S_{dyna}$ | $S_{stat}$ | Moderate | Severe | 说明 |
|------------|------------|----------|--------|------|
| ✗ | ✓ | 0.790 | 0.774 | 仅静态增强 |
| ✓ | ✗ | 0.786 | 0.764 | 仅动态增强 |
| ✓ | ✓ | **0.797** | **0.781** | 两者结合最优 |

### 骨架恢复质量

| 方法 | Gym99-Sev. MPJPE↓ | N-MPJPE↓ | MPJVE↓ |
|------|-------------------|----------|--------|
| SiC-Dyna | 0.192 | 0.174 | 0.321 |
| **FineTec** | **0.147** | **0.132** | **0.113** |

MPJPE 降低 **23.4%**。

### 关键发现
1. 三个模块各自贡献独立且互补，移除任一模块都会导致明显性能下降
2. 序列补全对严重损坏场景贡献最大（从 0.751 提升到 0.781）
3. 动态+静态分解的组合比单独使用任何一个都好
4. 交叉注意力融合（CA）优于 MLP 融合（0.797 vs. 0.779）

## 亮点与洞察

1. **物理可解释性**：引入拉格朗日动力学建模加速度，超越了纯数据驱动方法，让运动特征具有物理意义
2. **系统化解决方案**：三个模块对应恢复→分解→增强的完整流水线，环环相扣
3. **新数据集贡献**：构建了 Gym288-skeleton（288 类细粒度动作，38K 序列），极具挑战性
4. **ICL 范式用于骨架补全**：借鉴 NLP 的 in-context learning 思想来处理序列恢复问题，思路新颖

## 局限与展望

1. 骨架库使用固定的 Human3.6M 数据，未来可探索更自适应的数据驱动方法
2. 动态/静态分组基于手动定义（Top-2 最高运动区域），可考虑学习化的分组策略
3. 当前只在 2D 骨架上验证，扩展到 3D 骨架和多模态可能进一步提升
4. 关节级加速度建模可扩展到子组或肢体级别动力学

## 相关工作与启发

- 与 InfoGCN++ 使用 Neural-ODE 进行动作识别类似，但本文使用更经典的拉格朗日力学
- ICL 范式（来自 NLP）在骨架恢复中的创新应用，可推广到其他时序数据补全任务
- 生物先验引导的分解策略可推广至手势识别、手术动作分析等需要细粒度区分的领域

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 三模块组合虽各有先例，但整合方式和物理驱动思路新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ — 细粒度+粗粒度数据集、三级损坏强度、详尽消融、恢复质量评估
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数学推导完整
- **价值**: ⭐⭐⭐⭐ — 时序损坏场景具有显著实用性，新数据集有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MA-Bench: Towards Fine-grained Micro-Action Understanding](../../CVPR2026/video_understanding/ma-bench_towards_fine-grained_micro-action_understanding.md)
- [\[ECCV 2024\] FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition](../../ECCV2024/video_understanding/finepseudo_improving_pseudo-labelling_through_temporal-alignablity_for_semi-supe.md)
- [\[AAAI 2026\] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)
- [\[AAAI 2026\] R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios](r-avst_empowering_video-llms_with_fine-grained_spatio-temporal_reasoning_in_comp.md)
- [\[AAAI 2026\] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)

</div>

<!-- RELATED:END -->
