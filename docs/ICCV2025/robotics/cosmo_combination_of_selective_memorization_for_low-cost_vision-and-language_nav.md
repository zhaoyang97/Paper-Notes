---
title: >-
  [论文解读] COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation
description: >-
  [ICCV 2025][机器人][视觉语言导航] 提出 COSMO，一种结合选择性记忆的低成本 VLN 架构，通过两个定制化的选择性状态空间模块——Round Selective Scan（RSS，单轮扫描捕获全局上下文）和 Cross-modal Selective State Space Module（CS3，双流跨模态交互）——替代 Transformer 中的高成本注意力机制，以仅 15.5% 参数和 9.3% FLOPs 实现超越基线 DUET 的导航性能。
tags:
  - "ICCV 2025"
  - "机器人"
  - "视觉语言导航"
  - "状态空间模型"
  - "Mamba"
  - "混合架构"
  - "低成本导航"
---

# COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation

**会议**: ICCV 2025  
**arXiv**: [2503.24065](https://arxiv.org/abs/2503.24065)  
**代码**: 无  
**领域**: 机器人  
**关键词**: 视觉语言导航, 状态空间模型, Mamba, 混合架构, 低成本导航

## 一句话总结

提出 COSMO，一种结合选择性记忆的低成本 VLN 架构，通过两个定制化的选择性状态空间模块——Round Selective Scan（RSS，单轮扫描捕获全局上下文）和 Cross-modal Selective State Space Module（CS3，双流跨模态交互）——替代 Transformer 中的高成本注意力机制，以仅 15.5% 参数和 9.3% FLOPs 实现超越基线 DUET 的导航性能。

## 研究背景与动机

视觉语言导航（VLN）要求智能体根据自然语言指令在未见过的 3D 环境中导航，是具身智能的核心任务。当前 VLN 研究存在双重困境：

**性能提升的代价**：当前 SOTA 方法（如 BEVBert、GridMM、KERM）通过引入外部知识库、地图信息、深度信息等来提升性能，但模型参数从 DUET 的 181M 膨胀至 222M+，FLOPs 高达 15-18G。NaviLLM 甚至达到 6633M 参数、1011G FLOPs。

**长指令性能退化**：随着导航指令长度增加，路径复杂度增加，当前基于 Transformer 的方法性能显著下降。Transformer 的 $O(L^2)$ 注意力复杂度使其在处理长历史序列时效率低下。

**Mamba/SSM 的潜力与挑战**：状态空间模型（SSM）具有线性计算复杂度 $O(L)$ 和优秀的长序列建模能力，是替代 Transformer 的有力候选。但直接将 Mamba 用于 VLN 面临两个根本性问题：

**空间关系建模能力不足**：SSM 本质上是为1D序列设计的因果模型，无法像注意力机制那样学习视角间的空间关系

**输入选择能力不足**：虽然 SSM 在生成任务上优于 Transformer，但在需要从输入中选择动作的任务上（VLN 是典型例子）表现不佳

实验验证：直接用 Mamba 替换 DUET 中的 Transformer 组件，SR 从 46.98% 暴跌至 32.25%（下降 14.73%）。

## 方法详解

### 整体框架

COSMO 采用混合架构设计哲学："SSM 负责选择性记忆，Transformer 负责精确决策"：

1. **节点编码器**：构建拓扑地图，编码各节点的全景观测
2. **全局跨模态编码器**：对拓扑地图和指令进行跨模态融合（CS3 + GASA + 交叉注意力）
3. **局部跨模态编码器**：对当前节点的视角和指令进行细粒度融合（CS3 + RSS + 交叉注意力 + 自注意力）
4. **动态动作融合**：融合两个尺度的预测输出最终动作

### 关键设计

1. **Round Selective Scan (RSS)**:

    - 功能：在单轮扫描中捕获所有 token 之间的全局关系，替代双向扫描的高成本和因果性限制
    - 核心思路：将输入序列 $x'$ 翻转后拼接得到 $x = [x' | \text{flip}(x')]$，class token 同时出现在序列两端。仅进行一次前向扫描，当扫描到序列后半部分时，状态空间已包含所有 token 的信息。扫描完成后，将输出等分两半，第二半翻转，二者相加得到最终结果
    - 设计动机：双向扫描（Bi-Mamba）需要两次扫描且仍然是因果模式（每个token只能看到其扫描方向前面的token）。RSS 通过"翻转拼接"的巧妙设计，让每个 token 都能在状态空间中访问全局信息，且仅需一次扫描，利用硬件感知并行算法使序列长度增倍对训练/推理时间影响极小

2. **Cross-modal Selective State Space Module (CS3)**:

    - 功能：将 SSM 适配为双流跨模态交互架构，实现视觉和文本特征的深度融合
    - 核心思路：以更新 $x$（视觉）为例：用 $y$（文本）构建 $\mathbf{B}$ 和 $\mathbf{\Delta}$（控制信息如何写入状态空间），用 $x$ 的 class token 构建 $\mathbf{C}$（控制信息如何从状态空间读出到目标模态）。$y$ 先经过 RSS 式翻转拼接处理，扫描后 class token 的输出作为门控，选择性过滤 $x$ 中的相关信息。反向操作则更新 $y$
    - 设计动机：现有多模态 SSM 方法（如 VL-mamba）简单拼接两个模态的序列作为单流处理，但 VLN 中视觉和文本模态长度严重不平衡，需要细粒度的模态对齐和全面的跨模态交互。CS3 通过"用一个模态控制状态更新、用另一个模态控制输出读取"的设计实现了这一点

3. **混合架构设计（Hybrid Architecture）**:

    - 功能：结合 SSM 的高效记忆过滤和 Transformer 的精确上下文选择
    - 核心思路：在跨模态编码器中，先用 CS3 进行语义对齐和选择性记忆（过滤与指令无关的视觉信息），再用 RSS 广播上下文信息，最后用交叉注意力和自注意力进行token级的精确定位和动作决策
    - 设计动机：消融实验（Table 5）清晰 показал：纯 SSM 架构（SSM+SSM）SR 仅 41.92%，纯 Transformer（Trans+Trans）49.47%，**SSM→Trans（COSMO）达到最优 50.81%**，验证了"先过滤再选择"的混合策略优于单一架构

### 损失函数 / 训练策略

- 延续 DUET 的训练策略，使用相同的输入特征和超参数
- 文本编码器使用 TinyBert（hidden size 312，intermediate size 1200）
- RSS 和 CS3 的状态空间维度设为 16
- 最优检查点基于验证集 unseen split 的 SR+SPL 选择

## 实验关键数据

### 主实验

**REVERIE 数据集：**

| 方法 | Val OSR↑ | Val SR↑ | Val SPL↑ | Test SR↑ | Test SPL↑ | Param(M)↓ | FLOPs(G)↓ |
|------|---------|--------|---------|---------|----------|----------|----------|
| DUET | 51.07 | 46.98 | 33.73 | 52.51 | 36.06 | 181 | 4.95 |
| KERM | 55.21 | 50.44 | 35.38 | 52.43 | 39.21 | 222 | 15.24 |
| NaviLLM | 52.27 | 42.15 | 35.68 | 39.80 | 32.33 | 6633 | 1011 |
| **COSMO** | **56.09** | **50.81** | **35.93** | **52.53** | **36.12** | **28** | **0.46** |

**R2R 数据集：**

| 方法 | Val NE↓ | Val SR↑ | Val SPL↑ | Test NE↓ | Test SR↑ |
|------|--------|--------|---------|---------|---------|
| DUET | 3.31 | 72 | 60 | 3.65 | 69 |
| **COSMO** | **3.15** | **73** | **61** | **3.43** | **71** |

**R2R-CE 数据集 (Test)：**

| 方法 | SR↑ | SPL↑ |
|------|-----|------|
| DUET | 42 | 36 |
| **COSMO** | **47** | **40** |

### 消融实验

**RSS vs 其他扫描策略（REVERIE Val Unseen）：**

| 配置 | SR↑ | SPL↑ | 推理时间(s)↓ | 说明 |
|------|-----|------|------------|------|
| Mamba + CS3 (#1) | 47.20 | 32.04 | 10.46 | 普通Mamba替代RSS |
| Bi-Mamba + CS3 (#2) | 50.75 | 34.77 | 11.38 | 双向Mamba替代RSS |
| RSS + Bi-Mamba (#3) | 46.95 | 31.40 | 10.80 | Bi-Mamba替代CS3 |
| **RSS + CS3 (#4)** | **50.81** | **35.93** | **10.64** | COSMO完整版本 |

**架构设计消融：**

| 架构 | SR↑ | SPL↑ | 说明 |
|------|-----|------|------|
| SSM + SSM | 41.92 | 27.61 | 纯SSM（最差） |
| Trans + SSM | 46.58 | 31.38 | Transformer先→SSM后 |
| Trans + Trans | 49.47 | 31.10 | 纯Transformer缩小版 |
| **SSM + Trans** | **50.81** | **35.93** | COSMO：SSM先→Trans后（最优） |

**直接使用 Mamba 的性能（验证失败案例）：**

| 方法 | SR↑ | 说明 |
|------|-----|------|
| Mamba (单流) | 32.25 | 比DUET低14.73%，验证了直接用SSM不可行 |
| Bi-Mamba | 35.61 | 仅提升3.26%，仍远不够 |

### 关键发现

1. **参数和计算量的巨大缩减**：COSMO 仅需 28M 参数（DUET 的 15.5%）和 0.46G FLOPs（DUET 的 9.3%），实现甚至更好的导航性能
2. **SSM→Trans 的混合顺序至关重要**：SSM 先做选择性记忆过滤，Transformer 再做精确动作决策，反过来效果反而不好
3. **RSS 严格优于 Bi-Mamba**：单轮扫描不仅效率更高（推理快0.74s），SPL 还高出1.16%
4. **CS3 显著优于简单序列拼接**：SR 差距达 3.86%，SPL 差距达 4.53%，验证了双流架构对 VLN 的必要性
5. **COSMO 在 R2R-CE 上提升最为显著**：SR +5%，SPL +4%，说明在连续环境（更长决策序列）中优势更明显

## 亮点与洞察

- **"选择性记忆"概念精准定位了 VLN 的核心需求**：导航过程中积累大量视觉观察，但只有与指令相关的部分需要保留，SSM 的选择性机制天然适合这一需求
- **RSS 的翻转拼接设计极其巧妙**：用最小的实现复杂度实现了全局上下文感知，避免了多方向扫描的开销
- **CS3 的双流设计思路**：用源模态控制状态更新、目标模态控制输出读取，为跨模态 SSM 交互提供了优雅的解决方案
- **6.5× 参数缩减 + 10.7× FLOPs 缩减**：在边缘设备部署具有重大实际意义
- **节点编码器改进**：将拓扑地图构建前移到节点编码阶段，减少了跨模态编码器的参数需求

## 局限与展望

1. RSS 和 CS3 的状态空间维度（N=16）是手动设定的，更大的状态空间可能带来进一步提升
2. COSMO 在 REVERIE test split 的 SPL 指标与 DUET 几乎相同（36.12 vs 36.06），在导航效率上改进不够明显
3. 未与使用额外数据/知识的方法（如 ScaleVLN、BEVBert）进行公平对比，因为它们的性能提升部分来自额外信息
4. CS3 的门控机制使用 class token 的输出，当序列很长时 class token 可能无法有效压缩所有信息
5. 未探索更新的 SSM 变体（如 Mamba2）是否能进一步提升性能

## 相关工作与启发

- 与 VL-mamba、Cobra 等多模态 SSM 工作相比，CS3 的双流设计更适合处理模态间序列长度不平衡的场景（VLN 中视觉序列远长于文本）
- 混合架构（SSM + Transformer）的成功为其他需要"记忆+决策"的任务（如对话系统、长文本推理）提供了设计参考
- RSS 的翻转拼接技术可以推广到其他需要非因果 SSM 的视觉任务
- COSMO 的成功说明，模型结构创新可能比单纯增加模型规模或引入外部知识更具成本效益

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ RSS 和 CS3 是全新设计，混合架构哲学思想深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 三个VLN数据集，详尽的消融和架构设计验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但CS3的描述可以更直观
- 价值: ⭐⭐⭐⭐⭐ 10倍计算量缩减的同时性能持平或提升，对具身智能的实际部署意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments](navmorph_a_self-evolving_world_model_for_vision-and-language_navigation_in_conti.md)
- [\[ICCV 2025\] Selective Contrastive Learning for Weakly Supervised Affordance Grounding](selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)
- [\[CVPR 2025\] Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method](../../CVPR2025/robotics/towards_long-horizon_vision-language_navigation_platform_benchmark_and_method.md)
- [\[ECCV 2024\] LLM as Copilot for Coarse-Grained Vision-and-Language Navigation](../../ECCV2024/robotics/llm_as_copilot_for_coarse-grained_vision-and-language_navigation.md)
- [\[CVPR 2026\] AwareVLN: Reasoning with Self-awareness for Vision-Language Navigation](../../CVPR2026/robotics/awarevln_reasoning_with_self-awareness_for_vision-language_navigation.md)

</div>

<!-- RELATED:END -->
