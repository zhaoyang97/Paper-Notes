---
title: >-
  [论文解读] Local Guidance for Configuration-Based Multi-Agent Pathfinding
description: >-
  [AAAI 2026][MAPF] 提出局部引导（Local Guidance）概念改进 LaCAM 的多智能体路径规划，通过在每个配置生成步为每个智能体构造局部时空路径来缓解拥塞，最高可将解的代价降低 50%，同时保持 1000 智能体下几秒内完成。 领域现状：MAPF（多智能体路径规划）追求在图上为所有智能体找到无碰撞路…
tags:
  - "AAAI 2026"
  - "MAPF"
  - "局部引导"
  - "LaCAM"
  - "拥塞缓解"
  - "时空搜索"
---

# Local Guidance for Configuration-Based Multi-Agent Pathfinding

**会议**: AAAI 2026  
**arXiv**: [2510.19072](https://arxiv.org/abs/2510.19072)  
**代码**: [https://github.com/allegorywrite/lg_lacam](https://github.com/allegorywrite/lg_lacam)  
**领域**: 其他  
**关键词**: MAPF, 局部引导, LaCAM, 拥塞缓解, 时空搜索

## 一句话总结

提出局部引导（Local Guidance）概念改进 LaCAM 的多智能体路径规划，通过在每个配置生成步为每个智能体构造局部时空路径来缓解拥塞，最高可将解的代价降低 50%，同时保持 1000 智能体下几秒内完成。

## 研究背景与动机

**领域现状**：MAPF（多智能体路径规划）追求在图上为所有智能体找到无碰撞路径。LaCAM 等基于配置的求解器可快速处理上千智能体，但解的质量高度次优。

**现有痛点**：已有的"引导"方法（guidance）通过全局考虑整个环境和所有智能体来缓解拥塞，但全局引导是时间无关的，丢失了时空信息，在瓶颈区域效果有限。

**核心矛盾**：全局引导太粗糙——知道"哪条路不堵"但不知道"什么时候哪里堵"；精确解太昂贵——考虑完整碰撞约束计算量爆炸。

**切入角度**：局部引导位于两者之间——在每个智能体周围的局部时空窗口内构建引导路径，比全局引导更精确（有时空感知），比精确解更轻量。

## 方法详解

### 整体框架

在 LaCAM 框架内，每次配置生成时调用局部引导构造：为每个智能体在窗口 $w$ 内规划一条避免碰撞的短期路径 $\Phi[i]$，然后用 $\Phi[i][1]$（第一步目标位置）指导 PIBT 的偏好排序。

### 关键设计

1. **局部引导构造 (Algorithm 1)**:

    - 功能：为每个智能体独立规划 $w$ 步的短期路径，考虑与其他智能体路径的碰撞
    - 核心思路：每个智能体执行基于代价的 windowed space-time A*，代价函数为 $c(\pi[t], \pi[t+1]) = \langle 1 + \alpha \cdot \mathsf{Ind}[\chi > 0], \chi \rangle$，其中 $\chi$ 是与其他路径的碰撞数，$\alpha=3$ 为碰撞惩罚权重
    - 设计动机：软碰撞约束而非硬约束，允许一定碰撞但给予惩罚，使智能体在拥塞区域找到折衷路径

2. **路径缓存与初始化 (Algorithm 2)**:

    - 功能：利用上一步的引导路径初始化当前步，避免从头重新规划
    - 核心思路：如果智能体 $i$ 当前位置与上一步引导路径的第一步一致，则简单左移路径 $\Phi_k[i][t] \leftarrow \Phi_{k-1}[i][t+1]$
    - 设计动机：将计划迭代需求从 2 次降至 1 次，大幅节省计算开销

3. **智能体排序**:

    - 功能：按上一轮碰撞数降序排序智能体的规划顺序
    - 核心思路：碰撞最多的智能体先规划，获得更自由的路径选择
    - 设计动机：缓解顺序规划中的不公平性——先规划者约束少，后规划者约束多

4. **全局+局部引导融合**:

    - 核心思路：修改代价函数加入全局引导偏移 $c = \langle 1 + \alpha \cdot \mathsf{Ind}[\chi > 0], \delta(\pi[t+1]), \chi \rangle$，其中 $\delta(v)$ 是到全局引导路径的距离

### 时间复杂度

单次引导构造：$O(nw|V|\log(w|V|))$，其中 $n$ 为智能体数、$w$ 为窗口大小、$|V|$ 为图顶点数。

## 实验关键数据

### 主实验：多地图代价对比（1000 智能体）

| 方法 | maze-128 | random-64 | warehouse | den312d | 平均降幅 |
|------|----------|-----------|-----------|---------|----------|
| LaCAM | baseline | baseline | baseline | baseline | — |
| GG (全局引导) | -14% | -8% | -15% | -12% | ~12% |
| LG (局部引导) | **-38%** | **-20%** | -12% | **-25%** | ~24% |
| LG+GG | **-40%** | **-22%** | **-18%** | **-28%** | ~27% |
| LNS2 | -30% | -15% | -10% | failed | — |

### 消融实验

| 配置 | 效果 |
|------|------|
| LG w/ 缓存+1次迭代（默认） | 最佳性价比 |
| LG w/o 缓存（从头规划） | 质量显著下降 |
| LG w/ 2次迭代 | 微小提升，时间翻倍 |
| 不频繁更新（每2/3步）| 可能比无引导更差 |
| $\alpha=3, w=20$ | Sweet spot |

### 关键发现
- **局部引导在大多数地图上超越全局引导**，尤其在瓶颈区域（如 maze 地图减 38% vs 全局 14%）
- **唯一例外是 warehouse 地图**：全局引导在长窄走廊中更有效（能抑制来回移动）
- **"活"引导至关重要**：每步更新引导路径才有效，降低更新频率反而可能变差
- 与 anytime 方法（LNS）组合时，**更好的初始解在密集场景下比更多优化时间更重要**
- 扩展到 10000 智能体仍保持 ~30% 代价降低

## 亮点与洞察
- **局部 vs 全局引导的频谱观点**：将 MAPF 解法视为从"粗糙全局引导"到"精确无碰撞解"的连续频谱，局部引导是实用中间点——这个概念化方式很有启发性
- **缓存+增量更新**：利用 DFS 搜索的时序连续性复用路径缓存，将 2 次迭代的开销降为 1 次，工程上很巧妙
- **软碰撞约束的实践价值**：相比硬约束导致保守行为，软约束在拥塞缓解和路径效率间取得良好平衡

## 局限与展望
- 在 warehouse 类长窄走廊地图上不如全局引导，局部视野无法解决全局交通流优化
- 窗口大小 $w$ 和碰撞惩罚 $\alpha$ 需要针对不同地图调参
- 只验证了 PIBT 作为配置生成器，未探索其他生成器
- 每步更新的计算开销虽然可接受，但仍比无引导版本慢数倍

## 相关工作与启发
- **vs GG (全局引导, SUO)**：全局引导是时间无关的拥塞缓解，LG 是时空感知的，在瓶颈区更精确
- **vs LNS2**：同为次优 MAPF 方法，LG 在大多数地图上解质量更好且更快
- **vs lacam3 (SOTA)**：LG+LNS 在非 warehouse 地图上与 lacam3 持平或更优
- **vs 学习型方法（模仿学习/RL）**：本文为纯搜索方法，泛化性更好、无需训练数据，但可能在特定环境下不如针对性训练的学习方法

## 评分
- 新颖性: ⭐⭐⭐⭐ 局部引导概念清晰且实用，但本质是 windowed PP 的变体
- 实验充分度: ⭐⭐⭐⭐⭐ 多地图、多规模、丰富的消融和设计验证
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，图表直观
- 价值: ⭐⭐⭐⭐ 推动了实时 MAPF 的 Pareto 前沿

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents](symbolic_planning_and_multi-agent_path_finding_in_extremely_dense_environments_w.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](area-optimal_control_strategies_for_heterogeneous_multi-agen.md)
- [\[CVPR 2026\] Scalable Multi-View Subspace Clustering with Tensorized Anchor Guidance](../../CVPR2026/others/scalable_multi-view_subspace_clustering_with_tensorized_anchor_guidance.md)
- [\[AAAI 2026\] An Epistemic Perspective on Agent Awareness](an_epistemic_perspective_on_agent_awareness.md)
- [\[AAAI 2026\] ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance](toward_the_frontiers_of_reliable_diffusion_sampling_via_adversarial_sinkhorn_att.md)

</div>

<!-- RELATED:END -->
