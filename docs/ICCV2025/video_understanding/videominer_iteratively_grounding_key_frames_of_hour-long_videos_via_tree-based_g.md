---
title: >-
  [论文解读] VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization
description: >-
  [ICCV 2025][视频理解][长视频理解] 提出VideoMiner——基于强化学习的长视频理解树结构框架，通过迭代分割-描述-聚类构建层次化视频树，并提出T-GRPO（树结构Group Relative Policy Optimization）引导策略模型自适应探索关键帧，在4个长视频基准上取得SOTA，并发现T-GRPO可自发激发推理链。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "长视频理解"
  - "关键帧提取"
  - "强化学习"
  - "GRPO"
  - "树结构"
  - "层次化视频表示"
---

# VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization

**会议**: ICCV 2025  
**arXiv**: [2510.06040](https://arxiv.org/abs/2510.06040)  
**代码**: [GitHub](https://github.com/caoxinye/VideoMiner)  
**领域**: 视频理解 / 长视频问答  
**关键词**: 长视频理解, 关键帧提取, 强化学习, GRPO, 树结构, 层次化视频表示

## 一句话总结

提出VideoMiner——基于强化学习的长视频理解树结构框架，通过迭代分割-描述-聚类构建层次化视频树，并提出T-GRPO（树结构Group Relative Policy Optimization）引导策略模型自适应探索关键帧，在4个长视频基准上取得SOTA，并发现T-GRPO可自发激发推理链。

## 研究背景与动机

小时级长视频理解是MM-LLM面临的前沿挑战，涵盖体育精彩片段检测、电影叙事摘要、监控异常检测等应用。与静态图像和短视频相比，长视频包含**数千帧和复杂时序动态**，带来两大核心挑战：

### 挑战1：如何消除大量无关冗余信息？

- **端到端方法**（如LLaVA-Video、Qwen2-VL）：将视频简化为均匀采样的帧列表，但随视频长度增加，**无关信息指数增长**，LLM被淹没
- **层次化方法**（如VideoTree）：引入结构降低复杂度，但可能**破坏原始视频结构**，丢失时序信息

### 挑战2：如何在复杂层次结构中精确定位关键帧？

- VideoTree的视觉聚类+相关性评分在小时级视频中效果有限
- 关键帧提取需要同时满足三个原则：(1) 整合事件级时空信息 (2) 查询导向探索 (3) 适应层次树结构
- 现有方法缺乏**自适应决策**能力——何时停止探索、何时继续深入？

VideoMiner的核心思路：**从粗到细的层次分解**（视频→事件→帧）保持时序连贯性，T-GRPO训练策略模型学会**何时接受、何时继续、何时丢弃**树节点。

## 方法详解

### 整体工作流

VideoMiner由三个组件串联：

1. **场景分割+描述+聚类**：将长视频迭代分解为层次化树结构
2. **T-GRPO树探索**：策略模型决策每个节点的命运（accept/continue/delete）
3. **LLM推理**：选中的关键帧+问题输入VLM生成最终答案

### 场景分割

采用灰度直方图变化检测实现无参数分割：

1. 对每帧计算归一化灰度直方图 $H_t(k)$
2. 用**Bhattacharyya距离**量化相邻帧差异：

$$D_i = -\ln \sum_{k=0}^{255} \sqrt{H_i(k) \cdot H_{i+1}(k)}$$

3. 选取距离序列中Top $K-1$ 个变化点作为分割边界
4. 得到 $K$ 个事件段 $E = \{E_1, \ldots, E_K\}$

**亮点**：基于事件而非离散帧进行分割，保留了时序连贯性。

### 描述生成与聚类

1. **描述生成**：针对每个事件 $E_m$，结合用户问题 $Q$，用VLM生成描述：

$$\text{Caption}_m = \text{VLM}(E_m, Q)$$

问题导向的描述确保提取的信息与用户意图相关。

2. **聚类建树**：将描述通过embedding模型编码为向量 $v_m$，用**DBSCAN**聚类：

$$\{v_1, \ldots, v_K\} \xrightarrow{\text{DBSCAN}} \{l_1, \ldots, l_C\}$$

每个聚类簇构成一个树节点，$C \leq K$ 确保语义相关的场景被归并。

### 树探索：策略模型

策略模型 $\text{PM}$ 接收三个输入作出决策：

$$\text{State}(N_i) = \text{PM}(\text{Caption}_m, Q, \text{depth}(N_i))$$

- **事件描述**：提供时空信息
- **用户问题**：确保探索方向与查询对齐
- **节点深度**：提供层次位置信息

三种决策状态：
- **Accept**：节点包含足够关键帧，无需继续探索
- **Continue**：节点可能相关，展开为新的子节点（重新分割-描述-聚类）
- **Delete**：节点与问题无关，丢弃

### T-GRPO：树结构Group Relative Policy Optimization

T-GRPO基于DeepSeek的GRPO改进，适配树结构和视频理解任务。

**Rollout过程**：执行VideoMiner流程，生成 $n$ 棵不同的树 $T = \{\vec{T_1}, \ldots, \vec{T_n}\}$。

**奖励设计**分为两级：

**节点级奖励 $R_{\text{node}}$**包含三个分量：

1. **格式奖励** $r_{\text{format}}$：完全符合格式获得 $\delta_{\max}$，部分符合获得 $\delta_{\text{corr}}$

2. **长度奖励** $r_{\text{length}}$：高斯分布建模，控制策略模型的输出token长度：

$$r_{\text{length}}(l_o) = \rho \exp\left(-\frac{(l_o - l_t)^2}{2\sigma^2}\right)$$

更长的输出→更详细的推理→更高准确率。

3. **动作奖励** $r_{\text{action}}$：不同动作获得不同奖励（$\delta_d > \delta_a > \delta_c$），定义**树生长素**：

$$\lambda_{\text{auxin}} = \frac{\delta_d + \delta_a}{2\delta_c}$$

直觉来自植物生长素：适度抑制继续探索（continue），鼓励及时做出终止决策（accept/delete），提升定位效率。

**树级奖励 $R_{\text{tree}}$**：基于最终答案准确性。

**总奖励**：

$$R_{\text{total}} = r_{\text{format}} + (r_{\text{length}} + r_{\text{action}}) \cdot R_{\text{tree}}$$

**损失函数**：计算组优势后用PPO-clip风格的损失优化策略模型：

$$A_{ij} = \frac{r_{ij} - \text{mean}(\{r_{11}, \ldots, r_{nG_n}\})}{\text{std}(\{r_{11}, \ldots, r_{nG_n}\})}$$

## 实验

### 主实验：长视频理解基准 (表1)

| 方法 | 基座模型 | EgoSchema | Video-MME Long | LongVideoBench (900-3600s) | MLVU M-Avg |
|------|---------|-----------|----------------|--------------------------|------------|
| LLaVA-Video | Qwen2-7B | 60.2 | 49.3 | 45.5 | 62.1 |
| InternVL2.5 | InternVL-2-8B | 60.0 | 50.6 | 46.4 | 59.2 |
| VideoTree | Qwen-plus | 59.8 | 39.3 | 44.6 | 51.6 |
| LLoVi | Qwen-plus | 62.8 | 50.6 | 39.5 | 54.9 |
| **VideoMiner** | **Qwen2-VL-7B** | **66.2** | **52.2** | **49.3** | **65.1** |

在所有长视频子任务上全面SOTA。视频越长，VideoMiner相对优势越大（Video-MME Long超越最佳baseline +1.6pp，LongVideoBench超越+2.9pp）。

### 消融实验

**图3a：聚类方法对比**

| 方法 | 效果 | 效率 |
|------|------|------|
| 无聚类 | 低 | 最慢（节点指数增长） |
| 帧聚类 | 中 | 较慢 |
| **事件聚类** | **最高** | **最快** |

事件聚类保留更多时序信息，使策略模型能更早做出准确决策。

**图3b：强化学习方法对比**

| 方法 | 表现 |
|------|------|
| 无RL（基础模型） | 最差，随视频长度增加严重退化 |
| RF（无树级奖励） | 显著优于基线 |
| **T-GRPO（含树级奖励）** | **最优** |

树级奖励让策略模型考虑当前决策对未来的影响。

**图5b：生长素 $\lambda_{\text{auxin}}$ 的影响**

- $\lambda < 1$：模型偏好continue，探索彻底但效率低，且可能陷入无目的探索反而降低性能
- $\lambda \approx 1$：最佳平衡点——足够探索且及时终止
- $\lambda > 1$：过早终止，可能遗漏关键帧

## 亮点与洞察

1. **T-GRPO自发激发推理链**：策略模型在训练后自发生成CoT风格的推理过程（"这个节点展示了运动比赛…与问题相关…决定continue"），显著增强推理深度
2. **树生长素的生物学类比**精妙：借鉴植物学中auxin调控生长的概念，用奖励比例调控树的探索深度
3. **DBSCAN的自适应性**：不需要预设聚类数，自动根据描述语义分布确定节点数
4. 无需训练VLM本身，仅训练一个轻量策略模型即可大幅提升长视频理解

## 局限性

1. 多次调用VLM生成描述+最终推理的**级联式架构**延迟较高，难以实时应用
2. 场景分割依赖灰度直方图（简单但粗糙），对光照突变敏感，可能产生虚假分割点
3. 策略模型的奖励设计（6个超参数）需要仔细调优
4. 在短视频任务上相对端到端方法有劣势——对本不需要关键帧选择的场景引入了不必要的复杂性

## 相关工作

- **长视频理解**: LLoVi, VideoTree, VideoAgent, LLaVA-Video
- **视频RL**: GRPO, PPO, RWM-RL
- **关键帧提取**: VideoTree, VideoAgent

## 评分

- 创新性：⭐⭐⭐⭐⭐ — T-GRPO将GRPO拓展到树结构是RL+视频理解的新方向；自发涌现推理链令人惊喜
- 实用性：⭐⭐⭐⭐ — 直接提升现有VLM的长视频能力，但延迟较高
- 实验充分度：⭐⭐⭐⭐ — 4个基准、10个baseline、聚类/RL消融完整
- 写作质量：⭐⭐⭐⭐ — 工作流图清晰，case study生动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)
- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](../../AAAI2026/video_understanding/tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)
- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[CVPR 2025\] DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos](../../CVPR2025/video_understanding/decafnet_delegate_and_conquer_for_efficient_temporal_grounding_in_long_videos.md)

</div>

<!-- RELATED:END -->
