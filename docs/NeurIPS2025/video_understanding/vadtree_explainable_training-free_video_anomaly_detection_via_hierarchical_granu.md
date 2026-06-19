---
title: >-
  [论文解读] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity
description: >-
  [NeurIPS 2025][视频理解][视频异常检测] 提出 VADTree，一种训练无关的视频异常检测框架，利用预训练的通用事件边界检测（GEBD）模型构建层次粒度感知树（HGTree），实现对不同时间跨度异常事件的自适应采样和多粒度推理，在 UCF-Crime、XD-Violence 和 MSAD 三个基准上取得训练无关方法SOTA，甚至超越部分弱监督方法。
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "视频异常检测"
  - "训练无关"
  - "层次粒度树"
  - "通用事件边界检测"
  - "多粒度推理"
---

# VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity

**会议**: NeurIPS 2025  
**arXiv**: [2510.22693](https://arxiv.org/abs/2510.22693)  
**代码**: [GitHub](https://github.com/wenlongli10/VADTree)  
**领域**: 可解释性  
**关键词**: 视频异常检测, 训练无关, 层次粒度树, 通用事件边界检测, 多粒度推理

## 一句话总结

提出 VADTree，一种训练无关的视频异常检测框架，利用预训练的通用事件边界检测（GEBD）模型构建层次粒度感知树（HGTree），实现对不同时间跨度异常事件的自适应采样和多粒度推理，在 UCF-Crime、XD-Violence 和 MSAD 三个基准上取得训练无关方法SOTA，甚至超越部分弱监督方法。

## 研究背景与动机

视频异常检测（VAD）旨在定位视频中的异常事件，广泛应用于自动驾驶和工业制造等领域。传统方法（全监督、弱监督、无监督）依赖训练数据且缺乏可解释性。近期训练无关方法利用VLM和LLM的知识进行可解释异常检测，但普遍采用**固定长度时间窗口**采样策略。**核心矛盾**：固定时间窗口无法适应现实中异常事件千差万别的持续时间——短至几秒的交通事故、长达几分钟的入室盗窃。窗口太短会丢失长程上下文，太长则引入无关语义噪声。本文的核心idea是利用预训练GEBD知识构建层次化事件树结构，实现自适应多粒度异常检测。

## 方法详解

### 整体框架

VADTree 由三大模块组成：

- **输入**：长视频序列 $V = \{I_t\}_{t=1}^{T}$
- **输出**：帧级异常分数
- **Pipeline**：①GEBD模型生成边界置信度序列 → ②构建粒度感知二叉树并分层 → ③逐节点生成描述+异常评分 → ④簇内精炼+簇间融合 → ⑤帧级异常分数

### 关键设计

1. **层次粒度感知树（HGTree）构建**:

    - **边界置信度序列生成**：将长视频用滑动窗口切成短片段，送入预训练的 EfficientGEBD 模型，保留每个窗口中心区域的边界置信度，拼接为全局序列 $C$，取局部极大值得到候选边界集 $\hat{C}$
    - **通用事件节点初始化**：以整个视频为根节点，递归地在置信度最高的边界处分裂，构建二叉树 $\mathcal{T}$。每个节点 $\mathcal{N}_i = (\hat{c}_l^{(i)}, \hat{c}_r^{(i)}, V_{l:r}^{(i)})$ 记录左右边界置信度和对应视频片段
    - **自适应节点分层**：用 K-Means（K=2）将边界置信度聚成粗粒度（高置信边界）和细粒度（低置信边界）两个簇。去除冗余祖先节点（RemoveDup），补全不可再分的叶节点（Complete），最终得到 $\mathcal{T}' = \{\mathcal{S}'_{coarse}, \mathcal{S}'_{fine}\}$，两个簇各自覆盖完整视频

2. **先验注入的节点异常评分**:

    - 利用LLM生成三维先验知识：场景先验 $b_{scene}$、物体先验 $b_{obj}$、行为先验 $b_{act}$，同时排除VLM无法感知的微表情和音频线索
    - 将先验注入VLM提示，对每个节点的采样帧生成内容描述：$d_u^g = f_{VLM}(V_u^g, B \circ P_d)$
    - LLM基于描述进行异常评分（0-1离散值）：$a_u^g = f_{LLM}(d_u^g, P_s)$

3. **簇内节点精炼（Intra-cluster Node Refinement）**:

    - 独立节点评分缺乏长程上下文，容易产生局部误报
    - 用 ImageBind 视觉编码器提取节点特征，计算余弦相似度
    - 对每个节点取 top-K 最相似节点，用softmax加权平均精炼异常分数：$\hat{a}_u^g = \sum_{i=1}^{K} a_{\kappa_u^{(i)}} \cdot \frac{\exp(\text{sim}(u, \kappa_u^{(i)})/\tau)}{\sum_j^K \exp(\text{sim}(u, \kappa_u^{(j)})/\tau)}$

4. **簇间节点关联（Inter-cluster Node Correlation）**:

    - 对每个粗粒度父节点，计算其子节点异常分数的方差（内聚度 $w_i$）
    - 方差低→子节点语义一致→父节点主导融合；方差高→子节点冲突→依赖细粒度
    - 最终帧级分数：$\bar{a}_{n_{ij}} = \frac{1}{2}(1 - \beta\hat{w}_i)\hat{a}_{n_i} + \frac{1}{2}(1 + \beta\hat{w}_i)\hat{a}_{n_{ij}}$

### 损失函数 / 训练策略

本方法完全训练无关，不涉及任何参数更新。推理时使用 LLaVA-Video-7B 作为 VLM（64帧输入），DeepSeek-R1-Distill-Qwen-14B 作为 LLM（开启思考模式以增强推理），ImageBind 作为视觉编码器。关键超参：$\gamma_{min} = 0.4$（边界置信度阈值），$\beta \in [-1, 1]$（融合控制系数）。

## 实验关键数据

### 主实验

| 数据集 | 指标 | VADTree | 之前SOTA(训练无关) | 提升 | 弱监督最强 |
|--------|------|---------|-------------------|------|-----------|
| UCF-Crime | AUC(%) | **84.74** | SUVAD 83.90 | +0.84 | GS-MoE 91.58 |
| XD-Violence | AUC(%) | **90.44** | EventVAD 87.51 | +2.93 | GS-MoE 94.52 |
| XD-Violence | AP(%) | **67.82** | SUVAD 70.10 | -2.28 | π-VAD 85.37 |
| MSAD | AUC(%) | **89.32** | -- | -- | π-VAD 88.68 |

### 消融实验

| 配置 | AUC(%) | 说明 |
|------|--------|------|
| HGTree Fine Cluster (baseline) | 71.57 | 仅细粒度簇+简单评分 |
| + Prior-infused Node Scoring | 75.67 | 先验知识提升4.1% |
| + Intra-cluster Node Refinement | 83.05 | 簇内精炼提升7.4% (最大增益) |
| + Inter-cluster Node Correlation | 84.74 | 簇间融合提升1.7% |
| γ_min=0.3 (single cluster) | 80.89 | 过度分割 |
| γ_min=0.4 (single cluster) | 82.81 | 单簇最优 |
| γ_min=0.4 (Coarse+Fine) | 84.74 | 层次结构额外提升1.9% |
| K-Medoids替代K-Means | 85.24 | 更鲁棒的聚类+0.5% |

### 关键发现

1. **簇内节点精炼贡献最大**（+7.4%），说明利用语义相似节点抑制VLM/LLM幻觉和噪声至关重要
2. **层次结构优于单簇**（+1.9%），验证了粗细粒度协同推理的必要性
3. **HGTree的mIoU显著优于固定窗口**：在XD-Violence上从0.44提升到0.64，证明自适应采样对长异常事件尤为重要
4. **在MSAD上超越所有弱监督方法**（AUC 89.32% vs π-VAD 88.68%），说明框架泛化能力极强
5. 换用不同VLM/LLM组合性能波动不大（83.56%~84.74%），框架对模型选择鲁棒

## 亮点与洞察

- 将GEBD预训练知识引入VAD是非常自然且有效的思路——事件边界天然对应异常事件的起止
- 层次树结构设计精巧：粗粒度捕捉全局上下文，细粒度精确定位，两者通过方差动态加权融合
- 簇内节点精炼的设计类似图上的消息传递，有效利用了节点间的语义关系
- 完全无需训练即超越弱监督方法（MSAD），展示了"大模型+好的结构化推理"的巨大潜力

## 局限与展望

- GEBD模型本身的质量直接影响树结构质量，如果事件边界检测有偏差，后续推理也会受影响
- 推理开销较大：需要对每个节点分别调用VLM和LLM，实际部署成本高
- 仅支持两层粗细粒度，三层及以上的多粒度结构可能更适合极长视频
- AP指标上在XD-Violence未超过SUVAD，说明精准定位方面仍有提升空间

## 相关工作与启发

- HGTree的设计与VideoTree（ECCV 2024）有类似思路，都是基于事件的层次化视频表示
- 簇内精炼机制可以推广到其他需要抑制VLM幻觉的场景
- GEBD+VAD的组合思路可扩展到视频摘要、视频问答等需要事件感知的任务
- 方差驱动的粗细融合策略是一种通用的层次决策融合方法

## 评分

- 新颖性: ⭐⭐⭐⭐ 将GEBD引入VAD构建事件树是新颖设计，但层次化表示本身并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多种消融、不同模型组合、定性分析齐全
- 写作质量: ⭐⭐⭐⭐ 公式和流程清晰，但符号稍多
- 价值: ⭐⭐⭐⭐ 训练无关VAD的重要进展，MSAD上超弱监督方法意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)
- [\[AAAI 2026\] HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection](../../AAAI2026/video_understanding/headhunt-vad_hunting_robust_anomaly-sensitive_heads_in_mllm_.md)
- [\[CVPR 2026\] Fine-VAD: Towards Fine-Grained Video Anomaly Detection via Progressive Cross-Granularity Learning](../../CVPR2026/video_understanding/fine-vad_towards_fine-grained_video_anomaly_detection_via_progressive_cross-gran.md)
- [\[CVPR 2025\] Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity](../../CVPR2025/video_understanding/holmes-vau_towards_long-term_video_anomaly_understanding_at_any_granularity.md)
- [\[CVPR 2025\] VideoGEM: Training-Free Action Grounding in Videos](../../CVPR2025/video_understanding/videogem_training-free_action_grounding_in_videos.md)

</div>

<!-- RELATED:END -->
