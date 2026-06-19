---
title: >-
  [论文解读] SEAL: SEmantic Attention Learning for Long Video Representation
description: >-
  [CVPR 2025][视频理解][长视频理解] 提出SEAL统一长视频表征方法，将视频分解为场景/物体/动作三种语义token，通过query感知的子集选择优化来平衡相关性与多样性，在LVBench上以45.9%超越Qwen2-VL-72B的41.3%。 长视频理解面临三大挑战： - 计算复杂度高：小时级视频的帧数和像素量…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "长视频理解"
  - "语义分解"
  - "注意力学习"
  - "视频问答"
  - "时序定位"
---

# SEAL: SEmantic Attention Learning for Long Video Representation

**会议**: CVPR 2025  
**arXiv**: [2412.01798](https://arxiv.org/abs/2412.01798)  
**代码**: 无  
**领域**: Video Understanding  
**关键词**: 长视频理解, 语义分解, 注意力学习, 视频问答, 时序定位

## 一句话总结

提出SEAL统一长视频表征方法，将视频分解为场景/物体/动作三种语义token，通过query感知的子集选择优化来平衡相关性与多样性，在LVBench上以45.9%超越Qwen2-VL-72B的41.3%。

## 研究背景与动机

长视频理解面临三大挑战：
- **计算复杂度高**：小时级视频的帧数和像素量远超现有硬件承载能力
- **时间冗余严重**：场景和物体变化缓慢，大量帧携带重复信息
- **跨任务泛化**：有效表征需同时支持细粒度事实检索和高层推理

现有方法的局限：
- 均匀采样丢失关键信息且产生冗余
- 内存银行方法合并相似帧但仍依赖任务特定设计
- 仅关注单一任务（如QA或时序定位）的模型难以泛化

人脑启发：选择性注意新信息、在线持续更新记忆、根据任务动态调整关注焦点。SEAL据此设计了语义分解+注意力学习的统一框架。

## 方法详解

### 整体框架

SEAL包含两个核心步骤：
1. **语义分解**：将长视频从原始帧分解为场景token $\mathbf{T}_{\text{scene}}$、物体token $\mathbf{T}_{\text{object}}$ 和动作token $\mathbf{T}_{\text{action}}$ 三种压缩语义表示
2. **注意力学习**：基于query的子集选择优化，从全部语义token中选出固定大小的子集，送入视觉头或MLLM头完成下游任务

### 关键设计1：三类语义token分解

- **功能**：将高维密集视频压缩为紧凑的语义实体集合，大幅降低计算量
- **核心思路**：
    - 场景token：均匀采样 $N_{\text{scene}}$ 帧捕获背景环境信息
    - 动作token：用SAM-2等类无关追踪器提取动态轨迹（tracklet），短于 $L_{\min}$ 的丢弃，长于 $L_{\max}$ 的切分，对每条轨迹取帧间bounding box的空间并集
    - 物体token：在关键帧上用SAM进行类无关分割，获取静态物体mask
- **设计动机**：三类token分别回答"在哪里"（场景）、"是什么"（物体）、"怎么做"（动作），覆盖视频理解的核心维度。这种分解比粗暴采样更具信息效率，且与任务无关

### 关键设计2：子集选择注意力学习

- **功能**：从大量候选token中选出兼顾query相关性和token多样性的最优子集
- **核心思路**：形式化为组合优化问题 $T_s^* = \arg\max_{T_s \subset T_G} \alpha \sum_{t_s \in T_s} R(t_s, q) + (1-\alpha) \sum_{t_i, t_j \in T_s, i \neq j} \frac{1}{S(t_i, t_j)}$，其中 $R(\cdot)$ 为BLIP-2计算的token-query余弦相似度，$S(\cdot)$ 为token间余弦相似度
- **设计动机**：单纯按相关性选取会导致token集合高度冗余（都集中在某个区域），加入多样性项确保选出的token覆盖视频的不同方面。超参 $\alpha=0.9$ 平衡两个目标

### 关键设计3：流式与全局双模式

- **功能**：支持任意长度视频的在线处理
- **核心思路**：全局模式一次处理全部token输出统一表征；流式模式用固定大小滑动窗口，每步对当前窗口token和前一步选定子集的并集执行注意力学习：$T_{\text{sub}}^t = \text{Attention\_Learning}(T_t \cup T_{\text{sub}}^{t-1})$
- **设计动机**：全局模式适合离线分析，流式模式支持实时场景（如边看电影边回答问题），使表征不受视频长度限制

### 损失函数

下游任务特定：时序定位使用IoU距离+focal loss训练分类+回归头；视频QA使用MLLM的自回归next-token prediction负对数似然损失。

## 实验关键数据

### 主实验1：LVBench视频QA（小时级视频）

| 模型 | LLM大小 | Overall | KIR | EU | Sum | ER | Rea | TG |
|------|---------|---------|-----|-----|-----|-----|-----|-----|
| Qwen2-VL | 72B | 41.3 | 38.3 | 41.1 | 46.6 | 38.0 | 46.5 | 41.4 |
| InternVL2 | 34B | 39.6 | 43.4 | 39.7 | 41.4 | 37.4 | 42.5 | 31.4 |
| **SEAL** | **34B** | **45.9** | **51.5** | **41.3** | 39.7 | **47.9** | 43.3 | 32.3 |

SEAL以34B模型超越72B的Qwen2-VL 4.6%，在KIR和ER上分别领先8.1%和5.1%。

### 主实验2：Ego4D-NLQ时序定位（有限token约束）

| 模型 | #Token | R@1 IoU=0.3 | R@1 IoU=0.5 | R@5 IoU=0.3 | R@5 IoU=0.5 |
|------|--------|-------------|-------------|-------------|-------------|
| SnAG | 450 | 13.44 | 9.23 | 34.02 | 23.04 |
| **SEAL** | **450** | **13.78** | **9.26** | **34.79** | **23.10** |
| SnAG | 200 | 10.03 | 6.35 | 26.56 | 16.90 |
| **SEAL** | **200** | **10.83** | **7.06** | **27.39** | **17.41** |

### 关键发现

- 语义分解有效降低冗余：参数量更小的模型反而超越更大模型
- 动作和物体token在KIR和ER任务上贡献最大
- 流式模式性能仅略低于全局模式，验证了在线表征更新的可行性
- 不依赖特定LLM架构，统一表征可接不同预测头

## 亮点与洞察

1. **认知科学启发的设计**：三类语义token与人脑对视频的注意力分配机制高度吻合
2. **统一表征跨任务泛化**：同一表征接不同头即可完成QA和时序定位，无需任务特定编码
3. **小模型超大模型**：表明视频理解中"看什么"比"模型多大"更重要，高效的信息选择可弥补参数量差距

## 局限与展望

- 语义分解依赖SAM-2等外部模型的质量，对非常规场景（运动模糊、遮挡严重）可能失效
- 子集选择优化是NP-hard，目前用贪心近似，可能非最优
- 因果推理类问题表现相对较弱（如"why"类型）
- 未探索视频对话/多轮交互场景

## 相关工作与启发

- **MovieChat**：基于Atkinson-Shiffrin记忆模型的长视频方法，SEAL在其数据集也有优势
- **SnAG**：时序定位baseline，SEAL在有限token下一致优于它
- **TimeSformer**：SEAL中时空注意力block的基础架构

## 评分

⭐⭐⭐⭐ — 问题定义清晰，语义分解+注意力学习的设计优雅且实用，以34B模型在LVBench超越72B很有说服力。子集选择的贪心近似和外部模型依赖是主要弱点。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)
- [\[CVPR 2025\] Heterogeneous Skeleton-Based Action Representation Learning](heterogeneous_skeleton-based_action_representation_learning.md)
- [\[CVPR 2025\] MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)
- [\[CVPR 2025\] T*: Re-thinking Temporal Search for Long-Form Video Understanding](re-thinking_temporal_search_for_long-form_video_understanding.md)
- [\[CVPR 2025\] ReWind: Understanding Long Videos with Instructed Learnable Memory](rewind_understanding_long_videos_with_instructed_learnable_memory.md)

</div>

<!-- RELATED:END -->
