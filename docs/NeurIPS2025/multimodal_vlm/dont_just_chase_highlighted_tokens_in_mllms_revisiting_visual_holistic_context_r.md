---
title: >-
  [论文解读] Don't Just Chase "Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention
description: >-
  [NeurIPS 2025][多模态VLM][视觉token剪枝] 提出 HoloV，一个即插即用的视觉 token 剪枝框架，通过在不同空间裁剪区域自适应分配剪枝预算，保留全局视觉上下文而非仅保留注意力高亮 token，在 LLaVA-1.5 上剪枝 88.9% token 仍保留 95.8% 原始性能。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "视觉token剪枝"
  - "推理加速"
  - "整体上下文"
  - "自适应分配"
  - "MLLM效率"
---

# Don't Just Chase "Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention

**会议**: NeurIPS 2025  
**arXiv**: [2510.02912](https://arxiv.org/abs/2510.02912)  
**代码**: [GitHub](https://github.com/obananas/HoloV)  
**领域**: 多模态VLM  
**关键词**: 视觉token剪枝, 推理加速, 整体上下文, 自适应分配, MLLM效率

## 一句话总结

提出 HoloV，一个即插即用的视觉 token 剪枝框架，通过在不同空间裁剪区域自适应分配剪枝预算，保留全局视觉上下文而非仅保留注意力高亮 token，在 LLaVA-1.5 上剪枝 88.9% token 仍保留 95.8% 原始性能。

## 研究背景与动机

MLLM（如 LLaVA-1.5 将一张 336 分辨率图像编码为 576 个视觉 token，LLaVA-OneVision 更需要 7,290 个）面临严重的视觉 token 冗余问题。现有注意力优先（attention-first）的剪枝方法（如 FastV）存在三个关键缺陷：

**信息冗余**: 基于注意力评分的选择倾向于保留语义相似的相邻 token，在高剪枝率下导致信息重复

**位置偏差**: Transformer 的位置编码使序列首尾位置的 token 获得更高注意力权重，但图像目标通常位于中心区域

**注意力分散**: 文本-视觉交叉注意力分散在大量 token 上，top 20% token 仅占总注意力的 40%

作者通过两个重要观察验证了整体上下文的重要性：

- **随机剪枝 > FastV**: 在超过半数基准上，随机保留 token（保持空间多样性）优于注意力排序选择
- **全局缩略图 > 局部裁剪**: 仅使用全局缩略图在 MMBench、MME 等通用感知基准上即可取得强结果

## 方法详解

### 整体框架

HoloV 在 LLM decoder 之前执行 token 剪枝，通过以下步骤保留整体视觉上下文：

1. 将 $N_v$ 个视觉 token 均分为 $\mathcal{C}$ 个空间裁剪区域
2. 计算每个区域的整体得分（融合语义多样性 + 注意力显著性）
3. 根据区域重要性自适应分配保留配额
4. 各区域内部按得分选择 top-k token

### 关键设计

**裁剪内语义多样性**: 对每个裁剪区域 $c$ 中的归一化嵌入 $\mathbf{Z}_v^c$，计算余弦相似度矩阵（排除自相似）：

$$\mathbf{S}^c = (\mathbf{1} - \mathbf{I}_M) \odot \mathbf{Z}_v^c {\mathbf{Z}_v^c}^\top$$

然后求每个 token 的语义分布方差：

$$\mathcal{V}_i^c = \frac{1}{M-1} \sum (\mathbf{S}_{i,j}^c - \mu_i^c)^2$$

高 $\mathcal{V}_i^c$ 表示该 token 与其他 token 有多样化的连接关系，是信息含量高的 token。

**整体得分融合**: 将语义多样性 $\mathcal{V}^c$ 与 [CLS] 注意力 $\mathcal{A}^c$ 通过自适应缩放合并：

$$\mathcal{H}^c = \gamma_c \mathcal{V}^c + \mathcal{A}^c, \quad \gamma_c = \mathbb{E}[\|\mathcal{A}^c\|] / \mathbb{E}[\|\mathcal{V}^c\|]$$

**自适应配额分配**: 根据区域整体重要性分配保留 token 数量：

$$w_c = \frac{(\frac{1}{M} \sum_{t=1}^M \mathcal{H}_t^c)^\tau}{\sum_{c'=1}^{\mathcal{C}} (\frac{1}{M} \sum_{t=1}^M \mathcal{H}_t^{c'})^\tau}$$

初始配额 $q_c = \lfloor w_c \hat{N}_v \rfloor$，溢出和不足通过迭代重分配解决。

**视觉上下文快速补充**: 被剪枝的 token 在中间触发层通过 FFN 作为 "key-value memory" 重新注入 MLLM，仅在模型推理不确定性高时触发。

### 损失函数

HoloV 是训练无关（training-free）的即插即用方法，不需要额外损失函数。其 FLOPs 减少比例的理论分析为：

$$F \approx 1 - (1-R)^2 = 2R - R^2$$

其中 $R$ 是 token 减少比例。在解码阶段（有 KV cache），减少近似与 $R$ 成线性关系。

## 实验关键数据

### 主实验

**LLaVA-1.5 7B 不同剪枝率下的性能（9 个基准平均）**:

| 方法 | 保留 192 (↓66.7%) | 保留 128 (↓77.8%) | 保留 64 (↓88.9%) |
|------|-------------------|-------------------|------------------|
| FastV (ECCV24) | 90.5% | 85.4% | 76.7% |
| MustDrop | 97.2% | 95.7% | 90.1% |
| VisionZip (CVPR25) | 98.1% | 97.2% | 94.5% |
| DART (EMNLP25) | 98.5% | 97.5% | — |
| SparseVLM (ICML25) | 96.1% | 93.8% | — |
| **HoloV (Ours)** | **99.2%** | **98.0%** | **95.8%** |

**详细基准结果（保留 64 tokens，↓88.9%)**:

| 方法 | GQA | MMB | MME | POPE | VQAv2 | TextVQA | Average |
|------|-----|-----|-----|------|-------|---------|---------|
| FastV | 46.1 | 48.0 | 1256 | 48.0 | 55.0 | 47.8 | 76.7% |
| VisionZip | 55.1 | 60.1 | 1690 | 77.0 | 72.4 | 55.5 | 94.5% |
| **HoloV** | **57.7** | **63.9** | **1802** | **84.0** | **75.5** | **56.8** | **95.8%** |

### 消融实验

**随机策略 vs FastV 的验证**:

| 基准 | FastV 胜 | Random 胜 |
|------|---------|-----------|
| TextVQA | ✓ | |
| MMBench | | ✓ |
| MME | | ✓ |
| POPE | | ✓ |
| GQA | | ✓ |

**全局 vs 局部输入对比**: 仅全局缩略图在 MMBench（通用感知）上表现强，仅局部裁剪在 TextVQA（细粒度感知）上表现好，验证了需要同时保留两者。

### 关键发现

1. 在 88.9% 的极高剪枝率下，HoloV 仍保留 95.8% 原始性能，FastV 仅 76.7%
2. 注意力优先方法在高剪枝率下性能急剧下降的根本原因是位置偏差和注意力分散
3. 整体上下文（全局空间多样性）对通用视觉理解至关重要，局部显著性对细粒度感知不可或缺
4. HoloV 兼容 Flash-Attention 等硬件加速，适合实际部署

## 亮点与洞察

- **重新思考 token 重要性**: 挑战了"注意力高 = 信息量大"的隐含假设，指出空间语义多样性同样重要
- **简洁有效的设计**: 核心思想（保留全局上下文）简单直觉，实现为即插即用模块，无需重训练
- **全面的动机分析**: 通过位置偏差分析、注意力分散统计、随机 vs 注意力对比等多角度验证动机
- **理论支撑**: 基于 Lipschitz 连续性给出了语义差异的理论上界
- 本质洞察：图像中分散的 token 组合（如"雪"+"滑雪"+"山丘"）共同构成语义理解，不应被孤立评估

## 局限性

- 自适应分配中的超参数 $\tau$（温度系数）需要调整
- 视觉上下文补充（refetching）的触发条件（不确定性阈值）可能需要任务自适应
- 主要在 LLaVA 系列上验证，对其他架构（如 Qwen-VL、InternVL）的验证有限
- 裁剪区域的均匀划分可能不适合目标位置高度不均匀的场景

## 相关工作与启发

- **FastV (ECCV2024)**: 基于注意力排序的 token 剪枝基线，HoloV 的核心对比对象
- **LLaVA-PruMerge (ICCV2025)**: 基于 key 相似度的选择性保留与合并
- **VisionZip (CVPR2025)**: 表现仅次于 HoloV 的竞争方法
- **认知科学**: 人类视觉系统通过整合局部特征和全局场景线索形成完整语义理解
- 启发：MLLM 效率优化不应仅关注"减少多少 token"，更应关注"保留哪些 token"

## 评分

- ⭐ 新颖性: 4/5 — 从整体上下文视角重新审视 token 剪枝，动机分析深入
- ⭐ 实验充分度: 5/5 — 9 个基准、多种剪枝率、多种架构、丰富的消融和可视化分析
- ⭐ 写作质量: 4/5 — 图表精美，论证层层递进，分析深入
- ⭐ 价值: 4/5 — 提供了简洁实用的推理加速方案，对 MLLM 部署有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PEACE: Empowering Geologic Map Holistic Understanding with MLLMs](../../CVPR2025/multimodal_vlm/peace_empowering_geologic_map_holistic_understanding_with_mllms.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](../../ICCV2025/multimodal_vlm/mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[CVPR 2026\] Hugging Visual Prompt and Segmentation Tokens: Consistency Learning for Fine-Grained Visual Understanding in MLLMs](../../CVPR2026/multimodal_vlm/hugging_visual_prompt_and_segmentation_tokens_consistency_learning_for_fine-grai.md)
- [\[ICCV 2025\] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning](../../ICCV2025/multimodal_vlm/from_holistic_to_localized_local_enhanced_adapters_for_efficient_visual_instruct.md)
- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)

</div>

<!-- RELATED:END -->
