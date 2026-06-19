---
title: >-
  [论文解读] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering
description: >-
  [NeurIPS 2025][多模态VLM][细粒度VQA] 提出 FOCUS，一种无需训练的视觉裁剪方法，利用 MLLM 内部 KV-cache 中 value 特征的余弦相似度构建目标相关性图，高效定位问题相关的图像区域，在细粒度 VQA 上实现与 SOTA 可比的精度，同时计算效率提升 3-6.5 倍。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "细粒度VQA"
  - "视觉裁剪"
  - "KV-Cache"
  - "目标定位"
  - "MLLM"
---

# FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering

**会议**: NeurIPS 2025  
**arXiv**: [2506.21710](https://arxiv.org/abs/2506.21710)  
**代码**: [https://focus-mllm-vqa.github.io](https://focus-mllm-vqa.github.io) (项目页)  
**领域**: 多模态VLM  
**关键词**: 细粒度VQA, 视觉裁剪, KV-Cache, 目标定位, MLLM

## 一句话总结

提出 FOCUS，一种无需训练的视觉裁剪方法，利用 MLLM 内部 KV-cache 中 value 特征的余弦相似度构建目标相关性图，高效定位问题相关的图像区域，在细粒度 VQA 上实现与 SOTA 可比的精度，同时计算效率提升 3-6.5 倍。

## 研究背景与动机

**领域现状**：MLLM 在 VQA 任务上表现出色，但面对高分辨率图像中的细小目标时性能受限。全局视图 MLLM（如 LLaVA-1.5，仅支持 336x336）会下采样导致信息丢失；全局-局部视图 MLLM（如 LLaVA-OneVision）虽保留了局部 crop，但难以从大量视觉 token 中精准找到与问题相关的少数 token。

**现有痛点**：已有视觉裁剪方法各有不足——SEAL 需要任务特定微调；DC2 和 ZoomEye 采用穷举层级搜索效率极低（ZoomEye 每个候选区域需 3 次 forward pass）；ViCrop 依赖完整 Q-K 注意力权重，与 FlashAttention 不兼容。

**核心矛盾**：如何在无需额外训练、不穷举搜索、且兼容高效注意力实现的前提下，精准定位图像中与问题相关的细小目标区域。

**切入角度**：MLLM 推理时的 KV-cache 中已隐含视觉 token 与文本 token 的语义对应关系。目标物体对应的文本 token 和图像 token 在 value 特征空间中应具有高相似度，可直接从中提取空间定位信息，且不引入额外计算开销。

**核心 idea**：用 KV-cache 中 value 特征的余弦相似度替代传统注意力权重构建目标相关性图，实现无训练、高效、FlashAttention 兼容的细粒度目标定位。

## 方法详解

### 整体框架

FOCUS 分为四步：(1) 从 VQA 问题中用 ICL 提取目标物体名称；(2) 利用 KV-cache 中的 value 特征计算目标相关性图；(3) 基于相关性图提出候选 ROI 并排序；(4) 用最高置信度的区域执行最终 VQA。整个过程无需任何额外训练或微调，仅需标准 MLLM 推理。

### 关键设计

1. **目标物体提取**：

    - 利用 MLLM 的 in-context learning 能力，通过 few-shot 示例提示从 VQA 问题中提取需要关注的目标物体名称
    - 可提取单个或多个目标物体，后续为每个目标分别构建相关性图

2. **V-V 伪注意力与目标相关性图**：

    - 对每个目标 token 与所有视觉 token 在第 l 层计算余弦相似度，reshape 为 a x a 的空间图
    - 跨层聚合：用注意力 rollout 加残差连接聚合第 l 到第 L 层的信息
    - 多 token 交集：不同 target token 间用逐元素乘法取交集，确保只有同时匹配所有 token 的区域被保留（如 "red car" 只保留红色+车共现区域）
    - 设计动机：传统 Q-K 注意力权重在 FlashAttention 下不可用，而 value 特征已在推理必须的 KV-cache 中，零额外计算开销
    - 对全局-局部视图 MLLM，使用局部 crop 的视觉 token 来计算伪注意力，经验上能更好捕捉细粒度细节

3. **候选区域提出与排序**：

    - 选取相关性图中 top-k 最高分位置为锚点（保证最小间距）
    - 每个锚点生成初始 ROI（最小尺寸），向外扩展直至最大尺寸或平均相关性低于阈值
    - 用 NMS 去重后，对 top-n_steps 个 ROI 向 MLLM 询问"该区域是否存在目标"并计算存在置信度，据此重排序
    - 设计动机：相关性图可能有噪声（spurious high-activation tokens），需二次验证确认 ROI 中确实包含目标

4. **最终 VQA 推理**：

    - Type-1 问题（单目标）：选最高置信度 ROI 执行 VQA；涉及多个目标物体则合并各自最佳 ROI
    - Type-2 问题（多实例）：选取所有置信度超过阈值的 ROI
    - 对全局-局部视图 MLLM，利用 text-image-interleaved 能力，同时提供标注了目标位置的全局图和各目标的最佳 ROI

### 训练策略

无需训练。所有操作在推理时完成，通过 n_steps（1-8）控制计算预算。LLaVA-1.5 使用第 14-32 层，LLaVA-OneVision 使用第 14-28 层。

## 实验关键数据

### 主实验（LLaVA-1.5-7B）

| 数据集 | FOCUS Acc | ZoomEye Acc | 效率提升 |
|--------|-----------|-------------|---------|
| V*Bench | 72.77% | 77.48% | 3.43x |
| HRBench-4K | 51.75% | 49.75% | 4.39x |
| HRBench-8K | 45.00% | 49.00% | 4.72x |

### 主实验（LLaVA-OneVision-7B）

| 数据集 | FOCUS Acc | ZoomEye Acc | Vanilla |
|--------|-----------|-------------|---------|
| V*Bench | 92.15% | 89.53% | 74.46% |
| HRBench-4K | 72.00% | 68.50% | 58.00% |
| HRBench-8K | 66.50% | 64.75% | 56.25% |

### MME-RealWorld-Lite（LLaVA-OV-7B）

| 方法 | 感知 Acc | 推理 Acc | 感知 FP | 推理 FP |
|------|---------|---------|---------|---------|
| Vanilla | 52.01% | 40.93% | - | - |
| ZoomEye | 56.29% | 43.20% | 41.60 | 45.95 |
| FOCUS | 54.15% | 44.53% | 7.71 | 8.21 |

FOCUS 在推理任务上优于 ZoomEye，感知稍弱，但效率高 5.47 倍。

### Qwen-2.5-VL-7B 验证

| 数据集 | Vanilla | FOCUS |
|--------|---------|-------|
| V*Bench | 79.06% | 90.58% |
| HRBench-4K | 71.62% | 79.25% |
| HRBench-8K | 68.62% | 76.25% |

验证了 FOCUS 对不同 MLLM 架构的泛化能力。

### 消融实验

| 配置 | V*Bench Acc | V*Bench Recall | HRBench-4K Acc |
|------|------------|----------------|----------------|
| Full FOCUS | 72.77% | - | 51.75% |
| 随机相关性图 + 有排序 | 48.68% | 18.37% | 36.13% |
| 有相关性图 + 无排序 | 51.30% | 38.48% | 41.13% |
| K-K 伪注意力 (去 RoPE) | 69.10% | 63.47% | 45.63% |
| Layer 0-14 | 66.49% | 76.17% | 47.38% |
| Layer 0-32 | 71.20% | 75.56% | 49.38% |
| Layer 14-32（默认） | 72.77% | - | 51.75% |

### 关键发现

- 目标相关性图和 ROI 排序两个模块均不可或缺：去掉相关性图精度降约 24pp，去掉排序降约 21pp
- 即使用随机相关性图，排序机制仍远高于随机猜测（48.68% vs 35.99%），说明排序本身具有鲁棒性
- V-V 特征优于去除 RoPE 的 K-K 特征：key 特征中 RoPE 引入位置旋转导致邻近 token 余弦相似度虚高，去除 RoPE 又破坏语义完整性
- 后层表示（14-32）优于前层（0-14）和全层（0-32），与 Logit Lens 发现一致——后层编码更具语义判别性
- 大物体数据集上性能损失可控：A-OKVQA 仅降 3.23pp，GQA 降 1.63pp
- 超参数鲁棒性好：LLaVA-1.5 最大变化 4.71pp，LLaVA-OV 仅 2.62pp

## 亮点与洞察

- **KV-cache 的新用途**：推理时已有的 KV-cache value 特征被巧妙用于目标定位，零额外存储开销且天然兼容 FlashAttention，是典型的"免费午餐"式设计
- **V-V 伪注意力**：用 value-value 余弦相似度替代 Q-K 注意力权重，规避了高效注意力实现不输出注意力矩阵的问题；同时揭示了 value 特征比 key 特征更适合做语义相似度度量（因不受 RoPE 干扰），对注意力机制理解有启发意义
- **多 token 交集过滤**：逐元素乘法聚合多个 target token 的相关性图，用 AND 语义确保只有同时满足所有文本条件的区域被保留，思路简洁且有效
- **效率优势来源清晰**：ZoomEye 每个候选区域需 3 次 FP 且穷举搜索；FOCUS 仅 1 次 FP 即可构建全局相关性图，搜索是 informed 而非 exhaustive 的

## 局限性

- 受限于 MLLM 内部表示的空间分辨率：LLaVA-1.5 仅产生 24x24 相关性图，面对 8K 图像中极小目标可能无法检测
- 继承基座 MLLM 对空间关系理解的不足（如"在图像左侧/右侧"），无训练方法无法改善这一固有缺陷
- 在大物体数据集（如 GQA 用 LLaVA-OV）上降幅达 10.99pp，裁剪可能过度丢弃对大物体有用的全局上下文
- 目标物体提取依赖 ICL，复杂多目标场景可能提取不准确

## 相关工作对比

- **vs ZoomEye**：核心差异在于搜索策略——ZoomEye 穷举层级树搜索且每个候选区域 3 次 FP，FOCUS 用 KV-cache 构建相关性图做 informed search 仅需 1 次 FP 定位，效率提升 3-6.5x
- **vs ViCrop**：ViCrop 的 rel-attn 和 attn-grad 变体依赖完整 Q-K 注意力权重或梯度，与 FlashAttention 不兼容；FOCUS 用 V-V 伪注意力完美兼容现代高效推理框架
- **vs DC2**：DC2 通过 MLLM 为每个区域生成 caption 来判断是否包含目标，计算开销巨大；FOCUS 直接从内部表示获取空间信息，无需额外文本生成
- **vs SEAL**：SEAL 需额外解码器和任务特定微调预测 heatmap，FOCUS 完全无训练即插即用

## 启发与关联

- KV-cache 中隐含空间信息的发现可迁移至多种任务：无训练开放词汇检测、图像编辑区域定位、视频时空定位等
- V-V 伪注意力可作为 FlashAttention 场景下注意力可视化和解释性分析的通用替代方案
- 多 token 交集过滤的 AND 语义是处理组合属性查询的简洁范式，可用于多属性检索和组合推理

## 评分

- 新颖性: ⭐⭐⭐⭐ 利用 KV-cache value 特征做目标定位是新颖视角，但整体仍是"先定位再回答"的标准范式
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集、三种模型架构、详细消融和超参分析，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机推导清晰，方法描述精确，图表设计优秀，与竞品对比公允
- 价值: ⭐⭐⭐⭐ 解决了实际的效率痛点，方法简洁优雅且即插即用，对工业部署有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[CVPR 2025\] FLAIR: VLM with Fine-grained Language-informed Image Representations](../../CVPR2025/multimodal_vlm/flair_vlm_with_fine-grained_language-informed_image_representations.md)
- [\[NeurIPS 2025\] VITRIX-CLIPIN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions](vitrix-clipin_enhancing_fine-grained_visual_understanding_in_clip_via_instructio.md)
- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](benchmarking_retrievalaugmented_multimodal_generation_for_do.md)

</div>

<!-- RELATED:END -->
