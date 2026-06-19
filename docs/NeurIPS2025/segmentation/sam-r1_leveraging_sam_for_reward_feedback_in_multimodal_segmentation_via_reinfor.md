---
title: >-
  [论文解读] SAM-R1: Leveraging SAM for Reward Feedback in Multimodal Segmentation via Reinforcement Learning
description: >-
  [NeurIPS 2025][语义分割][推理分割] SAM-R1 提出了一个端到端的推理分割框架，首次将 SAM 作为强化学习训练回路中的奖励提供者，结合分级IoU精度奖励、非对称裁剪和 token 级损失归一化的改进 GRPO 算法，仅用 3K 训练样本即在 ReasonSeg 零样本设定下超越 Seg-Zero 等方法，gIoU 达 60.2%。
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "推理分割"
  - "强化学习"
  - "SAM"
  - "多模态大模型"
  - "GRPO"
---

# SAM-R1: Leveraging SAM for Reward Feedback in Multimodal Segmentation via Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2505.22596](https://arxiv.org/abs/2505.22596)  
**代码**: 无  
**领域**: 分割  
**关键词**: 推理分割, 强化学习, SAM, 多模态大模型, GRPO

## 一句话总结

SAM-R1 提出了一个端到端的推理分割框架，首次将 SAM 作为强化学习训练回路中的奖励提供者，结合分级IoU精度奖励、非对称裁剪和 token 级损失归一化的改进 GRPO 算法，仅用 3K 训练样本即在 ReasonSeg 零样本设定下超越 Seg-Zero 等方法，gIoU 达 60.2%。

## 研究背景与动机

多模态大语言模型（MLLM）在视觉理解中取得了长足进展，其能力已延伸至推理分割——需要模型理解隐含用户查询并进行逻辑推理以生成像素级分割。LISA 首先证明了通过特殊 token 连接 MLLM 与分割模型的可行性。但现有方法存在几个关键问题：

**数据成本高**：依赖大规模标注数据集做联合微调，获取包含显式推理过程的标注数据更是昂贵

**灾难性遗忘**：SFT 训练的模型在域内表现好但域外泛化差

**推理能力不足**：面对歧义和复杂文本查询时难以准确解读意图

近期研究表明，强化学习可以在无需推理标注数据的情况下赋予大模型推理能力（如 DeepSeek-R1 使用规则奖励）。Seg-Zero 已将 GRPO 引入推理分割，但其推理模型与分割解码器完全解耦，无法获得像素级反馈，增加了奖励欺骗风险。

核心切入：将分割模型（SAM）直接纳入 RL 训练回路作为奖励提供者，使 MLLM 能在端到端框架中接收任务相关的细粒度分割反馈。

## 方法详解

### 整体框架

SAM-R1 的工作流程：
1. **输入**：用户问题 + 图像 → MLLM（Qwen2.5VL-7B）
2. **推理**：MLLM 在 `<think>...</think>` 标签内生成推理链，在 `<answer>...</answer>` 标签内生成结构化答案（含边界框、参考点、文本描述）
3. **分割**：将 MLLM 的输出传递给 SAM2-Large 生成分割掩码
4. **奖励计算**：比较预测掩码与真值掩码的 IoU，计算分级精度奖励
5. **策略更新**：根据奖励优化 MLLM 的推理策略

### 关键设计

1. **分级分割精度奖励**：不同于简单的二元奖励或连续 IoU 值，采用分段式奖励设计：IoU>0.8 得 4 分，0.7-0.8 得 3 分，0.5-0.7 得 2 分，其他 0 分。这种阶梯式奖励在低 IoU 水平提供渐进改善信号，仅在预测区域高度匹配真值时给予强正反馈。

2. **改进的 GRPO 优化目标**：

    - **非对称裁剪**：将 PPO 的单一阈值 ε 替换为非对称边界 ε_low 和 ε_high（ε_high=0.3 更宽松），保持 ε_low 不变。这允许模型在有高优势动作时更积极地更新，同时保持 KL 散度约束以确保训练稳定性
    - **token 级损失归一化**：原始 GRPO 中长短回复承受相同总损失，长回复的每个 token 惩罚更轻，鼓励冗长低信息输出。SAM-R1 将损失归一化因子从 $\frac{1}{G} \frac{1}{|o_i|}$ 改为 $\frac{1}{\sum_{i=1}^{G}|o_i|}$，使每个 token 承受相同损失，抑制冗余重复输出

3. **三维奖励函数设计**：

    - 分级分割精度奖励（基于 SAM 输出的 IoU）
    - 推理格式奖励（检查 think/answer 标签结构）
    - 分割格式奖励（检查边界框、参考点、文本标志的 JSON 格式合规性）

### 损失函数 / 训练策略

使用 Qwen2.5VL-7B 作为基础模型，SAM2-Large 作为分割模型，8×A100 GPU 训练。每个问题采样 8 个响应，学习率 1.0×10⁻⁶。所有图像统一 resize 到 840×840。训练仅使用 RefCOCOg 训练集中随机抽取的 3,000 个样本。

## 实验关键数据

### 主实验 - ReasonSeg 零样本

| 方法 | val gIoU | val cIoU | test gIoU | test cIoU |
|------|----------|----------|-----------|-----------|
| LISA-7B | 53.6 | 52.3 | 48.7 | 48.8 |
| LISA-13B | 57.7 | 60.3 | 53.8 | 50.8 |
| Seg-Zero-7B* | 62.0 | 52.0 | 58.3 | 53.4 |
| **SAM-R1** | **64.0** | **55.8** | **60.2** | **54.3** |

### 指代表达分割 (cIoU)

| 方法 | RefCOCO | RefCOCO+ | RefCOCOg |
|------|---------|----------|----------|
| LISA-7B | 76.5 | 67.4 | 68.5 |
| PixelLM-7B | 76.5 | 71.7 | 70.5 |
| PerceptionGPT-7B | 78.6 | 73.9 | 71.7 |
| Seg-Zero-7B* | 79.2 | 73.9 | 73.3 |
| **SAM-R1** | **79.2** | **74.7** | 73.1 |

### 消融实验 - 阈值策略

| 策略 | ReasonSeg gIoU | ReasonSeg cIoU | RefCOCOg gIoU | RefCOCOg cIoU |
|------|---------------|---------------|---------------|---------------|
| 固定 0.5 | 56.5 | 51.9 | 74.7 | 72.8 |
| 固定 0.7 | 56.2 | 51.6 | 74.9 | 72.6 |
| 固定 0.8 | 58.6 | 50.8 | 74.6 | 71.9 |
| **分级** | **60.2** | **54.3** | **75.4** | **73.1** |

### 消融实验 - 算法组件

| 方法 | Token级 | 高裁剪 | ReasonSeg gIoU | ReasonSeg cIoU | RefCOCOg gIoU | RefCOCOg cIoU |
|------|---------|--------|---------------|---------------|---------------|---------------|
| GRPO基线 | - | - | 57.8 | 51.2 | 74.1 | 71.8 |
| +Token级 | ✓ | - | 58.0 | 51.7 | 74.5 | 72.4 |
| +高裁剪 | - | ✓ | 59.1 | 52.8 | 74.9 | 72.5 |
| **完整** | ✓ | ✓ | **60.2** | **54.3** | **75.4** | **73.1** |

### 关键发现

- 分级阈值策略比任何固定阈值都更有效，特别在 OOD 泛化上显著优于固定 0.8 阈值（gIoU +1.6，cIoU +3.5）
- 非对称高裁剪对 OOD 推理任务帮助更大（ReasonSeg gIoU +1.3 vs RefCOCOg +0.8）
- 两种改进的协同效应显著：合并使用时 ReasonSeg cIoU 提升 3.1%，超过各自单独提升之和
- 3K 训练样本已足够——增加到 10K 后性能几乎不变，展现出极强的数据效率
- 模型在未训练的 LISA-Grounding 基准上达 63.8（前方法最优 43.9），证明泛化能力
- 移除 KL 约束导致训练在约 100 步后崩溃，KL 散度对稳定性至关重要

## 亮点与洞察

- 将 SAM 从"下游分割工具"提升为"RL训练中的奖励提供者"，这一角色转换是核心创新
- 分级 IoU 奖励设计比连续 IoU 更适合 RL 训练——提供清晰的目标梯度层次
- 非对称裁剪 + token 级归一化是对 GRPO 的精细且有效的改进
- 3K 样本即可达到强性能的数据效率极为惊人
- 论文坦诚报告了多个失败尝试（移除 KL 约束崩溃、负参考点生成失败），提供了宝贵的负面结果

## 局限与展望

- SAM 参数冻结，信息流单向——无法根据推理模型的需求自适应调整分割策略
- 模型难以生成有意义的负参考点，RL 框架未能激励这种能力
- 高 IoU 区间仍存在不完整分割和过度分割问题（推理正确但掩码精度不足）
- 仅在 Qwen2.5VL-7B 上实验，未验证对其他 MLLM 基座的适用性
- 未探索更复杂的奖励组合策略（如加入语义一致性奖励）

## 相关工作与启发

- 在 DeepSeek-R1 → VLM-R1 → Seg-Zero 的技术演进链上更进一步，是 RL 增强多模态推理的自然延伸
- 与 Seg-Zero 的关键区别在于端到端整合 vs 解耦设计——端到端方案从根本上减少了奖励欺骗
- 启发方向：联合优化 SAM 和推理模型、设计更复杂的分级奖励体系、将方法推广到视频分割

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning](finers_fine-grained_reasoning_and_segmentation_of_small_objects_with_reinforceme.md)
- [\[CVPR 2025\] HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation](../../CVPR2025/segmentation/hfp-sam_hierarchical_frequency_prompted_sam_for_efficient_marine_animal_segmenta.md)
- [\[CVPR 2025\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation](../../CVPR2025/segmentation/effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] E-SAM: Training-Free Segment Every Entity Model](../../ICCV2025/segmentation/e-sam_training-free_segment_every_entity_model.md)
- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](../../AAAI2026/segmentation/saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)

</div>

<!-- RELATED:END -->
