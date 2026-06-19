---
title: >-
  [论文解读] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension
description: >-
  [ICCV 2025][多模态VLM][视觉语言模型] 提出 Vision Value Model (VisVM)，一个基于时序差分（TD）学习训练的视觉价值模型，用于在推理时指导 VLM 逐句搜索生成更高质量的描述性标注——相比贪心解码和 CLIP-PRM，VisVM 搜索显著减少幻觉（CHAIRs 从 32.4 降至 26.2），且生成的数据用于自训练可在 9 个基准上平均提升 10.8%。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "视觉语言模型"
  - "推理时搜索"
  - "视觉价值模型"
  - "幻觉缓解"
  - "自训练"
  - "时序差分学习"
---

# Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension

**会议**: ICCV 2025  
**代码**: 未公开  
**领域**: 多模态VLM  
**关键词**: 视觉语言模型, 推理时搜索, 视觉价值模型, 幻觉缓解, 自训练, 时序差分学习

## 一句话总结

提出 Vision Value Model (VisVM)，一个基于时序差分（TD）学习训练的视觉价值模型，用于在推理时指导 VLM 逐句搜索生成更高质量的描述性标注——相比贪心解码和 CLIP-PRM，VisVM 搜索显著减少幻觉（CHAIRs 从 32.4 降至 26.2），且生成的数据用于自训练可在 9 个基准上平均提升 10.8%。

## 研究背景与动机

视觉语言模型（VLM）在多模态任务上取得了快速进步，但仍面临两大核心问题：

**视觉幻觉**：模型生成与图像不一致的内容，包括不存在的物体和错误描述

**忽略次要区域**：对图像中不太显著的区域关注不足，导致描述不够详细

在 LLM 领域，推理时搜索（如 OpenAI-O1）已被证明是提升输出质量的有效手段。但将其扩展到 VLM 面临独特挑战：**VLM 任务缺乏清晰的结果度量**。数学和编程问题有明确的对错，但描述性标注是开放式任务——每句话不仅需要局部准确，还需要保持整体连贯。

**现有 PRM 的局限**：传统的过程奖励模型（PRM）仅评估当前步骤的即时奖励。但在图像描述任务中，一句当前看起来不错的话可能在后续步骤中引发幻觉。例如，描述一个复杂场景时，一句"画面中有一辆红色汽车"可能在当前步骤得到高分，但如果后续不断围绕这辆不存在的汽车展开，整体质量会急剧下降。

## 方法详解

### 整体框架

将 VLM 推理建模为马尔可夫决策过程（MDP），每一步生成一句话：

- **状态空间** $\mathcal{S}$：当前已生成的句子 + 图像
- **动作空间** $\mathcal{A}$：当前步骤生成的句子
- **奖励函数** $\mathcal{R}$：CLIP 相似度作为过程奖励
- **折扣因子** $\gamma$

VisVM 在此 MDP 上预测**长期价值**而非即时奖励，从而避免短视决策。

### VisVM 训练（TD Learning）

VisVM 的核心训练目标是通过时序差分学习预测句子的长期价值：

$$L(\rho) = -\mathbb{E}_{(y_i, y_{i+1}, I) \sim \mathcal{D}} \left( r_{s_i} + \gamma V_\rho(y_{i+1}, I) - V_\rho(y_i, I) \right)^2$$

其中：
- $V_\rho(y_i, I)$ 是 VisVM 对状态 $(y_i, I)$ 的长期价值预测
- $r_{s_i}$ 是 CLIP 相似度作为的即时奖励
- $\gamma$ 是折扣因子

**两个关键设计**：
1. **前瞻性连贯**（Forward-looking coherence）：不同于仅看当前句子奖励的 PRM，VisVM 通过 TD 学习预测未来后果，评估长期效果而非即时反应
2. **全面视觉对齐**（Comprehensive visual grounding）：利用 CLIP 的文本-图像相似度让奖励信号捕获丰富的视觉语义

### 模型实现

- 在 LLaVA-Next-Mistral-7B 的倒数第二层之上添加一个线性层作为价值头
- 价值头输出单个标量，代表当前句子的累计奖励/长期价值
- 使用 VLM 自身的视觉编码器（CLIP-ViT / SigLIP）作为 PRM——**自奖励机制**，无需外部模型

### 训练数据构建

- 从 COCO 2017 训练集采样 9,215 张图像
- 使用 LLaVA-150K 的 9 种描述性标注提示
- 对每个图像-提示对使用贪心解码和不同温度的采样生成 5 个不同响应
- 将段落分解为（当前句子, 下一句话, 图像）三元组
- 最终数据集包含 **378K** 个训练样本

### 推理时搜索

在每步搜索中：
1. 使用 $N$ 种不同温度配置（$[0.1, 0.3, 0.5, 0.7, 0.9]$）+ 贪心解码
2. 每种配置采样 $K$ 个候选句子
3. 用 VisVM 估计每个候选的长期价值
4. 选择最高价值的候选作为当前步骤的输出
5. 重复直到生成结束标记

## 实验关键数据

### 主实验：幻觉评估

| 基础模型 | 搜索方法 | CHAIRs ↓ | CHAIRi ↓ | MMHal ↑ | MMHal rate ↓ | AMBER Cov ↑ |
|---|---|---|---|---|---|---|
| LLaVA-Next-7B | Greedy (默认) | 32.4 | 5.9 | 2.94 | 0.52 | 63.9 |
| LLaVA-Next-7B | MCTS | 25.9 | 4.7 | 3.24 | 0.37 | 67.3 |
| LLaVA-Next-7B | BoN (N=30) | 27.1 | 5.2 | 3.06 | 0.45 | 65.3 |
| LLaVA-Next-7B | CLIP-PRM 搜索 | 28.4 | 5.5 | 2.96 | 0.49 | 66.1 |
| LLaVA-Next-7B | **VisVM 搜索** | **26.2** | **4.6** | **3.30** | **0.39** | **66.8** |
| Qwen2-VL-7B | Greedy (默认) | 30.8 | 5.2 | 3.27 | 0.37 | 69.4 |
| Qwen2-VL-7B | **VisVM 搜索** | **24.5** | **3.3** | **3.39** | **0.29** | **73.5** |

### 自训练实验（SFT 后基准提升）

| 基础模型 | SFT 数据来源 | MM-Vet | MMBench | MMMU | MathVista | CVBench | LLAVA-W | MMStar | CHAIRs ↓ | 平均提升 |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-Next-7B | 无（原始模型） | 45.2 | 74.9 | 34.2 | 38.5 | 65.8 | 76.9 | 36.0 | 32.4 | — |
| LLaVA-Next-7B | Greedy 解码 | 43.5 | 74.6 | 34.9 | 37.8 | 66.2 | 75.1 | 36.7 | 33.2 | -1.6% |
| LLaVA-Next-7B | GPT4o-BoN(30) | 47.1 | 76.1 | 35.4 | 40.9 | 67.9 | 77.3 | 36.9 | 30.0 | +4.9% |
| LLaVA-Next-7B | **VisVM 搜索** | **48.3** | **76.7** | **36.1** | **42.3** | **69.8** | **78.4** | **38.0** | **22.6** | **+10.8%** |
| Qwen2-VL-7B | **VisVM 搜索** | **58.9** | **84.1** | **49.7** | **61.1** | **76.2** | **88.2** | **57.0** | **21.4** | **+7.3%** |

### 消融实验：不同 PRM 的影响

| 搜索方法 | CHAIRs ↓ | CHAIRi ↓ | MMHal ↑ | MMHal rate ↓ | AMBER Cov ↑ |
|---|---|---|---|---|---|
| Greedy (默认) | 32.4 | 5.9 | 2.94 | 0.52 | 63.9 |
| CLIP-VisVM 搜索 | 26.2 | 4.6 | 3.30 | 0.39 | 66.8 |
| SigLIP-VisVM 搜索 | **25.6** | **4.4** | **3.31** | **0.36** | **67.5** |

### 关键发现

1. **人类评估验证**：VisVM 搜索 vs Greedy 的胜率高达 75.8%，vs CLIP-PRM 为 62.4%
2. **搜索预算扩展**：VisVM 搜索的效率是 CLIP-PRM 的 2 倍——步长 8 时的 VisVM 性能 ≈ 步长 16 时的 CLIP-PRM
3. **计算效率**：与 MCTS 相比，VisVM 搜索使用约 1/7 的 GPU 小时，但性能相当或更优
4. **更强 PRM → 更好 VisVM**：使用 SigLIP 替代 CLIP 作为 PRM 训练 VisVM，进一步降低幻觉

## 亮点与洞察

1. **将 RL 中的价值函数引入 VLM 推理**：TD 学习预测长期价值而非即时奖励，这是对现有 PRM 的本质改进
2. **自奖励 + 自训练闭环**：PRM 来自 VLM 自身的视觉编码器，SFT 数据也由 VLM 自身生成，真正实现了无需外部标注的自我提升
3. **推理时计算的缩放定律**：证实了在 VLM 视觉理解任务上，推理时计算也符合 scaling law
4. **实践意义**：仅用 9,215 张 COCO 图像的自训练数据，就能在 9 个基准上获得双位数提升

## 局限性

1. **推理成本显著增加**：即使效率优于 MCTS，VisVM 搜索仍需要比贪心解码多出数倍的计算量
2. **仅验证了描述性标注任务**：当前实验集中在图像描述，对于 VQA、推理等结构化任务的效果未探索
3. **奖励信号依赖 CLIP**：CLIP 的语义覆盖有限，对于需要细粒度理解的场景（如文字识别、空间关系）可能不够
4. **单步 = 一句话的粒度**：以句子为步骤的 MDP 建模较粗，如果句子质量差异主要发生在词级别则不够灵活

## 相关工作与启发

- **OpenAI O1** 系列的推理时搜索思想与本文一脉相承，但扩展到了视觉多模态领域
- **RLHF/DPO 的替代路线**：不需要人类偏好标注，而是通过视觉编码器的自奖励来提升
- **对未来自改进 VLM 的启示**：如果 VisVM 能用更强版本的自身替代，可能形成持续自我改进的循环

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐ — 将 RL 价值函数引入 VLM 推理时搜索，思路新颖
- **实验完整性**: ⭐⭐⭐⭐⭐ — 覆盖 3 个基础模型、多种基线、人类评估，且有自训练验证
- **实用性**: ⭐⭐⭐ — 推理成本增加限制了实际部署
- **写作质量**: ⭐⭐⭐⭐ — 思路清晰，公式推导完整

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs](instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)
- [\[ICCV 2025\] GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks](geobench-vlm_benchmarking_vision-language_models_for_geospatial_tasks.md)
- [\[ICCV 2025\] CaptionSmiths: Flexibly Controlling Language Pattern in Image Captioning](captionsmiths_flexibly_controlling_language_pattern_in_image_captioning.md)
- [\[ICCV 2025\] Training-free Generation of Temporally Consistent Rewards from VLMs](training-free_generation_of_temporally_consistent_rewards_from_vlms.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)

</div>

<!-- RELATED:END -->
