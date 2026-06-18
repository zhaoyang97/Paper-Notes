---
title: >-
  [论文解读] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment
description: >-
  [多模态VLM] 揭示了 LVLM 中文本引导视觉token重要性评估的三种跨模态失配问题（因果、语义、空间），提出 VisionDrop——一个仅依赖视觉自注意力的免训练渐进式token剪枝框架，跨视觉编码器和 LLM 解码器多阶段压缩，在保留 5.6% token 时仍能维持 91%+ 原始性能。
tags:
  - "多模态VLM"
---

# Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment

- **会议**: AAAI 2026
- **arXiv**: [2506.22283](https://arxiv.org/abs/2506.22283)
- **代码**: [https://github.com/Ruixxxx/VisionDrop](https://github.com/Ruixxxx/VisionDrop)
- **领域**: 多模态VLM
- **关键词**: 视觉token压缩, 跨模态对齐失配, 免训练剪枝, 注意力评分, 大视觉语言模型

## 一句话总结

揭示了 LVLM 中文本引导视觉token重要性评估的三种跨模态失配问题（因果、语义、空间），提出 VisionDrop——一个仅依赖视觉自注意力的免训练渐进式token剪枝框架，跨视觉编码器和 LLM 解码器多阶段压缩，在保留 5.6% token 时仍能维持 91%+ 原始性能。

## 背景与动机

大视觉语言模型（LVLM）将图像编码为密集的 patch-level token 序列以捕捉细粒度语义，但视觉token数量远超文本token（如 LLaVA-NeXT 单张图像产生 2880 个token），导致注意力计算的二次增长和推理效率瓶颈。

现有 LLM 内部的视觉token剪枝方法（如 FastV、PyramidDrop）大多依赖**文本引导的评分策略**——用文本token对视觉token的注意力来衡量重要性。然而这隐含假设了视觉和文本模态在 LLM 层内始终保持良好对齐，而本文揭示这一假设是不成立的。

作者发现三种跨模态失配：
1. **因果失配（Causal）**：自回归 LLM 的因果注意力使最后一个文本token倾向关注序列末尾的视觉token，形成位置偏置
2. **语义失配（Semantic）**：随着 token 在 LLM 中传播，视觉和文本表征深度纠缠，文本 token 不再能清晰反映视觉重要性
3. **空间失配（Spatial）**：视觉和文本token被展平为单一序列，位置编码混合，文本本身缺乏空间感知，导致空间相关区域被错误丢弃

控制实验表明：将 PyramidDrop 的文本引导评分替换为视觉自注意力评分后，在 GQA、MMBench 等基准上一致取得更好性能，且压缩比越高优势越明显。

## 方法详解

### 1. 渐进式主导 Token 选择（Progressive Dominant Token Selection）

将 LVLM 架构划分为 $N$ 个阶段 $\mathcal{S} = \{s_0, s_1, \ldots, s_N\}$，涵盖视觉编码器和 LLM 解码器。在每个阶段 $s_n$ 末尾，按阶段特定的保留比例 $\lambda_n$ 进行剪枝。

**核心思想**：不依赖文本信号，仅通过视觉-to-视觉的自注意力评估token重要性。

对于视觉 query token $\mathbf{x}_V^q \in \mathbb{R}^{L_1 \times D}$，计算注意力矩阵：

$$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D}}\right)$$

提取视觉key token对应的注意力权重 $\mathbf{A}_{:,\mathcal{V}}$，对所有视觉 query 取平均得到重要性评分：

$$\mathbf{S} = \frac{1}{L_1}\sum_{l=1}^{L_1}\mathbf{A}[l, \mathcal{V}]$$

根据阈值 $\tau_n$（由 $\lambda_n$ 确定）选择 top-ranked token 传递至下一阶段：

$$\mathbf{x}_V^{(s_{n+1})} = \{x_V^i \in \mathbf{x}_V^{(s_n)} \mid S(i) \geq \tau_n\}$$

**视觉 Query 选择策略**：
- 在 LLM 中：提取视觉子空间内 image-to-image 的注意力图
- 在视觉编码器中：若有 [CLS] token（如 CLIP），用其对各 patch 的注意力作为重要性；若无（如 SigLIP），与 LLM 策略一致取均值

### 2. 阶段式上下文 Token 合并（Stage-wise Contextual Token Merging）

被剪枝的non-dominant token 中可能包含微妙但有用的视觉线索。为避免信息丢失，在每个阶段末尾执行轻量级合并：

- 复用注意力模块中的 key embedding 计算 token 间语义相似度（点积）
- 在 LLM 中，显式提取图像token部分的 key 向量确保模态纯净合并
- 将 non-dominant token 分为候选集和参考集，每个候选 token 与最相似的参考 token 配对融合
- 产生丰富的上下文 token 与 dominant token 一同传入下一阶段

**实现细节**：模型被划分为5个阶段，第一阶段在视觉编码器输出，后4个阶段分别在 LLM 的第 8、16、24 和最终解码层。第二阶段保留数为最终目标的 1.5 倍（图像理解）或 3 倍（视频理解）。

## 实验结果

### 表1：LLaVA-1.5-7B 不同保留率下的性能对比

| 方法 | Token数 | GQA | MMB | POPE | SQA | VQAv2 | Avg. |
|------|---------|------|------|------|------|-------|------|
| Full (上界) | 576 | 61.92 | 66.31 | 86.81 | 69.51 | 78.53 | 100% |
| FastV | 192 | 52.62 | 57.74 | 75.59 | 68.07 | 70.51 | 88.45% |
| PyramidDrop | 192 | 57.27 | 63.51 | 82.40 | 69.56 | 75.57 | 96.11% |
| SparseVLM | 192 | 59.44 | 65.41 | 86.45 | 68.86 | 77.01 | 98.64% |
| **VisionDrop** | **192** | **59.99** | **65.19** | **87.23** | **69.06** | **77.28** | **98.76%** |
| VisionZip | 32 | 51.80 | 58.02 | 75.11 | 68.72 | 67.12 | 89.92% |
| **VisionDrop** | **32** | **52.79** | **60.31** | **77.19** | **69.41** | **68.55** | **91.46%** |

### 表2：LLaVA-NeXT-7B 效率分析

| 方法 | Token数 | FLOPs (T) | 延迟 (ms) | 加速比 |
|------|---------|-----------|-----------|--------|
| LLaVA-1.5 原始 | 576 | 9.06 | 237 | 1.0× |
| VisionDrop | 64 | 2.11 | 117 | 2.0× |
| LLaVA-NeXT 原始 | 2880 | 46.25 | 593 | 1.0× |
| VisionDrop | 320 | 7.70 | 216 | 2.7× |

在 LLaVA-NeXT 上实现 6.0× FLOPs 减少，同时保持 95.71% 的原始性能。

## 关键发现

1. **文本引导评分在高压缩率下严重失效**：控制实验证明，保留 64 token 时视觉自注意力评分全面优于文本引导评分，差距随压缩率增大而扩大
2. **因果注意力导致位置偏置可视化明显**：浅层剪枝后保留的 token 始终聚集在图像底部（序列末尾），与语义无关
3. **渐进式剪枝优于单阶段**：跨编码器和 LLM 的多阶段策略比仅在某一端剪枝更稳定
4. **视频理解同样有效**：在 Video-LLaVA 上保留 12.5% token 仍取得最佳平均准确率 47.3%
5. **消融实验**：视觉编码器 33.3% 保留率为最优；dominant-to-contextual 比例变化对性能影响稳定

## 亮点

- **深刻的问题洞察**：系统性揭示因果/语义/空间三种跨模态失配，有理论分析和可视化支撑
- **简洁优雅的解决方案**：免训练、无需额外模块、仅复用已有注意力图，即插即用
- **统一管线设计**：首次将视觉编码器和 LLM 视为统一系统进行渐进式剪枝
- **广泛实验覆盖**：横跨 9 个图像基准 + 3 个视频基准，多种压缩比，与 5+ 个 SOTA 方法对比

## 局限性

- 仅在 LLaVA 系列模型上验证，未测试 Qwen-VL、InternVL 等更新架构的泛化性
- 阶段划分（第 8/16/24 层）为手动设定，缺乏自适应机制
- 上下文 token 合并的配对策略较简单（最近邻），可能不是最优融合方式
- 在 VizWiz（低质量图像）等特定任务上，有时会被 VisionZip 等编码器端方法超越
- 未探讨与模型蒸馏或量化等其他效率方法的联合使用

## 相关工作

- **LLM 内部剪枝**：FastV (ECCV 2024) 按生成时文本注意力排序；PyramidDrop (CVPR 2025) 用最后指令token做渐进剪枝；VScan 全局-局部扫描
- **视觉编码器剪枝**：VisionZip / VisPruner 基于注意力选择 dominant token + 相似性合并；FlowCut 利用跨层信息流；CDPruner 最大化指令条件多样性
- **跨模态引导**：SparseVLM 通过交叉注意力与文本引导计算 token 重要性
- **模态对齐研究**：Venhoff et al. (2025) 发现联合自注意力引入模态纠缠

## 评分

⭐⭐⭐⭐ — 问题发现精准且重要，三种失配的分析深入透彻；方法设计简洁有效，免训练即用；实验全面扎实。美中不足是模型覆盖有限、阶段划分缺乏自适应性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](../../ACL2025/multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)

</div>

<!-- RELATED:END -->
