---
title: >-
  [论文解读] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices
description: >-
  [多模态VLM] 本文系统研究了多模态 LLM 中多层视觉特征融合的两个核心问题：(1) 如何选择最有效的视觉层：和 (2) 如何最好地融合到语言模型中：，发现从不同表示相似性阶段各选一层 + 外部直接融合是最优实践。 领域现状 多模态大语言模型（MLLM）通过结合预训练视觉编码器和 LLM 取得了显著进展。然而…
tags:
  - "多模态VLM"
---

# Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices

| 信息 | 内容 |
|------|------|
| 会议 | CVPR 2025 |
| arXiv | [2503.06063](https://arxiv.org/abs/2503.06063) |
| 代码 | [EIT-NLP/Layer_Select_Fuse_for_MLLM](https://github.com/EIT-NLP/Layer_Select_Fuse_for_MLLM) |
| 领域 | 多模态大语言模型 / 视觉特征融合 |
| 关键词 | Multi-layer Visual Features, Fusion Strategy, Layer Selection, MLLM, LLaVA |

## 一句话总结

本文系统研究了多模态 LLM 中多层视觉特征融合的两个核心问题：**(1) 如何选择最有效的视觉层**和 **(2) 如何最好地融合到语言模型中**，发现从不同表示相似性阶段各选一层 + 外部直接融合是最优实践。

---

## 研究背景与动机

### 领域现状

多模态大语言模型（MLLM）通过结合预训练视觉编码器和 LLM 取得了显著进展。然而，大多数模型（如 LLaVA、InternVL）仅使用视觉编码器的单层输出（通常是倒数第二层），浪费了其他层的信息。一些工作（Dense Connector、EVLM）尝试使用多层特征，但选择方式和融合策略缺乏系统性。

### 现有痛点

1. **层选择任意**：Dense Connector 按比例选层，EVLM 直接用后半部分层的特征——缺乏理论指导
2. **融合策略混乱**：有的在 LLM 输入前融合，有的在 LLM 中间层融合；有的用额外模块（交叉注意力），有的直接拼接——各方法互不可比
3. **实验不公平**：许多方法使用了更大的数据集或更复杂的架构，难以判断改进来自融合策略还是模型容量

### 核心矛盾

多层视觉特征**确实有用**，但**缺乏系统性研究**告诉我们"选哪些层"和"怎么融合"。

### 本文切入角度

控制其他变量（模型大小、训练数据），仅改变层选择和融合策略，在统一框架下进行全面对比实验。

---

## 方法详解

### 整体框架

基于 Mini-LLaVA（LLaVA-1.5 架构 + MobileLLaMA 1.4B 替代 Vicuna 7B），系统实验两个维度：层选择策略 × 融合策略。

### 关键设计1：视觉层选择策略

- **相似度分层选择（Similarity-based）**：基于不同层视觉特征的余弦相似度，将 24 层编码器划分为三个阶段：
    - Beginning 阶段（低层）：代表层为第 3 层——捕捉低级细节特征
    - Middle 阶段（中层）：代表层为第 18 层——编码中层语义
    - Ending 阶段（高层）：代表层为第 23 层——包含高级判别特征
    - 组合：Single {18}、Double {3, 18}、Triple {3, 18, 23}
- **比例分层选择（Proportion-based）**：按编码器深度等分为前半和后半：
    - Former {1-12}、Latter {13-24}、All {1-24}
- **核心发现**：从**不同阶段各选一个代表层**效果最好；从**同一阶段选多层反而性能下降**

### 关键设计2：四种融合策略分类

按两个维度分类：

| | 模块融合 (Modular) | 直接融合 (Direct) |
|---|---|---|
| **内部融合 (Internal)** | 交叉注意力模块插入 LLM 中间层 | 直接在 LLM 中间层加入视觉 token |
| **外部融合 (External)** | 额外模块处理后与文本 token 一起输入 | 直接拼接/相加后与文本 token 一起输入 |

- **内部模块融合**：在 LLM 对应层通过 pre-/post-/parallel cross-attention 融合视觉特征
- **内部直接融合**：在 LLM 对应层直接添加视觉 token
- **外部模块融合**：先用额外模块处理多层视觉特征再输入 LLM
- **外部直接融合**：多层特征通过逐元素加法/维度拼接后直接与文本 token 拼接输入 LLM

### 关键设计3：Mini-LLaVA 轻量实验平台

- **功能**：降低探索性实验的计算成本，使大量消融实验成为可能
- **配置**：CLIP-ViT-L/14（24 层）+ MobileLLaMA 1.4B（24 层）
- **训练**：预训练阶段 558K 图像字幕 + 指令微调阶段 665K 对话
- **设计动机**：视觉编码器和 LLM 都是 24 层，可以一一对应进行内部融合实验

---

## 实验关键数据

### 外部融合 vs 内部融合（Triple {3, 18, 23}）

| 融合策略 | GQA | MMB | TextVQA | POPE | 平均 |
|----------|-----|-----|---------|------|------|
| Mini-LLaVA (基线) | 56.95 | 46.91 | 35.47 | 85.83 | 48.51 |
| 内部模块融合 (Pre-Cross) | 57.56 | 49.66 | 34.06 | 84.69 | 46.91 |
| 内部直接融合 | 58.59 | 47.47 | 36.24 | 85.87 | 48.54 |
| **外部直接融合** | **59.12** | **51.20** | **36.87** | **86.10** | **49.85** |

### 层选择对比（外部直接融合）

| 层集合 | 平均性能 |
|--------|----------|
| Single {18} | 49.20 |
| Double {3, 18} | 49.52 |
| **Triple {3, 18, 23}** | **49.85** |
| Former {1-12} | 下降 |
| Latter {13-24} | 下降 |
| All {1-24} | 不稳定/下降 |

### 在 LLaVA-1.5 (7B) 上验证

| 方法 | GQA | MMB | TextVQA | 平均 |
|------|-----|-----|---------|------|
| LLaVA-1.5 基线 | 62.0 | 64.3 | 58.2 | — |
| + 外部直接融合 (Triple) | **63.1** | **65.7** | **59.4** | — |

提升一致，验证了结论的可迁移性。

### 核心实验发现总结

1. **跨阶段选层优于同阶段多选**：从 beginning/middle/ending 各选一层（Triple {3, 18, 23}）最佳
2. **外部直接融合一致最优**：简单、稳定、参数效率高
3. **内部模块融合存在训练困难**：层数增多时 loss 难以收敛，All 配置甚至无法完成训练
4. **三种交叉注意力变体差异不大**：pre-cross、post-cross、parallel 表现接近
5. **内部直接融合有潜力**：在大数据集训练时可能赶上外部融合

---

## 亮点与洞察

1. **系统性强**：2 种层选择标准 × 4 种融合策略 = 全面的实验矩阵，是该领域最系统的研究
2. **结论清晰可操作**：从不同相似度阶段各取一层 + 外部直接融合 = 简单有效的最佳实践
3. **反直觉发现**：更多层不一定更好——同阶段多层特征冗余反而有害
4. **控制变量公平**：所有实验使用相同的模型和数据，排除了容量差异的干扰

## 局限性

1. 代表层的选择（3/18/23）基于经验训练筛选，缺乏自动化的层选择方法
2. Mini-LLaVA (1.4B) 上的结论在更大模型（70B+）上是否成立存疑
3. 仅在 CLIP-ViT-L/14 一种视觉编码器上实验，其他编码器（SigLIP、InternViT）的层间特性可能不同
4. 外部直接融合的"最优融合运算"（加法 vs 拼接）未充分探讨

## 相关工作与启发

- **与 Dense Connector 的对比**：Dense Connector 按比例选层+用额外连接器融合，本文证明了更简单的策略更优
- **与 EVLM 的对比**：EVLM 使用后半段层特征，本文发现应从三个不同阶段选层
- **与 DeepStack 的对比**：DeepStack 做内部融合（高分辨率子图注入 LLM 中间层），本文发现外部融合更稳定
- **启发**：视觉编码器不同层的特征具有"阶段性"——同阶段内冗余、跨阶段互补。这一发现可指导所有需要多层特征的视觉任务设计

---

## 评分

⭐⭐⭐⭐ — 系统性强，实验扎实，结论实用且可直接指导工程实践；但层选择机制仍偏经验化，缺乏自动化方案。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](../../ICCV2025/multimodal_vlm/hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](../../ACL2025/multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](../../ACL2025/multimodal_vlm/teaching_vlm_ask_ambiguity.md)
- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](../../ICCV2025/multimodal_vlm/mm-ifengine_towards_multimodal_instruction_following.md)

</div>

<!-- RELATED:END -->
