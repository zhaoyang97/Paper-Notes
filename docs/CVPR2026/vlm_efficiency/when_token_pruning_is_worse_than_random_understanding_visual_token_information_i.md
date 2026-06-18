---
title: >-
  [论文解读] When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs
description: >-
  [CVPR 2026][多模态VLM][token剪枝] 发现VLLM深层中现有token剪枝方法不如随机剪枝的现象，提出基于输出概率变化量化视觉token信息的方法，揭示了"信息地平线"——视觉token信息在某层均匀消散至零的临界层，其位置受任务视觉复杂度和模型能力动态影响，并证明简单集成随机剪枝能有效提升现有方法。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "token剪枝"
  - "信息地平线"
  - "视觉token信息量"
  - "随机剪枝"
  - "VLM推理加速"
---

# When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs

**会议**: CVPR 2026  
**arXiv**: [2512.07580](https://arxiv.org/abs/2512.07580)  
**代码**: [https://github.com/YahongWang1/Information-Horizon](https://github.com/YahongWang1/Information-Horizon)  
**领域**: 多模态VLM  
**关键词**: token剪枝, 信息地平线, 视觉token信息量, 随机剪枝, VLM推理加速

## 一句话总结
发现VLLM深层中现有token剪枝方法不如随机剪枝的现象，提出基于输出概率变化量化视觉token信息的方法，揭示了"信息地平线"——视觉token信息在某层均匀消散至零的临界层，其位置受任务视觉复杂度和模型能力动态影响，并证明简单集成随机剪枝能有效提升现有方法。

## 研究背景与动机
**领域现状**：VLLM将图像编码为大量视觉token（LLaVA-1.5为576个，Qwen2.5-VL可达数千个），免训练token剪枝是主流加速方案，分为重要性方法（FastV/SparseVLM）和多样性方法（DivPrune/DART）。

**现有痛点**：关键观察——在深层decoder层（如第20层之后），所有现有剪枝方法的表现不优于甚至劣于随机剪枝。LLaVA-1.5-7B上第16-20层、Qwen2.5-VL-7B上第21层后，这一现象一致出现。

**核心矛盾**：现有剪枝方法无论用注意力还是相似度作为选择准则，在深层都无法识别出比随机选择更有信息量的token——这意味着视觉token的信息在深层已经"消散"。

**本文目标** (a) 为什么深层剪枝不如随机？(b) 视觉token信息如何随层变化？(c) 在哪层可以安全移除所有视觉token？(d) 如何利用这些发现改进现有方法？

**切入角度**：定义并量化单个视觉token在特定层的信息量，追踪其跨层变化模式。

**核心 idea**：视觉token信息在深层均匀消散到零（"信息地平线"），超过此层剪枝等效于随机，结合随机剪枝可改进现有方法。

## 方法详解

### 整体框架
这篇论文不提新剪枝方法，而是想搞清楚一件反直觉的事：为什么到了 decoder 深层，所有现有剪枝方法反而打不过随机剪枝。它的整条线索是先给「单个视觉 token 在某一层到底有多少信息」下一个可测量的定义，再用这个定义去追踪信息怎么随层变化，结果发现信息会在某一层集体消散到零（"信息地平线"），并进一步刻画这条地平线深浅由什么决定，最后把结论翻译成一个零成本的改进：在地平线之后干脆改用随机剪枝。所以四个设计点是一条因果链——量化工具 → 现象 → 现象的成因 → 现象的用法。

### 关键设计

**1. 视觉 Token 信息定义：用"删掉它输出概率掉多少"来量化单个 token 的贡献**

要回答"深层剪枝为什么失效"，先得能说清每个视觉 token 在每一层值多少。本文用一种因果式的留一测量：在第 $i$ 层，先把除目标 token $\mathbf{V}_k$ 以外的所有视觉 token 都移除、做一次前向得到 Ground Truth 标签的输出概率 $p_k$，再把所有视觉 token 都移除得到纯文本基线概率 $p_{text}$，两者之差就是这个 token 单独携带的信息量：

$$I_i(\mathbf{V}_k) = p_k - p_{text}$$

先隔离出 target token、把其余视觉 token 排除掉，是为了避免其他 token 的信息混入、让度量真正归到 $\mathbf{V}_k$ 头上。这个定义不只是分析工具——按它把低信息 token 删掉后模型性能反而一致上升（LLaVA-1.5-7B 上 MME 涨 27.8%），说明这些低信息 token 不是无害噪声，而是实打实的干扰源，也反过来印证了度量抓到的确实是"有没有用"。

**2. 信息地平线：视觉 token 信息均匀消散至零的那一临界层**

有了逐层的信息量，就能看它怎么演化。本文发现一个稳定的模式：浅层时不同 token 的信息量差异很大（高方差，有的 token 很关键、有的没用），越往深层这种差异越被抹平，到某一层所有 token 的信息量一起趋近于零——这一层就是"信息地平线"。越过它再移除全部视觉 token 也几乎不影响输出。LLaVA-1.5-7B 上 MME 的地平线约在第 16 层、TextVQA 约在第 24 层。这正好解释了开篇那个反直觉现象：深层之所以剪枝不如随机，是因为此时每个 token 的信息都接近零、彼此没差别，注意力也好相似度也好都失去了可依据的信号，任何"精挑细选"都退化成瞎选，自然不比随机更好。

**3. 信息地平线的动态性：它的深浅由任务视觉复杂度和模型视觉能力共同决定**

地平线不是一个固定层号，而是随场景浮动的。一方面看任务的视觉复杂度：知识型 QA、幻觉检测这类靠文本知识就能答的任务地平线偏浅（Qwen2.5-VL 上约第 20 层就消散），而 OCR 这类必须持续读图的视觉密集任务地平线更深（约第 27 层）。另一方面看模型自身的视觉能力：更强的 Qwen2.5-VL 比较弱的 LLaVA-1.5 地平线更深，意味着它能把视觉信息一直用到更深的层。两条规律合起来给出一个直观图景——越需要看图、模型越会看图，视觉 token 的"有效寿命"就越长。

**4. 随机剪枝集成策略：地平线之后放弃精挑细选，直接随机剪**

既然深层信息已经均匀、任何准则都没有优势，那在地平线之后就没必要再花力气算重要性或相似度，直接随机剪枝即可，且可以零成本叠加到任何现有方法上。这套 +Random 在 Qwen2.5-VL-7B 上剪掉 50% token 仍保住原模型 96.9% 的性能（DivPrune+Random）。它的价值不在精巧而在务实：把"信息消散"这个诊断结论直接变成一行可落地的修补。

## 实验关键数据

### 主实验——Qwen2.5-VL-7B 剪除50%

| 方法 | MME | TextVQA | MMB | OCRBench | Avg. | Rel.(%) |
|------|-----|---------|------|---------|------|---------|
| 原模型 | 2313 | 85.4 | 79.8 | 88.5 | 83.6 | 100.0 |
| DART | 2295 | 82.1 | 79.6 | 75.5 | 77.3 | 92.7 |
| DivPrune | 2291 | 83.1 | 79.4 | 84.1 | 80.7 | 96.7 |
| DART+Random | 2318 | 82.7 | 79.6 | 77.9 | 78.3 | 93.9 |
| **DivPrune+Random** | **2302** | **83.4** | **79.5** | **85.3** | **80.9** | **96.9** |

### 消融实验——信息量化有效性（LLaVA-1.5-7B）

| 操作 | MME变化 | TextVQA变化 |
|------|---------|-----------|
| 移除75%低信息token@第10层 | **+27.8%** | **+6.1%** |
| 移除88%低信息token@第10层 | 仍优于原模型 | 仍优于原模型 |

### 关键发现
- **信息消散是普遍的**：在LLaVA-1.5和Qwen2.5-VL上都观察到，与模型架构无关
- 浅层（1-7层）剪枝方法能有效保留高信息token，多样性方法优于重要性方法
- 深层（14层后）所有方法退化到随机水平，因为信息方差趋零
- 移除低信息token竟然提升性能，说明低信息token是干扰源而非无害噪声
- +Random的改进在OCRBench上最明显（DART: 75.5→77.9），因为OCR任务地平线更深，深层仍有信息可利用

## 亮点与洞察
- **"剪枝不如随机"的反直觉发现**：通过严格实验和信息量化机制性地解释了这一现象，将观察提升为可操作的理论
- **信息地平线概念**：提供了一个简洁而实用的框架来理解视觉token在VLLM中的生命周期——信息产生→传播→消散
- **简单有效的改进策略**：+Random几乎零成本即可提升现有方法，实用价值高
- **任务-模型-地平线三角关系**：揭示了视觉复杂度和模型能力共同决定视觉token有用深度的动态机制

## 局限与展望
- 信息定义需要Ground Truth标签，无法在推理时直接使用
- 信息度量需要额外的前向传播（逐token移除），计算开销大
- 信息地平线的精确位置需要对每个任务/模型/样本单独测量，缺乏预测模型
- 仅测试了LLaVA-1.5和Qwen2.5-VL两个模型
- +Random策略虽然有效但缺乏理论保证，本质上是在信息消散后"放弃精细选择"

## 相关工作与启发
- **vs FastV/SparseVLM/DART**：本文不是提出新的剪枝方法，而是从信息角度解释为什么现有方法在深层失败，并提供简单修补方案
- **vs EmbedLens论文**：互补关系——EmbedLens从表示结构角度发现sink/dead/alive分类，本文从信息量化角度发现深层信息消散；两者共同支持"视觉token对MLLM的真正贡献集中在中浅层"的结论
- **实际应用**：+Random策略可直接叠加到任何现有method上

## 评分
- 新颖性: ⭐⭐⭐⭐ 信息地平线概念新颖，信息量化方法直接有效
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型×多基准×多种剪枝方法的全面分析
- 写作质量: ⭐⭐⭐⭐ 观察→假设→验证→应用的逻辑链清晰
- 价值: ⭐⭐⭐⭐ 为token剪枝研究提供了重要的理论理解

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models](hawk_head_importance-aware_visual_token_pruning_in_multimodal_models.md)
- [\[CVPR 2026\] TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model](transprune_token_transition_pruning_for_efficient_large_vision-language_model.md)
- [\[ICLR 2026\] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](../../ICLR2026/multimodal_vlm/index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)
- [\[CVPR 2026\] DocPrune: Efficient Document Question Answering via Background, Question, and Comprehension-aware Token Pruning](docpruneefficient_document_question_answering_via_background_question_and_compre.md)
- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)

</div>

<!-- RELATED:END -->
