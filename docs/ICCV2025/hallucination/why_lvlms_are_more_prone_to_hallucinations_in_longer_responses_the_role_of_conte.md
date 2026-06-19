---
title: >-
  [论文解读] Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context
description: >-
  [ICCV 2025][幻觉检测][幻觉缓解] 深入探究 LVLM 长文本生成中幻觉频发的根本原因——不是长度本身，而是上下文的连贯性（coherence）和完备性（completeness）需求驱动模型外推产生幻觉，并据此提出 HalTrapper 的"诱导-检测-抑制"三阶段框架。 LVLM 在生成较长的自由形式回答时幻…
tags:
  - "ICCV 2025"
  - "幻觉检测"
  - "幻觉缓解"
  - "长文本生成"
  - "上下文分析"
  - "对比解码"
  - "大视觉语言模型"
---

# Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context

**会议**: ICCV 2025  
**arXiv**: [2510.20229](https://arxiv.org/abs/2510.20229)  
**代码**: [GitHub](https://github.com/SooLab/HalTrapper)  
**领域**: 幻觉检测  
**关键词**: 幻觉缓解, 长文本生成, 上下文分析, 对比解码, 大视觉语言模型

## 一句话总结

深入探究 LVLM 长文本生成中幻觉频发的根本原因——不是长度本身，而是上下文的连贯性（coherence）和完备性（completeness）需求驱动模型外推产生幻觉，并据此提出 HalTrapper 的"诱导-检测-抑制"三阶段框架。

## 研究背景与动机

LVLM 在生成较长的自由形式回答时幻觉明显增多，这一现象广为人知但缺乏深入理解：

- **表面现象**：幻觉对象的出现频率与其在输出序列中的位置正相关（越靠后越容易幻觉）
- **传统归因**：先前工作简单归因于自回归生成中长度带来的累积误差和不确定性增加
- **核心疑问**：幻觉增多究竟是长度导致的累积误差，还是存在更深层的机制？

作者通过创新的上下文操作实验（裁剪图像、丰富文本提示）发现，修改上下文后幻觉出现的位置会发生显著偏移（提前出现），这挑战了"长度=幻觉"的简单归因，揭示了**上下文是冰山下的深层因素**。

## 方法详解

### 整体框架

HalTrapper 是一个"诱导-检测-抑制"（induce-detect-suppress）三阶段框架，包含两条互补的路径：
- **内部锚定（Internal Grounding, IG）**：基于上下文连贯性的注意力相似度检测
- **外部扩展（External Expansion, EE）**：基于上下文完备性的跨提示一致性检测

### 关键设计

1. **上下文连贯性分析（Contextual Coherence）**：

    - **假设**：上下文连贯性在图像注意力上产生矛盾——模型既需关注已描述区域以保持一致性，又需转向新区域以避免重复。这种张力导致注意力分散和幻觉。
    - **验证**：计算同一回答中对象对的图像注意力相似度。定义幻觉对象集 $\mathcal{H}$ 和非幻觉对象集 $\mathcal{N}$ 的注意力相似度：
    $S_{\mathcal{H}} = \{\text{sim}(A_{s,i}, A_{s,j}) \mid o_{s,i}, o_{s,j} \in \mathcal{H}\}$
    $S_{\mathcal{N}} = \{\text{sim}(A_{s,i}, A_{s,j}) \mid o_{s,i}, o_{s,j} \in \mathcal{N}\}$
   结果表明：**幻觉对象之间的注意力相似度显著高于真实对象**，说明幻觉对象共享相似的分散注意力模式。

2. **上下文完备性分析（Contextual Completeness）**：

    - **假设（a）发生机制**：当回答已包含正确识别的对象但在信息或结构上仍不完整时，模型通过想象/编造细节来补偿——即幻觉。
    - **假设（b）固有倾向**：外推产生的幻觉依赖于多模态上下文，特别是视觉输入。
    - **验证**：向提示中逐步添加图像描述，观察幻觉的 PoScore（相对位置分数）。结果发现：**上下文越完整，幻觉越早出现**。同时，对同一图像使用5个不同提示，约 70% 的幻觉对象会重复出现，说明幻觉遵循固有模式。

3. **诱导阶段**：

    - **IG 诱导**：在模型完成回答后，将 EOS token 替换为 "There is also"，强制模型继续生成。由于完备性已被满足，新生成的对象更容易是幻觉，作为参考锚点 $o_s^{ref}$。
    - **EE 诱导**：使用 "Please imagine what object might be outside the frame" 等提示诱导模型想象图像外的内容，通过"先推理再想象"的提示设计过滤已存在的对象。

4. **检测阶段**：

    - **IG 检测**：计算诱导的幻觉参考对象与先前对象的注意力相似度得分：
    $\text{IGScore}_{s,i} = \text{sim}(A_s^{ref}, A_{s,i})$
    $S_{IG} = \{o_{s,i} \mid \text{IGScore}_{s,i} > \theta_{IG}\}$
    - **EE 检测**：通过多方向提示的一致性评分：
    $\text{EEScore}_{s,i} = \sum_{d \in \mathcal{D}} [\mathbb{1}(o_{s,i} \in S_{I,d}) - \mathbb{1}(o_{s,i} \in S_{R,d})]$
    - 合并检测结果：$S_{induction} = S_{IG} \cup S_{EE}$

5. **抑制阶段——对比上下文解码（CCD）**：
   将检测到的潜在幻觉对象编码为**对比上下文Token（CCT）** $x_{cct}$，作为对比分支的额外输入：
    $p_{ccd}(y_i|v,x,y_{<i}) = \text{softmax}[(1+\alpha)\text{logit}_\theta(y_i|v,x,y_{<i}) - \alpha \cdot \text{logit}_\theta(y_i|v,x,x_{cct},y_{<i})]$
   CCT 使对比分支自然地增加幻觉对象的概率，从而在原始分支中有效降低其生成概率。

### 损失函数 / 训练策略

HalTrapper 是**完全无训练**（training-free）的方法，不修改模型参数，仅在解码阶段进行干预。超参数 $\alpha=1.0$, $\beta=0.1$。

## 实验关键数据

### 主实验（CHAIR 指标，LLaVA v1.5 7B）

| 解码策略 | 方法 | CHAIR_S ↓ | CHAIR_I ↓ | Precision ↑ | F1 ↑ |
|---------|------|-----------|-----------|-------------|------|
| Greedy | Vanilla | 52.2 | 14.6 | 73.7 | 76.9 |
| Greedy | ICD | 51.4 | 14.7 | 73.4 | 77.0 |
| Greedy | CODE | 50.0 | 13.7 | 75.8 | 76.4 |
| Greedy | **HalTrapper** | **41.6** | **11.9** | **78.7** | **79.4** |
| Nucleus | Vanilla | 58.6 | 18.8 | 68.1 | 72.0 |
| Nucleus | VCD | 58.2 | 16.9 | 70.8 | 74.6 |
| Nucleus | **HalTrapper** | **48.6** | **14.5** | **74.6** | **76.1** |

### 消融实验（幻觉检测性能）

| 模型 | 检测方法 | AUROC | TPR@5%FPR | F1_max | Accuracy |
|------|---------|-------|-----------|--------|----------|
| LLaVA v1.5 | PoScore | 70.7 | 4.3 | 38.3 | 70.7 |
| LLaVA v1.5 | Top Logit | 64.0 | 13.0 | 32.2 | 61.9 |
| LLaVA v1.5 | Logits' Entropy | 67.7 | 16.6 | 36.6 | 71.4 |
| LLaVA v1.5 | Image Attn. Ratio | 44.9 | 6.0 | 27.3 | 32.0 |
| LLaVA v1.5 | **IG Score** | **82.3** | **43.3** | **54.8** | **86.3** |
| LLaVA v1.5 | EE Score | 77.5 | - | 46.1 | 72.9 |

### 关键发现

- IG Score 的 AUROC 达到 82.3，比最佳基线 PoScore 高 11.6 个百分点
- HalTrapper 在 Greedy 解码下将 CHAIR_S 从 52.2 降至 41.6（↓10.6），CHAIR_I 从 14.6 降至 11.9（↓2.7）
- 检测到的幻觉位置分布与真实幻觉分布高度一致，验证了上下文假说
- 在 AMBER 基准上，对 LLaVA v1.5、Qwen2 VL、Janus Pro 三个模型均有显著提升
- **关键洞察**：幻觉并非由长度直接导致，而是上下文驱动的——通过上下文操作可以控制幻觉出现的位置

## 亮点与洞察

- **深层因果分析**：不满足于现象层面的"长度→幻觉"关联，而是深入到"上下文→幻觉"的因果机制
- **优雅的实验设计**：通过裁剪图像和丰富提示两种互补的上下文操作，精确地揭示了上下文的作用
- **注意力相似度的发现**：幻觉对象共享高度相似的分散注意力模式，这一发现本身就有独立价值
- **幻觉的可预测性**：跨提示的幻觉重复率高达 70%，说明幻觉不是随机的而是有规律可循的
- **假设驱动的方法设计**：从假设出发→统计验证→方法设计→通过方法效果反向验证假设，形成完整闭环

## 局限与展望

- IG 方法依赖注意力图访问，对某些闭源模型或优化后的注意力实现不适用
- EE 方法在指令遵循能力较弱的模型上（如 MiniGPT-4）效果有限
- CCT 的构建依赖检测阶段的质量，假阳性可能抑制正确内容
- 仅关注对象级幻觉，属性幻觉和关系幻觉未深入探讨
- 增加了推理时的计算成本（需要诱导和检测过程）

## 相关工作与启发

- 与 VCD、ICD、OPERA 等对比解码方法不同，HalTrapper 使用检测到的幻觉对象（而非扰动输入）构建对比分支
- 对 LVLM 幻觉研究提供了新的分析视角：应关注上下文的作用而非仅关注长度
- 注意力相似度分析可能启发新的幻觉检测指标

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 从上下文连贯性和完备性角度解释幻觉是全新视角，假设验证闭环设计优秀
- **实验充分度**: ⭐⭐⭐⭐ 检测和抑制都有充分实验，但主要在 COCO 上评估，数据集多样性可加强
- **写作质量**: ⭐⭐⭐⭐⭐ 叙事结构清晰（现象→假设→验证→应用→再验证），图表精美且信息量大
- **价值**: ⭐⭐⭐⭐⭐ 对幻觉机理的洞察具有长远研究价值，不仅提供方法还提供理解

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](../../NeurIPS2025/hallucination/glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)
- [\[ACL 2025\] Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs](../../ACL2025/hallucination/stochastic_chameleons_irrelevant_context_hallucinations_reveal_class-based_misge.md)
- [\[ACL 2025\] Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence](../../ACL2025/hallucination/cracking_hallucination_vhd.md)
- [\[ICLR 2026\] LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals](../../ICLR2026/hallucination/lumina_detecting_hallucinations_in_rag_system_with_context-knowledge_signals.md)
- [\[AAAI 2026\] Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs](../../AAAI2026/hallucination/causally-grounded_dual-path_attention_intervention_for_objec.md)

</div>

<!-- RELATED:END -->
