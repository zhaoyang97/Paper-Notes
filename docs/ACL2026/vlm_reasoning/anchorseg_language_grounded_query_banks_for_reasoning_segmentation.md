---
title: >-
  [论文解读] AnchorSeg: Language Grounded Query Banks for Reasoning Segmentation
description: >-
  [ACL 2026][语义分割][推理分割] 提出AnchorSeg，将推理分割重构为基于语言引导查询库的结构化条件生成过程，通过锚点查询显式解耦空间定位与语义推理，配合Token-Mask循环一致性训练目标，在ReasonSeg上达到SOTA（67.7% gIoU, 68.1% cIoU）。 领域现状：推理分割要求模型根据…
tags:
  - "ACL 2026"
  - "语义分割"
  - "推理分割"
  - "语言引导查询库"
  - "空间先验"
  - "Token-Mask一致性"
  - "SAM"
---

# AnchorSeg: Language Grounded Query Banks for Reasoning Segmentation

**会议**: ACL 2026  
**arXiv**: [2604.18562](https://arxiv.org/abs/2604.18562)  
**代码**: [https://github.com/rui-qian/AnchorSeg](https://github.com/rui-qian/AnchorSeg)  
**领域**: 推理分割 / 多模态VLM  
**关键词**: 推理分割, 语言引导查询库, 空间先验, Token-Mask一致性, SAM

## 一句话总结

提出AnchorSeg，将推理分割重构为基于语言引导查询库的结构化条件生成过程，通过锚点查询显式解耦空间定位与语义推理，配合Token-Mask循环一致性训练目标，在ReasonSeg上达到SOTA（67.7% gIoU, 68.1% cIoU）。

## 研究背景与动机

**领域现状**：推理分割要求模型根据复杂、隐含的文本查询（如"这个场景中提供遮荫的物体"）预测像素级掩码。LISA等方法引入`<SEG>` token，将其隐藏状态作为单一查询送入SAM解码器来预测掩码。

**现有痛点**：现有方法将语义推理和空间定位都压缩到单一`<SEG>` token的隐藏表示中，这种隐式压缩限制了模型显式区分"分割什么"（语义推理）和"在哪分割"（空间定位）的能力，在复杂推理场景下表现受限。

**核心矛盾**：单一embedding需要同时编码语义理解和空间位置两种本质不同的信息，这造成了表征瓶颈——推理越复杂，单一向量越难以同时承载两种信号。

**本文目标**：将推理分割重新定义为结构化条件生成问题，在图像token层面显式建模空间定位，并用语言引导的查询来提供条件。

**切入角度**：引入多个可学习token构成"查询库"，让不同token承担不同角色——上下文查询负责语义推理，锚点查询负责空间定位。

**核心 idea**：用语言引导的查询库替代单一SEG token，通过因子化条件分布显式解耦空间定位（锚点查询）与语义调制（上下文查询）。

## 方法详解

### 整体框架

输入图像和文本查询，LMM（如LLaVA）自回归生成K个潜在推理token和1个分割锚点token `<SEG>`，构成查询库 $\mathbf{Q} = (\boldsymbol{q}_1, ..., \boldsymbol{q}_K, \boldsymbol{q}_{anc})$。锚点查询与图像token计算相似度产生空间先验，注入视觉特征后，整个查询库送入SAM解码器预测最终掩码。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["输入：图像 + 文本查询"] --> B
    subgraph B["语言引导查询库构建"]
        direction TB
        B1["LMM 自回归生成<br/>K 个潜在推理 token + 1 个 &lt;SEG&gt;"] --> B2["上下文查询（分割什么）<br/>+ 锚点查询（在哪分割）"]
    end
    B --> C["语言引导空间条件化<br/>锚点查询·图像 token 内积 → 空间先验 P<br/>逐元素注入视觉特征 f"]
    C --> D["SAM 解码器<br/>整个查询库条件化"]
    D --> E["预测掩码"]
    C -.->|训练约束| T["Token-Mask 循环一致性 TMCC<br/>token 级响应 ↔ 像素级掩码 双向对齐"]
    E -.->|训练约束| T
```

### 关键设计

**1. 语言引导查询库构建：把挤在一个 token 里的"推什么"和"在哪"拆到不同 token 上**

旧范式把语义推理和空间定位全压进单个 `<SEG>` token，推理越复杂这个向量越扛不住。AnchorSeg 扩展 LMM 的词表，引入 $K$ 个潜在推理 token `<LAT_1>,...,<LAT_K>` 和一个分割 token `<SEG>`，自回归生成时 `<SEG>` 显式条件化在前面的推理 token 之上：上下文查询 $\boldsymbol{q}_{1:K}$ 负责编码中间推理状态、对应"分割什么"，锚点查询 $\boldsymbol{q}_{anc}$ 专门承载"在哪分割"的空间信号。这样模型内部自然形成"先推理、后定位"的有序分工，不再让单个 embedding 同时背两份本质不同的信息。

**2. 语言引导空间条件化：让锚点查询直接在图像 token 上算出一张空间先验图**

光是拆出锚点查询还不够，得把它变成解码器能用的显式定位信号。AnchorSeg 把空间定位建模成图像 token 上的因子化条件分布 $p(\boldsymbol{S}|\mathbf{Q}) = \prod_i p(s_i | \boldsymbol{i}_i, \boldsymbol{q}_{1:K}, \boldsymbol{q}_{anc})$，落地时就是锚点查询与每个图像 token 做内积算空间响应 $s_i = \boldsymbol{i}_i^\top \boldsymbol{q}_{anc}$，reshape 成空间先验图 $\mathbf{P}$，再逐元素加回视觉特征 $\tilde{\mathbf{f}} = \mathbf{f} \oplus \mathbf{P}$ 后送进 SAM 解码器。锚点查询直接产出定位响应，而上下文查询通过自回归生成链路隐式塑造锚点查询的内容，于是语义对空间的调制就显式发生在特征层面，而非埋在一个不可分解的向量里。

**3. Token-Mask 循环一致性（TMCC）：用双向约束补上 token 级响应与像素级掩码的分辨率落差**

空间响应在低分辨率的 token 网格上算，监督却是高分辨率的像素掩码，两个层次各算各的容易打架。TMCC 加一对双向约束把它们锁住：Token-to-Mask 把 token 级响应上采样到图像分辨率，用 BCE+Dice 损失对齐高斯平滑后的 GT 掩码；Mask-to-Token 反过来把 GT 掩码下采样到 token 分辨率，与 token 级响应对齐。一上一下互相校准，保证语义-视觉两个层级上的空间推理保持一致、不至于训练发散。

### 损失函数 / 训练策略

总损失包含三部分：自回归文本生成损失 $\mathcal{L}_{txt}$、SAM掩码预测损失 $\mathcal{L}_{mask}$（BCE+Dice）、以及TMCC损失 $\mathcal{L}_{T2M} + \mathcal{L}_{M2T}$。TMCC的BCE和Dice权重与掩码损失共享。

总损失包含三部分：自回归文本生成损失 $\mathcal{L}_{txt}$、SAM掩码预测损失 $\mathcal{L}_{mask}$（BCE+Dice）、以及TMCC损失 $\mathcal{L}_{T2M} + \mathcal{L}_{M2T}$。TMCC的BCE和Dice权重与掩码损失共享。

## 实验关键数据

### 主实验

在ReasonSeg测试集上的表现：

| 方法 | gIoU | cIoU |
|------|------|------|
| LISA-7B | 54.3 | 58.1 |
| GSVA-7B | 55.6 | 59.4 |
| READ-7B | 57.2 | 60.5 |
| RSVP-7B | 63.7 | 64.8 |
| **AnchorSeg-7B** | **67.7** | **68.1** |

### 消融实验

| 配置 | gIoU | 说明 |
|------|------|------|
| 单一SEG token (baseline) | 54.3 | LISA原始设计 |
| + 查询库 (无空间先验) | ~62 | 多token推理有帮助 |
| + 空间先验注入 | ~65 | 显式定位信号提升大 |
| + TMCC | 67.7 | 双向一致性进一步提升 |

### 关键发现

- 从单一SEG token到查询库的提升最为显著，说明多token推理结构是核心贡献
- 空间先验的显式注入（而非仅作为查询）带来明显额外收益，验证了解耦设计的必要性
- TMCC的双向一致性约束虽然提升幅度不大，但有效防止了训练不稳定
- 在RefCOCO/+/g上也展现出竞争力，表明方法泛化性好

## 亮点与洞察

- 因子化条件分布的建模方式非常优雅：将空间定位显式建模为"每个图像token的相关性"，数学表达清晰且物理意义明确。这种token级的空间推理可以迁移到其他需要精确定位的多模态任务。
- 查询库内部的角色分工（上下文查询 vs 锚点查询）类似于人类的认知过程：先理解问题语义，再进行空间定位，最后精细分割。
- TMCC的跨分辨率一致性约束是一个简洁但有效的正则化手段，可应用于任何涉及不同分辨率表征对齐的场景。

## 局限与展望

- 查询库中的K值（潜在推理token数量）是超参数，不同复杂度的查询可能需要不同数量的推理token
- 空间先验仅通过简单内积计算，可能在需要复杂空间推理（如遮挡关系）时不够强大
- 目前仅在推理分割和referring segmentation上评估，未探索在视觉问答等其他任务中的泛化
- 方法依赖SAM作为掩码解码器，受SAM本身能力的限制

## 相关工作与启发

- **vs LISA**: 使用单一SEG token，语义和空间信息压缩在一起；AnchorSeg通过查询库显式解耦，gIoU提升13.4个点
- **vs GSVA**: 扩展到多目标推理和非存在物体拒绝，但仍基于单token范式；AnchorSeg从根本上改变了表征结构
- **vs RSVP**: 引入多模态CoT推理，但推理过程与分割模块耦合；AnchorSeg的因子化设计更加模块化和可解释

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 查询库+因子化空间条件化的设计思路非常新颖
- 实验充分度: ⭐⭐⭐⭐ 在ReasonSeg和RefCOCO上全面评估
- 写作质量: ⭐⭐⭐⭐ 形式化清晰，但部分符号较重
- 价值: ⭐⭐⭐⭐ 为推理分割提供了更结构化的解决范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](../../ICLR2026/segmentation/regionreasoner_region-grounded_multi-round_visual_reasoning.md)
- [\[ICCV 2025\] VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation](../../ICCV2025/segmentation/veggie_instructional_editing_and_reasoning_video_concepts_with_grounded_generati.md)
- [\[NeurIPS 2025\] LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation](../../NeurIPS2025/segmentation/langhops_language_grounded_hierarchical_open-vocabulary_part_segmentation.md)
- [\[ECCV 2024\] VISA: Reasoning Video Object Segmentation via Large Language Models](../../ECCV2024/segmentation/visa_reasoning_video_object_segmentation_via_large_language_models.md)
- [\[CVPR 2026\] PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation](../../CVPR2026/segmentation/pixdlm_uav_reasoning_segmentation.md)

</div>

<!-- RELATED:END -->
