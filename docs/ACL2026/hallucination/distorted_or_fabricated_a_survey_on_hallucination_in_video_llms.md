---
title: >-
  [论文解读] Distorted or Fabricated? A Survey on Hallucination in Video LLMs
description: >-
  [ACL 2026 Findings][幻觉检测][视频LLM幻觉] 本文首次对视频大语言模型（Vid-LLM）中的幻觉现象进行系统分类，提出"动态失真"（时空关系和引用一致性错误）和"内容捏造"（统计先验驱动和音视频冲突）的机制驱动分类体系，综述评估基准、缓解策略和根因分析。 领域现状：视频大语言模型在动作识别、时序推理等…
tags:
  - "ACL 2026 Findings"
  - "幻觉检测"
  - "视频LLM幻觉"
  - "动态失真"
  - "内容捏造"
  - "时空推理"
  - "多模态"
---

# Distorted or Fabricated? A Survey on Hallucination in Video LLMs

**会议**: ACL 2026 Findings  
**arXiv**: [2604.12944](https://arxiv.org/abs/2604.12944)  
**代码**: [GitHub](https://github.com/hukcc/Awesome-Video-Hallucination)  
**领域**: 幻觉检测  
**关键词**: 视频LLM幻觉, 动态失真, 内容捏造, 时空推理, 多模态

## 一句话总结
本文首次对视频大语言模型（Vid-LLM）中的幻觉现象进行系统分类，提出"动态失真"（时空关系和引用一致性错误）和"内容捏造"（统计先验驱动和音视频冲突）的机制驱动分类体系，综述评估基准、缓解策略和根因分析。

## 研究背景与动机

**领域现状**：视频大语言模型在动作识别、时序推理等任务上取得进展，但幻觉问题——生成看似合理但与视频内容矛盾的输出——仍然普遍。图像 VLM 的幻觉已被广泛研究，但视频的时序结构、运动动态和音视频整合使问题更加复杂。

**现有痛点**：现有多模态幻觉综述（Sahoo et al., Bai et al.）仅简略提及视频幻觉，缺乏结构性或因果分析。图像幻觉的分类（对象、属性、关系）无法直接迁移到视频——视频特有的时序错误（如事件排序错误、动作频率误计）和跨段引用不一致需要专门的分类框架。

**核心矛盾**：视频幻觉的根因与图像不同——动态失真源于有限的时序表示能力，内容捏造源于不充分的视觉 grounding——但现有缓解策略大多从图像幻觉迁移而来，未针对视频特性设计。

**本文目标**：建立首个视频幻觉的机制驱动分类体系，综合评述评估基准和缓解方法，分析根因并指出未来方向。

**切入角度**：基于"可视化证据是否存在"这一判据划分两大类——动态失真（有视觉证据但时空关系被错误建模）vs 内容捏造（无视觉证据，输出由先验驱动）。

**核心 idea**：视频幻觉的二分法——Distorted（扭曲了存在的内容）vs Fabricated（捏造了不存在的内容）。

## 方法详解

### 整体框架
分类体系分两层四类：
- **动态失真**：（1）时空动态错误（事件排序、持续时间、频率）；（2）引用不一致（角色混淆、场景混淆）
- **内容捏造**：（3）上下文驱动捏造（对象-动作共现先验、场景-事件先验）；（4）音视频冲突（音频主导动作推断、音频主导情感推断）

### 关键设计

**1. 机制驱动的分类体系：用"失败模式"而非"输入属性"做分类轴**

视频幻觉若按视频长度、领域这类输入属性来分，会把结构相同的失败硬拆开。本文改以"可观察的失败机制"为轴，建立可操作的诊断框架：一级判据是"输出是否有对应的视觉证据"，二级判据是"错误机制"。它给出一份决策清单（Figure 3）——输出是否有对应的视觉证据？有的话再查时空关系是否正确、时空正确但引用一致性是否出错；没有视觉证据则进一步判断是先验驱动还是音频驱动。因为同一种失败模式可以跨越不同输入设置出现，按失败机制分类才能把它们归到一起，让诊断真正落到可操作的判据上。

**2. 根因分析与未来方向映射：让缓解策略对准根因而非症状**

现有缓解策略大多从图像幻觉直接迁移，并未对准视频特有的根因。本文把每类幻觉与其根本原因对应起来：动态失真的根因是有限的时序编码（缺乏细粒度运动线索）外加长视频中的弱长程记忆和差时序定位；内容捏造的根因是不充分的视觉 grounding，使得预训练先验或主导音频信号盖过了视觉证据。由此推出对齐根因的方向——对动态失真应加强时序表示（如运动感知视觉编码器），对内容捏造应加强视觉 grounding（如反事实训练策略），避免"症状不对症下药"。

**3. 评估基准的系统化综述：按幻觉类型重组散落的 benchmark，暴露覆盖空白**

现有 benchmark 散落各处、口径不一，研究者很难快速找到匹配自己方向的评测。本文把 15+ 个 benchmark 按四种幻觉类型（时空动态、引用不一致、上下文捏造、音视频冲突）重新组织，并为每个 benchmark 标注视频长度、领域、评估格式、是否含专门基线和 SOTA 性能。这样组织之后，覆盖空白一目了然——例如音视频冲突这一类只有 3 个 benchmark，直接指向了一个被严重忽视的研究方向。

### 损失函数 / 训练策略
本文是综述论文，不涉及具体模型训练。

## 实验关键数据

### 主实验

| 幻觉类型 | 代表 Benchmark | SOTA 性能 | 说明 |
|---------|---------------|----------|------|
| 时空动态 | VidHalluc (CVPR'25) | GPT-4o: 81.2% | 动作顺序/持续时间 |
| 时空动态 | HAVEN | Valley-Eagle: 61.3% | 频率误计 |
| 引用不一致 | EGOILLUSION (EMNLP'25) | Gemini-Pro: 59.4% | 角色混淆 |
| 引用不一致 | ELV-Halluc | Gemini2.5-Flash: 53.1% | 长视频场景混淆 |
| 上下文捏造 | FactVC (EMNLP'23) | - | 对象-动作共现先验 |
| 音视频冲突 | - | - | benchmark 最少的类型 |

### 消融实验
本文为综述，无消融实验。

### 关键发现
- 时空动态错误在短视频中已很普遍，长视频中问题更严重（引用不一致和长程记忆失败）
- 内容捏造的根因是预训练阶段的统计先验过强——即使视觉输入不支持，模型仍会基于共现统计生成输出
- 音视频冲突是最被忽视的类型，benchmark 和缓解策略都极少
- SOTA 模型（如 GPT-4o）在最好的 benchmark 上也仅 ~80%，说明视频幻觉远未解决

## 亮点与洞察
- **Distorted vs Fabricated 的二分法**简洁有力——直接对应到"有证据但推理错误"和"无证据但先验补脑"两种根本不同的失败模式
- 综述结构清晰，从分类→评估→缓解→根因→未来方向的逻辑链条完整
- 指出音视频冲突是未来重要方向——随着多模态模型整合更多模态，跨模态冲突解决将越来越关键

## 局限与展望
- 综述聚焦于幻觉的"检测和分类"，对"为什么 Transformer 在时序编码上弱"的机制分析不够深入
- 缺乏对不同缓解策略的定量对比
- 分类体系的可操作性有待实际标注实验验证
- 音视频冲突部分文献较少，讨论深度有限

## 相关工作与启发
- **vs 图像 VLM 幻觉综述**: 图像幻觉关注对象/属性/关系错误，本文关注视频特有的时序和跨模态错误
- **vs MLLM 幻觉综述 (Sahoo et al.)**: 他们仅简略提及视频，本文做了深入的分类和根因分析
- **vs 特定 benchmark 论文**: 本文将散落的 benchmark 统一到一个分类框架中

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个视频幻觉专题综述，分类体系清晰
- 实验充分度: ⭐⭐⭐ 综述论文无实验，但 benchmark 覆盖全面
- 写作质量: ⭐⭐⭐⭐⭐ 结构层次分明，决策清单设计实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Hallucination Detection in LLMs with Topological Divergence on Attention Graphs](hallucination_detection_in_llms_with_topological_divergence_on_attention_graphs.md)
- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)
- [\[ACL 2026\] MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)
- [\[AAAI 2026\] Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs](../../AAAI2026/hallucination/does_less_hallucination_mean_less_creativity_an_empirical_investigation_in_llms.md)
- [\[NeurIPS 2025\] Robust Hallucination Detection in LLMs via Adaptive Token Selection](../../NeurIPS2025/hallucination/robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)

</div>

<!-- RELATED:END -->
