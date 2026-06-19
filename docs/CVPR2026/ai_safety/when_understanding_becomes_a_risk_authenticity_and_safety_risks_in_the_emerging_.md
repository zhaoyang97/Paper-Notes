---
title: >-
  [论文解读] When Understanding Becomes a Risk: Authenticity and Safety Risks in the Emerging Image Generation Paradigm
description: >-
  [CVPR 2026][AI安全][多模态大语言模型] 系统性对比分析了 MLLM（多模态大语言模型）与扩散模型在安全风险上的差异，发现 MLLM 因更强的语义理解能力而更容易生成不安全图像（抽象/非英语提示也能理解），且其生成的图像更难被现有假图检测器识别，即便针对性微调检测器也可通过丰富提示细节来规避。
tags:
  - "CVPR 2026"
  - "AI安全"
  - "多模态大语言模型"
  - "图像安全"
  - "不安全内容生成"
  - "虚假图像检测"
  - "扩散模型对比"
---

# When Understanding Becomes a Risk: Authenticity and Safety Risks in the Emerging Image Generation Paradigm

**会议**: CVPR 2026  
**arXiv**: [2603.24079](https://arxiv.org/abs/2603.24079)  
**代码**: 无  
**领域**: AI安全 / 图像生成  
**关键词**: 多模态大语言模型, 图像安全, 不安全内容生成, 虚假图像检测, 扩散模型对比

## 一句话总结

系统性对比分析了 MLLM（多模态大语言模型）与扩散模型在安全风险上的差异，发现 MLLM 因更强的语义理解能力而更容易生成不安全图像（抽象/非英语提示也能理解），且其生成的图像更难被现有假图检测器识别，即便针对性微调检测器也可通过丰富提示细节来规避。

## 研究背景与动机

**领域现状**：文本到图像生成范式正从扩散模型（如 Stable Diffusion）向多模态大语言模型（MLLM，如 Janus、Bagel）转变。扩散模型通过迭代去噪生成图像，而 MLLM 将语言理解、跨模态推理和图像生成集成在统一架构中，具有更强的语义理解和组合灵活性。

**现有痛点**：围绕扩散模型已建立了相当成熟的安全防线——安全过滤器（如 NSFW 检测器）和假图检测器（如 DE-FAKE）。但这些防护手段都是针对扩散模型优化的。MLLM 具有本质不同的生成机制：它能理解微妙、抽象甚至非直接表达的不安全意图，这意味着现有防线可能系统性失效。

**核心矛盾**：MLLM 更强的语义理解能力是一把双刃剑——在提升可用性和生成质量的同时，也使得它能理解并执行那些扩散模型根本"看不懂"的不安全指令。而研究社区尚未系统认识到这一范式转移带来的新安全风险。

**本文目标** 两个核心研究问题：（1）MLLM 是否比扩散模型更容易生成不安全图像？（2）MLLM 生成的假图是否比扩散模型更难检测？

**切入角度**：以扩散模型作为参照基准，使用 1,184 条不安全提示生成 82,880 张图像，以及 2,000 条良性提示生成 14,000 张图像，对 7 个模型（2 个扩散 + 5 个 MLLM）进行系统对比实验。

**核心 idea**：MLLM 的语义理解优势在安全维度成为劣势——它能理解抽象不安全提示并生成有意义的不安全内容，而扩散模型因理解失败反而"安全"地产出损坏图像。

## 方法详解

### 整体框架

这篇论文不提新方法，而是一项系统性的测量研究（measurement study）：把"文生图"从扩散模型迁移到 MLLM 之后，原有的安全防线还守不守得住。整套实验围绕两个问题展开——MLLM 是不是更容易生成不安全图像（RQ1），以及它生成的图像是不是更难被假图检测器抓到（RQ2）。为了让对比公平，作者固定一批模型（2 个扩散 + 5 个 MLLM）、固定数据集，再用统一的分类器把"不安全 / 安全 / 损坏"量化出来，所有结论都建立在 8 万多张实际生成图像上，而非个案演示。

### 关键设计

**1. 不安全图像生成评估框架：量出模型本身的"作恶倾向"，而非它的过滤器**

要回答 RQ1，关键是把"模型愿不愿意生成"和"模型外面那层过滤器拦不拦得住"分开。作者的做法是**禁用全部外部安全机制**，让 7 个模型（SD3.5 Large / Turbo、Bagel、Janus、Janus Pro、TokenFlow、VILA-U）在 5 个不安全提示数据集（I2P、Lexica、4chan、Template、TemplateLong）上裸跑，每条提示生成 10 张图。再用 OpenAI Moderation API 当裁判（与人工标注一致率 89.2%），算出每个"提示-模型"对的**不安全得分**——即这 10 张里被判为 NSFW 的比例。这样得到的分数衡量的是模型内在的语义服从度，而不是它那层可以随时被绕过的 NSFW 滤镜。

**2. 损坏图像因素分析：拆穿"扩散模型更安全"这个假象**

裸测下来扩散模型的不安全得分确实更低，但作者怀疑这不是因为它对齐得好，而是因为它根本看不懂复杂提示、直接生成了一堆乱码图。为了验证，作者对 50 条 4chan 提示各造一个"安全对照版"（把不安全词替换掉），让每个模型各生成 10 张，统计**损坏图像率**。结果很说明问题：SD3.5 Large 的损坏率高达 80%，而最差的 MLLM（Bagel）也只有 4.5%；并且这些损坏图里 99.4% 被分类器判成了"安全"。换句话说，扩散模型的低分是被它自己的"生成失败"喂出来的——理解力不足反而成了一道意外的安全阀。

**3. 假图检测评估框架：看现有检测器能不能识别 MLLM 的"指纹"，以及提示越长是不是越难抓**

RQ2 关心的是已部署的假图检测器在 MLLM 时代还灵不灵。作者用 MSCOCO、Flickr30k 的良性提示生成图像，丢给 4 个检测器（商业的 Winston.AI、Illuminarty + 开源的 DE-FAKE、AIorNot-SigLIP2）打分。更关键的是一个**提示丰富度梯度实验**：同一意图写成 v0（原始提示）→ v1（GPT-4o 扩写）→ v2（再扩写）三档，看提示越详细、图像是否越难被判为假。这个设计直接探针式地检验了 MLLM 是否提供了一条扩散模型没有的规避路径。

**4. 跨语言 / 偏见 / 联想三组对照：把"理解力即风险"具体化**

前三个框架给出宏观分数，这一组对照则把 MLLM "懂得更多"的副作用拆给读者看。**跨语言**：换成中文版不安全提示（TemplateLongChinese），扩散模型彻底失灵（不安全得分为 0），MLLM 照样生成不安全图像。**性别偏见**：在性别中立的色情提示下，MLLM 明显偏向生成女性图像。**联想**：给一条抽象俚语提示（如 "f\*ck that the place is a sh\*t hole"），扩散模型只会把文字字面渲染上去，MLLM 却能联想出对应的肮脏环境。三组对照共同指向同一结论——扩散模型的"安全"来自看不懂，而 MLLM 的语义理解把抽象、隐晦、跨语言的不安全意图全都打通了。

## 实验关键数据

### 主实验：不安全图像生成

| 数据集 | 指标 | SD3.5 Large | SD3.5 Large Turbo | Janus Pro (最差MLLM) | VILA-U | 
|--------|------|-------------|-------------------|---------------------|--------|
| TemplateLong | 平均不安全得分 | 0.280 | 0.200 | 0.613 | 高于扩散模型 |
| TemplateLongChinese | 平均不安全得分 | 0.000 | 0.000 | 高 | 高 |

各数据集上，扩散模型的不安全得分一致低于 5 个 MLLM。差距在 TemplateLong 上最为显著。

### 损坏图像率分析

| 模型 | 不安全提示损坏率 | 安全提示损坏率 |
|------|----------------|--------------|
| SD3.5 Large | 80.0% | 75.5% |
| SD3.5 Large Turbo | 66.0% | 66.5% |
| Bagel | 4.5% | 4.0% |
| Janus | 4.5% | 5.5% |
| Janus Pro | 0% | 0% |
| TokenFlow | 0% | 4.5% |
| VILA-U | 0% | 0% |

### 假图检测结果

MLLM 生成的图像对现有检测器构成更大挑战。即使针对 MLLM 数据微调检测器，跨 MLLM 泛化性较差。更关键的是，MLLM 可以通过丰富提示细节（v0→v1→v2）降低被检测概率——这一现象在扩散模型中不存在。

### 关键发现

- **MLLM 系统性更不安全**：在所有 5 个数据集上，MLLM 不安全得分一致高于扩散模型
- **不安全的根因是理解能力而非安全对齐失败**：扩散模型因无法理解复杂提示而产出损坏图像（被标记为安全），而非因安全训练更好
- **外部安全防护不够**：即使加上 NSFW 检测器，MLLM 的不安全得分仍高于扩散模型
- **检测器的泛化困境**：针对某个 MLLM 微调的检测器对其他 MLLM 效果有限，且 MLLM 可通过提示工程规避检测
- **MLLM 存在性别偏见**：性别中立的色情提示倾向于生成女性图像

## 亮点与洞察

- **"理解力即风险"的核心洞察极具启发性**：通常我们认为模型越"聪明"越好，但本文展示了理解能力在安全维度是双刃剑。这个框架可以推广到任何 AI 能力的安全性分析——任何提升"理解"的技术进步都可能同时扩大滥用面。
- **损坏图像作为"意外安全阀"的发现很有趣**：扩散模型不是因为"训练得更安全"而安全，而是因为"理解不了所以生成失败"。这意味着随着扩散模型理解能力提升，其安全风险也会相应增加。
- **提示丰富度-检测难度的关系**揭示了一种新型攻击向量：用户可以通过添加更多细节描述来让 MLLM 生成更难被检测的图像，这在扩散模型上无效，说明 MLLM 的生成机制从根本上不同。

## 局限与展望

- 所有实验都禁用了模型的安全机制，这虽然测量了模型内在倾向，但不完全反映实际部署场景
- 仅测试了 7 个模型，未覆盖 GPT-Image-1、Gemini 等主流闭源 MLLM 生成能力
- 安全分类依赖 OpenAI Moderation API（虽然与人类标注 89.2% 一致，但仍有约 10% 误差）
- 未深入分析 MLLM 内部安全对齐训练的具体机制，仅从外部行为推断
- 可探索：（1）基于 MLLM 语义理解特性的新型安全过滤器设计；（2）利用提示理解能力反过来做安全检测的思路

## 相关工作与启发

- **vs NSFW 安全过滤器**：本文表明即使加上外部过滤器，MLLM 仍比扩散模型更不安全，说明需要专门针对 MLLM 设计的安全机制
- **vs DE-FAKE / AIorNot 等假图检测器**：这些检测器针对扩散模型优化，对 MLLM 生成的图像效果显著下降，且微调后泛化性差，揭示了"检测器-生成器"军备竞赛在 MLLM 时代面临新挑战
- **vs 已有安全基准（I2P、Lexica）**：本文首次将这些基准扩展到 MLLM-扩散模型的跨范式对比，补充了 TemplateLong 等新测试集来测试语义理解差异

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性对比 MLLM 与扩散模型的安全风险，"理解即风险"的核心洞察新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个模型、5 个不安全数据集、4 个检测器、82,880 张生成图像，多维度定量+定性分析
- 写作质量: ⭐⭐⭐⭐ 研究问题清晰，实验设计严谨，案例分析直观
- 价值: ⭐⭐⭐⭐⭐ 对 MLLM 安全部署敲响警钟，揭示了现有安全基础设施在范式转移面前的系统性失效

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Hidden Dangers of Compositional Generation: Diagnosing Semantic Safety Failures in Text-to-Image Models](hidden_dangers_of_compositional_generation_diagnosing_semantic_safety_failures_i.md)
- [\[CVPR 2026\] When LoRA Betrays: Backdooring Text-to-Image Models by Masquerading as Benign Adapters](when_lora_betrays_backdooring_text-to-image_models_by_masquerading_as_benign_ada.md)
- [\[CVPR 2026\] SafeRoPE: Risk-specific Head-wise Embedding Rotation for Safe Generation in Rectified Flow Transformers](saferope_risk-specific_head-wise_embedding_rotation_for_safe_generation_in_recti.md)
- [\[CVPR 2026\] GenBreak: Red Teaming Text-to-Image Generation Using Large Language Models](genbreak_red_teaming_text-to-image_generation_using_large_language_models.md)
- [\[CVPR 2026\] Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting](generative_adversarial_perturbations_with_cross-paradigm_transferability_on_loca.md)

</div>

<!-- RELATED:END -->
