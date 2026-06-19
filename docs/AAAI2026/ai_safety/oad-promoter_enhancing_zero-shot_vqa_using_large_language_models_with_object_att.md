---
title: >-
  [论文解读] OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description
description: >-
  [AAAI 2026][AI安全][视觉问答] 本文提出OAD-Promoter，通过对象集中样例生成（OEG）、记忆知识辅助（MKA）和OAD Prompt三个模块协同工作，在零样本设置下缓解LLM继承的语言偏差并提升领域迁移能力，在VQAv2等多个基准上取得SOTA。 领域现状：LLM已成为VQA任务中处理知识密集型问题…
tags:
  - "AAAI 2026"
  - "AI安全"
  - "视觉问答"
  - "零样本VQA"
  - "语言偏差"
  - "领域迁移"
  - "对象属性描述"
---

# OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description

**会议**: AAAI 2026  
**arXiv**: [2511.12131](https://arxiv.org/abs/2511.12131)  
**代码**: 无  
**领域**: 信息检索  
**关键词**: 视觉问答, 零样本VQA, 语言偏差, 领域迁移, 对象属性描述

## 一句话总结
本文提出OAD-Promoter，通过对象集中样例生成（OEG）、记忆知识辅助（MKA）和OAD Prompt三个模块协同工作，在零样本设置下缓解LLM继承的语言偏差并提升领域迁移能力，在VQAv2等多个基准上取得SOTA。

## 研究背景与动机

**领域现状**：LLM已成为VQA任务中处理知识密集型问题的关键工具。现有LLM-based KBVQA方法（PICa、Prophet、Img2LLM等）在few-shot和zero-shot场景下取得了显著成果。

**核心痛点——语言偏差**：语言偏差是VQA领域的顽疾。例如，训练数据中"What color...bananas?"的主导答案是"yellow"，模型倾向于利用这种表面相关性而非真正理解图像。这个问题不仅存在于传统VQA模型中，**在LLM-based方法中同样严重**——因为LLM在大规模数据上预训练时不可避免地学到了虚假相关性（shortcut learning）。

**两大负面影响**（图1）：

**预测不可靠**：LLM利用继承的语言偏差做推理，导致答案有偏

**OOD泛化差**：虽然LLM知识推理能力强，但语言偏差加剧了领域迁移的困难

**现有方法的盲区**：
- 现有LLM-based KBVQA方法忽视了全局和区域视觉信息的结合
- 没有辅助记忆模块帮助LLM应对分布变化的场景
- 去偏方法（如LMH、CSS）直接整合到LLM pipeline反而降低性能（实验验证）

**本文核心思路**：
1. 更精细的视觉信息可以缓解语言偏差（让LLM "看到"更多，减少依赖语言先验）
2. 记忆样例辅助可以提升推理可靠性，特别在领域迁移场景
3. 整合以上两点的Prompt可以持续提升领域适应能力

## 方法详解

### 整体框架
OAD-Promoter包含三个协同模块（图2）：
1. **OEG模块**（绿框）：生成全局标题和对象集中样例
2. **MKA模块**（蓝框）：利用存储样例辅助LLM处理新输入
3. **OAD Prompt**（红框）：整合前两个模块的输出，引导LLM推理

整个流程不依赖任何外部知识源或需要检索的数据，是纯零样本方法。

### 关键设计

#### 1. **OEG模块（Object-concentrated Example Generation）**

包含两个生成过程：

**多层级标题生成**：
- 使用预训练BLIP2生成**全局标题**（global caption），捕获图像整体语义
- 使用VinVL检测器生成**区域标题**（object-concentrated captions），聚焦单个对象属性

**合成问题生成**：
- 用标题评估工具从对象标题中提取潜在答案（名词短语、动词短语、形容词、数字、布尔词）
- 用在SQuAD2.0、MultiRC、BookQA、CommonsenseQA和Social IQa上微调过的T5-large模型生成对应问题
- 形成完整的对象集中样例 $E_i = (C, Q, A)$

设计动机：全局标题提供宏观理解，区域标题补充细粒度信息。二者结合使LLM获得更完整的视觉信息，减少依赖语言先验的机会。这些生成的样例同时用作MKA的记忆库和Prompt的组成部分。

#### 2. **MKA模块（Memory Knowledge Assistance）**

包含两个过程：

**答案估计**：
- **普通VQA模型**（UpDn）：输出普通答案 $A_O$（包含视觉信息）
- **偏差QA模型**（LMH中的off-shift模型）：输出偏差答案 $A_B$（不含视觉信息，纯语言偏差）

选择模式判定：
$$M = \begin{cases} Positive, & \text{if } A_O \neq A_B \\ Negative, & \text{if } A_O = A_B \end{cases}$$

关键洞察：若 $A_B = A_O$，说明即使不看图像也能得到相同答案——这暗示普通模型在利用语言偏差。由于LLM的训练规模远大于普通VQA模型，LLM利用该偏差的概率更高。

**相似度计算**：
$$E_S = \begin{cases} \text{argTopN} \frac{f^T f_j}{\|f\|_2 \|f_j\|_2}, & \text{if } M = Positive \\ \text{argBottomN} \frac{f^T f_j}{\|f\|_2 \|f_j\|_2}, & \text{if } M = Negative \end{cases}$$

- Positive模式：选最相似的存储样例（支持正常推理）
- **Negative模式**：选**最不相似**的存储样例（对抗语言偏差）

设计动机：通过预判偏差信号，主动选择与偏差方向相反的辅助样例，从而引导LLM避开语言偏差。随着推理进行，记忆库持续增长，领域适应能力不断增强。

#### 3. **OAD Prompt**

整合前两个模块输出的结构化Prompt：
$$[\text{Instruction } I \;/\; \text{Global Caption } C_G \;/\; \text{Object Examples } E_O \;/\; \text{Memory Examples } E_S \;/\; \text{Question } Q_O]$$

初始时MKA记忆为空，Prompt为 $[I / C_G / E_O / Q_O]$；第二个样本起变为完整形式。

与现有方法的关键区别：同时考虑全局描述和对象属性描述，而非仅用全局标题。消融实验证明CQA-CQA-CQA（每个样例保持完整triple）优于CCC-QAQAQA（分离排列）。

### 损失函数 / 训练策略
- OAD-Promoter本身不训练，是推理时的框架
- UpDn模型先在VQAv2+Visual Genome上预训练，再在OKVQA训练集上微调
- 主实验使用GPT-3和OPT作为冻结LLM
- 避免数据污染：从预训练数据中移除OKVQA测试集中出现的图像

## 实验关键数据

### 主实验——零样本设置下的性能

| 方法 | VQAv2 test | A-OKVQA test | OKVQA test |
|------|-----------|-------------|------------|
| Flamingo-80B | 56.21 | - | 50.57 |
| Img2LLM w/ GPT-3 | 59.22 | 43.39 | 42.80 |
| Img2LLM w/ OPT | 61.83 | 40.69 | 45.58 |
| **OAD-Promoter w/ OPT** | **61.93** | **40.68** | **45.58** |
| **OAD-Promoter w/ GPT-3** | **61.98** | **41.71** | **45.61** |

### 不同LLM的泛化验证（OKVQA零样本）

| LLM | 参数量 | OKVQA |
|-----|--------|-------|
| GPT-Neo | 2.7B | 33.41 |
| GPT-J | 6B | 38.89 |
| BLOOM | 7.1B | 33.77 |
| OPT | 6.7B | 36.18 |
| OPT | 30B | 40.46 |
| OPT | 175B | 45.58 |
| GPT-3 | 175B | 45.61 |

### 消融实验

| 配置 | OKVQA (Few-shot) | OKVQA (Zero-shot) | 说明 |
|------|-----------------|-------------------|------|
| 无OEG + 无MKA | 47.33 | 42.50 | 基线 |
| 有OEG + 无MKA | 54.68 | 44.26 | OEG贡献最大 |
| 无OEG + 有MKA | 48.95 | 43.64 | MKA独立也有帮助 |
| **有OEG + 有MKA** | **60.04** | **45.61** | 二者协同效果最佳 |

| MKA记忆样例数K | OKVQA | 说明 |
|---------------|-------|------|
| 0 | 43.64 | 无记忆 |
| 60 | 43.65 | 少量样例 |
| 200 | 43.92 | 中等样例 |
| 400 | 44.15 | 样例越多越好 |

### 领域迁移实验（Few-shot，不同LLM）

| LLM | VQA-CP | GQA-OOD |
|-----|--------|---------|
| GPT-4 (GRACE) | 57.61 | 50.19 |
| **GPT-4 (OAD-Promoter)** | 55.93 | **50.21** |

### 关键发现
1. 在VQAv2零样本设置下取得新SOTA（61.98），超越所有大规模多模态预训练方法和冻结LLM方法
2. OEG模块在few-shot下贡献最大（+7.35），证明细粒度视觉信息是缓解语言偏差的关键
3. 传统去偏方法（LMH、CSS）直接整合到LLM pipeline反而降低OKVQA性能（表4），说明LLM的偏差问题需要不同的解决策略
4. 在GPT-4上OAD-Promoter在GQA-OOD取得最佳成绩（50.21），说明更强的LLM能更好地发挥领域迁移能力
5. 记忆库随推理增长的机制使性能持续改善（K=400 > K=200 > K=60）
6. 改变输入顺序对OAD-Promoter无影响（100%正确率），而Img2LLM受顺序影响

## 亮点与洞察
1. **揭示了LLM中语言偏差的严重性**：不仅传统VQA有偏差问题，基于LLM的方法同样存在，且传统去偏方法无效
2. **对抗偏差的创新策略**：通过Negative模式选择最不相似的样例来对抗偏差，这个思路新颖且有效
3. **零样本方法超越few-shot**：得益于MKA的记忆增长机制，零样本设置下的推理能力持续增强
4. **即插即用**：框架可与不同LLM（GPT-3、OPT、BLOOM、GPT-Neo、GPT-J、GPT-4等）组合使用
5. **自增长的记忆库**：MKA模块随推理进行自然积累知识，这是一种优雅的持续学习形式

## 局限与展望
1. 依赖VinVL和BLIP2等预训练模型的质量，若这些模型失败则OEG模块输出受损
2. MKA模块的偏差检测依赖UpDn和LMH的QA模块，这些模型本身能力有限
3. 零样本设置下在A-OKVQA和OKVQA上的提升幅度有限（<1%），说明方法在更难的知识推理问题上的天花板
4. 记忆库无限增长可能带来存储和检索效率问题
5. Positive/Negative选择模式是硬切换，可考虑软插值

## 相关工作与启发
- 偏差检测思路（比较有视觉 vs 无视觉的预测）可以推广到其他需要检测shortcut learning的场景
- 记忆增强推理的框架对其他需要持续改善的推理任务有参考价值
- 多粒度视觉信息（全局+区域）的整合策略值得其他VLM工作借鉴

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Hierarchically Robust Zero-shot Vision-language Models](../../CVPR2026/ai_safety/hierarchically_robust_zero-shot_vision-language_models.md)
- [\[ICML 2026\] Calibrating Uncertainty for Zero-Shot Adversarial CLIP](../../ICML2026/ai_safety/calibrating_uncertainty_for_zero-shot_adversarial_clip.md)
- [\[CVPR 2026\] Towards Robust Multimodal Large Language Models Against Jailbreak Attacks](../../CVPR2026/ai_safety/towards_robust_multimodal_large_language_models_against_jailbreak_attacks.md)
- [\[CVPR 2026\] SIF: Semantically In-Distribution Fingerprints for Large Vision-Language Models](../../CVPR2026/ai_safety/sif_semantically_in-distribution_fingerprints_for_large_vision-language_models.md)
- [\[CVPR 2026\] GenBreak: Red Teaming Text-to-Image Generation Using Large Language Models](../../CVPR2026/ai_safety/genbreak_red_teaming_text-to-image_generation_using_large_language_models.md)

</div>

<!-- RELATED:END -->
