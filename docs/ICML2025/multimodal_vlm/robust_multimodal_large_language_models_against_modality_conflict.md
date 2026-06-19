---
title: >-
  [论文解读] Robust Multimodal Large Language Models Against Modality Conflict
description: >-
  [ICML 2025][多模态VLM][多模态幻觉] 揭示 MLLM 幻觉的一个被忽视来源——模态冲突（视觉输入与文本输入之间的固有矛盾），从对象/属性/关系三个层面形式化定义模态冲突，构建 20K 样例的 MMMC 数据集，并提出 prompt engineering、SFT 和 RL 三种缓解方法，其中 RL 效果最佳。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "多模态幻觉"
  - "模态冲突"
  - "鲁棒性"
  - "强化学习"
  - "SFT"
---

# Robust Multimodal Large Language Models Against Modality Conflict

**会议**: ICML 2025  
**arXiv**: [2507.07151](https://arxiv.org/abs/2507.07151)  
**代码**: [https://github.com/zmzhang2000/MMMC](https://github.com/zmzhang2000/MMMC)  
**领域**: 多模态VLM  
**关键词**: 多模态幻觉, 模态冲突, 鲁棒性, 强化学习, SFT

## 一句话总结

揭示 MLLM 幻觉的一个被忽视来源——模态冲突（视觉输入与文本输入之间的固有矛盾），从对象/属性/关系三个层面形式化定义模态冲突，构建 20K 样例的 MMMC 数据集，并提出 prompt engineering、SFT 和 RL 三种缓解方法，其中 RL 效果最佳。

## 研究背景与动机

### 1. MLLM 幻觉问题

多模态大语言模型在 VQA 等任务上表现出色，但容易产生幻觉——生成与输入不一致的信息。现有工作主要关注模型输出与输入之间的不一致。

### 2. 被忽视的幻觉来源：模态冲突

本文关注的是输入本身的矛盾：当文本提问预设了图像中不存在的信息时，MLLM 陷入困境。例如：图片是一只狗在冲浪，用户问"球是什么颜色的？"——图中根本没有球，但模型可能幻觉出"球是绿色的"。

### 3. 与已有工作的区别

已有幻觉缓解方法（改进训练数据、调整解码策略、RLHF 对齐）主要追求更精确的跨模态特征对齐。但即使模态对齐完美，面对固有冲突的输入，模型仍会产生幻觉——这需要从根本上增强模型识别和处理输入矛盾的能力。

## 方法详解

### 整体框架

1. 形式化定义三类模态冲突（对象/属性/关系）
2. 构建 MMMC 数据集（20K 样本，18K 训练 + 2K 测试）
3. 提出三种缓解方法并对比：prompt engineering、SFT、RL

### 关键设计

#### 1. 模态冲突形式化定义

设视觉输入为 $\mathcal{V}$，文本输入为 $\mathcal{T}$：

**对象冲突**：文本涉及图中不存在的对象
$$\text{Obj}(\mathcal{T}) \not\subseteq \text{Obj}(\mathcal{V})$$

**属性冲突**：相同对象但属性不同（如文本说红苹果，图片是绿苹果）
$$\text{Attr}(\mathcal{O}_i^{\mathcal{T}}) \neq \text{Attr}(\mathcal{O}_i^{\mathcal{V}})$$

**关系冲突**：相同对象但关系不同（如文本说猫在桌上，图片是猫在地上）
$$\text{Rel}(\mathcal{O}_i^{\mathcal{T}}, \mathcal{O}_j^{\mathcal{T}}) \neq \text{Rel}(\mathcal{O}_i^{\mathcal{V}}, \mathcal{O}_j^{\mathcal{V}})$$

#### 2. MMMC 数据集构建

基于 Visual Genome 数据集，通过 4 步构建：
1. **基础问题采样**：从原始数据集随机采样问题
2. **关键组件检测**：用 LLM 检测图像中的对象/属性/关系
3. **组件替换**：将问题中的组件替换为与图像矛盾的信息
4. **答案生成**：不直接让 VLM 看图回答（避免幻觉），而是基于文本信息用 LLM 生成正确答案（如"图中没有球"）
5. 人工审核确保质量

#### 3. 方法一：Prompt Engineering

在问题前加提示："Please check if the image contains mentioned information and answer the question"

$$\mathcal{A} \sim \pi_\theta(\mathcal{A}|\mathcal{V}, p(\mathcal{T}))$$

优点：零成本，不需额外训练。缺点：效果取决于模型的指令遵循能力。

#### 4. 方法二：监督微调（SFT）

在 MMMC 训练集上用语言模型目标微调：

$$\pi_\theta^* = \arg\min_\theta \mathbb{E}[-\log \pi_\theta(\mathcal{A}|\mathcal{V}, \mathcal{T})]$$

优点：可利用训练数据。缺点：主要学习目标域的风格适配，对未见数据泛化有限。

#### 5. 方法三：强化学习（RL）

将条件生成建模为 MDP，设计奖励函数衡量模型是否正确识别了模态冲突：
- 状态：$s_t = (\mathcal{V}, \mathcal{T}, a_{<t})$
- 动作：$a_t$（生成的 token）
- 奖励：基于回复是否正确指出冲突

优点：通过试错学习更鲁棒的策略。在实验中效果最好。

## 实验关键数据

### 主实验：不同 MLLM 在 MMMC 上的表现

| 模型 | 对象冲突 Acc | 属性冲突 Acc | 关系冲突 Acc | 平均 |
|------|------------|------------|------------|------|
| InternLM-XComposer2 | 32.1 | 28.5 | 25.3 | 28.6 |
| LLaVA-1.5 | 35.7 | 31.2 | 27.8 | 31.6 |
| Qwen-VL-Chat | 38.2 | 33.6 | 30.1 | 34.0 |
| GPT-4o | 62.5 | 55.3 | 48.7 | 55.5 |

大多数 MLLM 在模态冲突场景下准确率极低（<40%），即使 GPT-4o 也仅 ~55%。

### 三种方法对比（以 LLaVA-1.5 为例）

| 方法 | 对象冲突 | 属性冲突 | 关系冲突 | 平均 | 原始VQA保持 |
|------|---------|---------|---------|------|------------|
| 基准（无干预） | 35.7 | 31.2 | 27.8 | 31.6 | 100% |
| Prompt Engineering | 42.3 | 37.8 | 33.5 | 37.9 | ~99% |
| SFT | 68.5 | 62.1 | 56.3 | 62.3 | ~95% |
| **RL** | **74.2** | **67.8** | **61.5** | **67.8** | **~93%** |

- RL 提升最大（+36.2% 平均），但原始 VQA 性能略有下降
- SFT 效果稳定，与原始能力平衡更好
- Prompt Engineering 提升有限但零成本

### 关键发现

- 关系冲突最难处理（所有方法表现最差），因为需要更复杂的空间推理
- RL 方法在三类冲突上都显著优于 SFT，说明试错探索对这类"辨别力"任务更有效
- SFT 在训练分布内效果好但泛化受限——作者指出 SFT 更多学习了"风格"而非"能力"

## 亮点与洞察

- **问题定义的独特性**：首次正式定义和研究输入间的模态冲突作为幻觉源，与已有的"输出-输入冲突"视角互补
- **三级冲突分类**：对象/属性/关系的层次化定义既系统又实用
- **方法对比的公平性**：三种方法代表了从零成本到重训练的完整谱系，消融设计合理
- **数据构建的巧妙**：用 LLM 基于文本信息生成答案（而非让 VLM 看图），避免了引入新的幻觉

## 局限与展望

- MMMC 数据集基于 Visual Genome 构建，场景多样性受限于该数据集的覆盖范围
- RL 方法的奖励函数设计仍较简单，结合更精细的冲突检测奖励可能进一步提升
- 未测试在真实用户交互场景中的表现——实际用户的冲突提问可能更隐晦
- 可探索将冲突识别能力集成到预训练阶段而非仅后训练
- SFT+RL 的联合训练（先 SFT 再 RL）的效果待验证

## 相关工作与启发

- **vs POPE/CHAIR 等幻觉检测**：这些方法评估输出幻觉，本文关注输入端的冲突引发
- **vs Longpre et al. (2021)**：在 LLM 中研究知识冲突，本文扩展到多模态视觉-语言场景
- **vs 视觉对抗攻击**：对抗攻击修改图像像素，本文通过自然语言制造语义冲突
- **启发**：可将模态冲突作为 MLLM 安全评测的标准维度之一

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次形式化定义和系统研究多模态输入冲突
- 实验充分度: ⭐⭐⭐⭐ 多模型评测+三种方法对比+消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 定义清晰、方法层次分明、图示直观
- 价值: ⭐⭐⭐⭐⭐ 开辟了 MLLM 鲁棒性研究的新维度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DeepAlign: Mitigating Modality Conflict through Modality-Specific Alignment](../../CVPR2026/multimodal_vlm/deepalign_mitigating_modality_conflict_through_modality-specific_alignment.md)
- [\[CVPR 2026\] Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/predictive_regularization_against_visual_representation_degradation_in_multimoda.md)
- [\[ICML 2025\] Ranked from Within: Ranking Large Multimodal Models Without Labels](ranked_from_within_ranking_large_multimodal_models_without_labels.md)
- [\[CVPR 2026\] Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](../../CVPR2026/multimodal_vlm/enhance-then-balance_modality_collaboration_for_robust_multimodal_sentiment_anal.md)
- [\[ACL 2025\] Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation](../../ACL2025/multimodal_vlm/single-to-mix_modality_alignment_with_multimodal_large_language_model_for_docume.md)

</div>

<!-- RELATED:END -->
