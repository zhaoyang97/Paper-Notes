---
title: >-
  [论文解读] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning
description: >-
  [多模态VLM] SC-Captioner 提出了一种基于策略梯度的多轮强化学习框架，通过设计包含正确性奖励和错误惩罚的纠错奖励函数，使大型视觉语言模型获得图像描述的自纠错能力，同时提出改进的 CAPTURE 评估指标。 大型视觉语言模型（LVLM）在图像描述任务中面临精确率与召回率的平衡问题： - 精确率：生成内容中可能包…
tags:
  - "多模态VLM"
---

# SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2508.06125](https://arxiv.org/abs/2508.06125)
- **代码**: [GitHub](https://github.com/zl2048/SC-Captioner)
- **领域**: 多模态 / 图像描述
- **关键词**: 图像描述, 自纠错, 强化学习, LVLM, 评估指标

## 一句话总结

SC-Captioner 提出了一种基于策略梯度的多轮强化学习框架，通过设计包含正确性奖励和错误惩罚的纠错奖励函数，使大型视觉语言模型获得图像描述的自纠错能力，同时提出改进的 CAPTURE 评估指标。

## 研究背景与动机

大型视觉语言模型（LVLM）在图像描述任务中面临精确率与召回率的平衡问题：
- **精确率**：生成内容中可能包含图像中不存在的描述（幻觉问题）
- **召回率**：更长的描述虽能提高召回，但更易引入幻觉

一个自然的问题是：**LVLM 能否通过自纠错改善初始描述？**

先前研究表明 LLM 的内在自纠错（无额外训练）难以有效改善输出。在图像描述中，直接提示模型自纠错可能删除正确描述并添加幻觉。因此需要专门的训练策略来激活自纠错能力。

## 方法详解

### 整体框架

SC-Captioner 采用多轮强化学习：
1. 策略模型根据图像和初始指令生成第一轮描述 $y_1$
2. 将第一轮对话与自纠错指令拼接，生成第二轮描述 $y_2$
3. 计算基于纠错的奖励，通过策略梯度更新模型

### 训练目标

$$L = -R(y_1, y_2, y^*) \cdot \log \pi_\theta(y_2 | [I, x_1, y_1, x_2]) + \beta L_{\text{KL}}$$

- 当奖励为正时，增大 $y_2$ 的生成概率（鼓励正确纠错）
- 当奖励为负时，减小 $y_2$ 的生成概率（抑制错误纠错）
- KL 损失 $L_{\text{KL}}$ 保持初始描述能力

### 关键设计：奖励函数

奖励函数基于场景图解析，将描述分解为**物体、属性、关系**三个集合。

**正确性奖励（Correctness Bonus）**：
- 使用 FACTUAL 解析器提取物体/属性/关系集合
- 计算 $y_1$ 和 $y_2$ 的集合差获取新增和删除的元素
- 将这些元素与 GT 集合匹配，计算软分数和硬分数

对于新增元素：$\mathbf{S_a} = \{\max_{o^* \in \mathbf{O}_{y^*}} s(o_a, o^*) \mid o_a \in \mathbf{O}_{y_2} \setminus \mathbf{O}_{y_1}\}$

软分数 $\sum_{s \in \mathbf{S_a}}(s - \tau_a)$ 衡量添加的有效性，硬分数 $\sum_{s \in \mathbf{S_a}} \mathbf{1}_{\{s > \tau_a'\}}$ 奖励正确添加。

**错误惩罚（Mistake Punishment）**：
- 添加了幻觉（$y_2$ 中有但 $y_1 \cup y^*$ 中无）：降低奖励
- 删除了正确描述（$y_1 \cup y^*$ 中有但 $y_2$ 中无）：降低奖励
- 关系部分不施加惩罚（匹配过于复杂）

### 改进的评估指标

对 CAPTURE 指标进行三方面改进：

1. **物体精确率**：引入 GPT-4o 生成的额外物体用于精确率计算，避免因 GT 不完整导致的假阳性惩罚
2. **属性匹配**：仅匹配同一或相似物体的属性（CAPTURE 全局匹配导致不同物体的相似属性错误匹配）
3. **关系评估**：用问答（QA）方式替代三元组匹配，每张图像生成 5 个关系问题由 LLM 回答

### 数据集 RefinedCaps

- 从 COCO 2017 训练集采样 6.5K 图像
- GPT-4o 生成初始描述
- 人工专家修正幻觉、添加遗漏描述
- 确保至少 80% 的图像物体出现在描述中

## 实验关键数据

### 主实验：DOCCI500 数据集

| 基础模型 | 训练方式 | CAPTURE | Objects F1 | Attributes F1 | Relations QA |
|----------|----------|---------|------------|---------------|-------------|
| Qwen2-VL-7B | Zero-shot* | 57.62 | 65.88 | 52.92 | 17.01 |
| Qwen2-VL-7B | SFT* | 62.02 | 69.32 | 55.78 | 27.93 |
| Qwen2-VL-7B | SFT+DPO* | 62.51 | 70.67 | 55.60 | 27.33 |
| **Qwen2-VL-7B** | **SFT+Ours*** | **63.34** | **71.63** | **57.67** | **30.51** |
| LLaVA-1.5-7B | SFT* | 61.03 | 68.55 | 54.05 | 20.11 |
| LLaVA-1.5-7B | SFT+DPO* | 61.09 | 68.89 | 53.61 | 21.73 |
| **LLaVA-1.5-7B** | **SFT+Ours*** | **62.29** | **70.30** | **57.10** | **22.69** |

（* 表示自纠错后的描述评分）

### 消融实验：奖励函数组成

| Object | Attribute | Relation | Mistake | CAPTURE | Obj F1 | Attr F1 | Rel QA |
|--------|-----------|----------|---------|---------|--------|---------|--------|
| | | | | 62.02 | 69.32 | 55.78 | 27.93 |
| ✓ | | | | 62.34 | 70.43 | 57.02 | 29.66 |
| ✓ | ✓ | | | 62.83 | 71.44 | 56.30 | 29.42 |
| ✓ | ✓ | ✓ | | 63.20 | 71.54 | 57.11 | 29.96 |
| ✓ | ✓ | ✓ | ✓ | **63.34** | **71.63** | **57.67** | **30.51** |

### 关键发现

1. **自纠错有效**：SFT+Ours 在两个测试集上一致提升，CAPTURE 分数提升 1.3+
2. **DPO 不理想**：DPO 虽能激活自纠错动作，但高召回低精确率，即"会纠但纠不好"
3. **错误惩罚关键**：移除 mistake punishment 后精确率显著下降
4. **各组件互补**：物体、属性、关系奖励分别提升对应指标，组合后效果最佳
5. **纠错行为分析**：SC-Captioner 的插入多于删除（相对 SFT），表明学会了有效添加缺失内容

## 亮点与洞察

1. **首次将自纠错训练引入图像描述**：不同于之前仅在数学/代码上探索自纠错的工作
2. **奖励函数设计精巧**：基于场景图的元素级奖励直接反映纠错效果，比使用整体指标作为奖励更有效
3. **评估指标改进实用**：解决了 CAPTURE 的三个主要缺陷，特别是属性的全局匹配和关系的简单三元组匹配问题
4. **RefinedCaps 数据集**：6.5K 高质量人工精修数据，SFT 本身就带来显著提升

## 局限性

1. 奖励函数依赖 FACTUAL 场景图解析器的质量，解析错误会传播到训练信号
2. GT 描述不完整时，错误惩罚可能惩罚正确但未在 GT 中提及的新增内容
3. 仅在 7B 模型上验证，未验证更大或更小模型的效果
4. 关系评估基于 QA 方式，需要 GPT-4o 参与测试集构建和 LLM 回答问题

## 相关工作与启发

- **SCoRe**：在数学推理上的多轮 RL 自纠错，本文将其思路拓展到视觉-语言领域
- **CAPTURE**：提出基于场景图的描述评估框架，本文改进了其三个缺陷
- **DPO**：作为对比基线，证明偏好优化不足以学习有效的自纠错策略

## 评分

⭐⭐⭐⭐ — 问题定义新颖，奖励函数设计巧妙且实验充分，但对场景图解析质量的依赖和 GT 不完整性的处理仍有改进空间。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](../../AAAI2026/multimodal_vlm/recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](../../ICLR2026/multimodal_vlm/vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](mm-ifengine_towards_multimodal_instruction_following.md)

</div>

<!-- RELATED:END -->
