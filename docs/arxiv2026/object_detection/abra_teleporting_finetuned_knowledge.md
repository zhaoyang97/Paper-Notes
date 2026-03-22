# ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection

**会议**: arXiv 2026  
**arXiv**: [2603.12409](https://arxiv.org/abs/2603.12409)  
**作者**: Mattia Bernardi, Chiara Cappellino, Matteo Mosconi, Enver Sangineto, Angelo Porrello
**代码**: 待确认  
**领域**: 目标检测  
**关键词**: abra, teleporting, fine-tuned, knowledge, across  

## 一句话总结
尽管最近的开放词汇目标检测架构（例如 Grounding DINO）展示了强大的零样本能力，但它们的性能在域转换下会显着下降。

## 背景与动机
Although recent Open-Vocabulary Object Detection architectures, such as Grounding DINO, demonstrate strong zero-shot capabilities, their performance degrades significantly under domain shifts.. Moreover, many domains of practical interest, such as nighttime or foggy scenes, lack large annotated datasets, preventing direct fine-tuning.

## 核心问题
此外，许多实际感兴趣的领域，例如夜间或雾天场景，缺乏大型注释数据集，无法直接进行微调。

## 方法详解

### 整体框架
- In this paper, we introduce Aligned Basis Relocation for Adaptation(ABRA), a method that transfers class-specific detection knowledge from a labeled source domain to a target domain where no training images containing these classes are accessible.

### 关键设计
待深读后补充。

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 尽管最近的开放词汇目标检测架构（例如 Grounding DINO）展示了强大的零样本能力，但它们的性能在域转换下会显着下降。
- 跨越具有挑战性的领域转换的广泛实验表明，ABRA 在多种不利条件下成功传送了职业级别的专业化。

## 局限性 / 可改进方向
- 待深读后补充

## 与相关工作的对比
待深读后补充。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
