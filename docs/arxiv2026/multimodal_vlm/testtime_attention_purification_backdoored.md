# Test-Time Attention Purification for Backdoored Large Vision Language Models

**会议**: arXiv 2026  
**arXiv**: [2603.12989](https://arxiv.org/abs/2603.12989)  
**作者**: Zhifang Zhang, Bojun Yang, Shuo He, Weitong Chen, Wei Emma Zhang et al.
**代码**: 待确认  
**领域**: 多模态/VLM / LLM/NLP  
**关键词**: test-time, attention, purification, backdoored, large  

## 一句话总结
尽管具有强大的多模态性能，但大型视觉语言模型（LVLM）在微调后门攻击期间很容易受到攻击，其中对手将嵌入触发器的样本插入训练数据中，以植入可在测试时恶意激活的行为。

## 背景与动机
Despite the strong multimodal performance, large vision-language models (LVLMs) are vulnerable during fine-tuning to backdoor attacks, where adversaries insert trigger-embedded samples into the training data to implant behaviors that can be maliciously activated at test time.. Existing defenses typically rely on retraining backdoored parameters (e.g., adapters or LoRA modules) with clean data, which is computationally expensive and often degrades model performance.

## 核心问题
尽管具有强大的多模态性能，但大型视觉语言模型（LVLM）在微调后门攻击期间很容易受到攻击，其中对手将嵌入触发器的样本插入训练数据中，以植入可在测试时恶意激活的行为。

## 方法详解

### 整体框架
- Motivated by this, we propose CleanSight, a training-free, plug-and-play defense that operates purely at test time.

### 关键设计
1. **关键组件1**: Existing defenses typically rely on retraining backdoored parameters (e.g., adapters or LoRA modules) with clean data, which is computationally expensive and often degrades model performance.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 大量实验表明，CleanSight 在不同的数据集和后门攻击类型上显着优于现有的基于像素的净化防御，同时保留了模型在干净和中毒样本上的实用性。

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
