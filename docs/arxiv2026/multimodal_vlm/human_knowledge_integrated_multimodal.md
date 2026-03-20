# Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization

**会议**: arXiv 2026  
**arXiv**: [2603.12369](https://arxiv.org/abs/2603.12369)  
**作者**: Ayan Banerjee, Kuntal Thakur, Sandeep Gupta
**代码**: 待确认  
**领域**: 多模态/VLM  
**关键词**: human, knowledge, integrated, multi-modal, learning  

## 一句话总结
在基于眼底图像的糖尿病视网膜病变 (DR) 分级和静息态 fMRI 癫痫发作区 (SOZ) 检测等关键任务中，跨领域推广图像分类仍然具有挑战性。
## 背景与动机
Generalizing image classification across domains remains challenging in critical tasks such as fundus image-based diabetic retinopathy (DR) grading and resting-state fMRI seizure onset zone (SOZ) detection.. When domains differ in unknown causal factors, achieving cross-domain generalization is difficult, and there is no established methodology to objectively assess such differences without direct metadata or protocol-level information from data collectors, which is typically inaccessible.

## 核心问题
在基于眼底图像的糖尿病视网膜病变 (DR) 分级和静息态 fMRI 癫痫发作区 (SOZ) 检测等关键任务中，跨领域推广图像分类仍然具有挑战性。
## 方法详解

### 整体框架
- We first introduce domain conformal bounds (DCB), a theoretical framework to evaluate whether domains diverge in unknown causal factors.
- Building on this, we propose GenEval, a multimodal Vision Language Models (VLM) approach that combines foundational models (e.g., MedGemma-4B) with human knowledge via Low-Rank Adaptation (LoRA) to bridge causal gaps and enhance single-source domain generalization (SDG).

### 关键设计
1. **关键组件1**: We first introduce domain conformal bounds (DCB), a theoretical framework to evaluate whether domains diverge in unknown causal factors.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 在 8 个 DR 和两个 SOZ 数据集上，GenEval 实现了卓越的 SDG 性能，平均准确度为 69.2% (DR) 和 81% (SOZ)，分别比最强基线高出 9.4% 和 1.8%。
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
