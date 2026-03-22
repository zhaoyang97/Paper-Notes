# Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation

**会议**: arXiv 2026  
**arXiv**: [2603.12845](https://arxiv.org/abs/2603.12845)  
**作者**: Fei Wang, Xinye Zheng, Kun Li, Yanyan Wei, Yuxin Liu
**代码**: 待确认  
**领域**: LLM/NLP  
**关键词**: multimodal, protein, language, models, enzyme  

## 一句话总结
预测酶动力学参数可以量化酶在规定的生化条件下催化特定底物的效率。

## 背景与动机
Predicting enzyme kinetic parameters quantifies how efficiently an enzyme catalyzes a specific substrate under defined biochemical conditions.. Canonical parameters such as the turnover number ($k_\text{cat}$), Michaelis constant ($K_\text{m}$), and inhibition constant ($K_\text{i}$) depend jointly on the enzyme sequence, the substrate chemistry, and the conformational adaptation of the active site during binding.

## 核心问题
许多学习流程将这个过程简化为酶和底物之间的静态兼容性问题，通过浅层操作融合它们的表示并回归单个值。

## 方法详解

### 整体框架
- Many learning pipelines simplify this process to a static compatibility problem between the enzyme and substrate, fusing their representations through shallow operations and regressing a single value.
- In this regard, we reformulate kinetic prediction as a staged multimodal conditional modeling problem and introduce the Enzyme-Reaction Bridging Adapter (ERBA), which injects cross-modal information via fine-tuning into Protein Language Models (PLMs) while preserving their biochemical priors.
- ERBA performs conditioning in two stages: Molecular Recognition Cross-Attention (MRCA) first injects substrate information into the enzyme representation to capture specificity; Geometry-aware Mixture-of-Experts (G-MoE) then integrates active-site structure and routes samples to pocket-specialized experts to reflect induced fit.

### 关键设计
1. **关键组件1**: ERBA performs conditioning in two stages: Molecular Recognition Cross-Attention (MRCA) first injects substrate information into the enzyme representation to capture specificity; Geometry-aware Mixture-of-Experts (G-MoE) then integrates active-site structure and routes samples to pocket-specialized experts to reflect induced fit.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 通过跨越三个动力学端点和多个 PLM 主干的实验，与仅序列和浅融合基线相比，ERBA 提供了一致的增益和更强的分布外性能，为可扩展的动力学预测提供了一条基于生物学的途径，并为添加辅因子、突变和时间分辨结构线索奠定了基础。

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
- 对我的价值: ⭐⭐⭐
