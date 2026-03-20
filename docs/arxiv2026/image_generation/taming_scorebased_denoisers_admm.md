# Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework

**会议**: arXiv 2026  
**arXiv**: [2603.10281](https://arxiv.org/abs/2603.10281)  
**作者**: Rajesh Shrestha, Xiao Fu
**代码**: 待确认  
**领域**: 扩散模型/生成 / 优化/理论  
**关键词**: taming, score-based, denoisers, admm, convergent  

## 一句话总结
虽然基于分数的生成模型已成为解决逆问题的强大先验，但将它们直接集成到 ADMM 等优化算法中仍然很重要。
## 背景与动机
While score-based generative models have emerged as powerful priors for solving inverse problems, directly integrating them into optimization algorithms such as ADMM remains nontrivial.. Two central challenges arise: i) the mismatch between the noisy data manifolds used to train the score functions and the geometry of ADMM iterates, especially due to the influence of dual variables, and ii) the lack of convergence understanding when ADMM is equipped with score-based denoisers.

## 核心问题
虽然基于分数的生成模型已成为解决逆问题的强大先验，但将它们直接集成到 ADMM 等优化算法中仍然很重要。
## 方法详解

### 整体框架
- To address the manifold mismatch issue, we propose ADMM plug-and-play (ADMM-PnP) with the AC-DC denoiser, a new framework that embeds a three-stage denoiser into ADMM: (1) auto-correction (AC) via additive Gaussian noise, (2) directional correction (DC) using conditional Langevin dynamics, and (3) score-based denoising.
- Experiments on a range of inverse problems demonstrate that our method consistently improves solution quality over a variety of baselines.

### 关键设计
1. **关键组件1**: In terms of convergence, we establish two results: first, under proper denoiser parameters, each ADMM iteration is a weakly nonexpansive operator, ensuring high-probability fixed-point $\textit{ball convergence}$ using a constant step size; second, under more relaxed conditions, the AC-DC denoiser is a bounded denoiser, which leads to convergence under an adaptive step size schedule.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 对一系列反问题的实验表明，我们的方法在各种基线上持续提高了解决方案的质量。
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
