# Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems

**会议**: arXiv 2026  
**arXiv**: [2603.13069](https://arxiv.org/abs/2603.13069)  
**作者**: Ann Dooms
**代码**: 待确认  
**领域**: 扩散模型/生成  
**关键词**: fractals, made, practical, denoising, diffusion  

## 一句话总结
当扩散模型将噪声转化为照片时，它实际上在做什么？
## 背景与动机
What is a diffusion model actually doing when it turns noise into a photograph?. We show that the deterministic DDIM reverse chain operates as a Partitioned Iterated Function System (PIFS) and that this framework serves as a unified design language for denoising diffusion model schedules, architectures, and training objectives.

## 核心问题
当扩散模型将噪声转化为照片时，它实际上在做什么？我们展示了确定性 DDIM 反向链作为分区迭代函数系统 (PIFS) 运行，并且该框架充当去噪扩散模型计划、架构和训练目标的统一设计语言。
## 方法详解

### 整体框架
- We show that the deterministic DDIM reverse chain operates as a Partitioned Iterated Function System (PIFS) and that this framework serves as a unified design language for denoising diffusion model schedules, architectures, and training objectives.
- Through the study of the fractal geometry of the PIFS, we derive three optimal design criteria and show that four prominent empirical design choices (the cosine schedule offset, resolution-dependent logSNR shift, Min-SNR loss weighting, and Align Your Steps sampling) each arise as approximate solutions to our explicit geometric optimization problems tuning theory into practice.

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
- 待深读后补充

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
