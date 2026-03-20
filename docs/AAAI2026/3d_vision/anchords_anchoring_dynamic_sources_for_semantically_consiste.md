# AnchorDS: Anchoring Dynamic Sources for Semantically Consistent Text-to-3D Generation

**会议**: AAAI 2026  
**arXiv**: [2511.11692v1](https://arxiv.org/abs/2511.11692v1)  
**代码**: 暂无  
**领域**: 3D视觉  
**关键词**: 知识蒸馏, 生成, 3D  

## 一句话总结
Optimization-based text-to-3D methods distill guidance from 2D generative models via Score Distillation Sampling (SDS), but implicitly treat this guidance as static.

## 背景与动机
Optimization-based text-to-3D methods distill guidance from 2D generative models via Score Distillation Sampling (SDS), but implicitly treat this guidance as static. We cast the problem into a dual-conditioned 潜空间, conditioned on both the text prompt and the intermediately rendered image.

## 核心问题
Optimization-based text-to-3D methods distill guidance from 2D generative models via Score Distillation Sampling (SDS), but implicitly treat this guidance as static.

## 方法详解

### 整体框架
本工作 shows that ignoring source dynamics yields inconsistent trajectories that suppress or merge semantic cues, leading to "semantic over-smoothing" artifacts.

### 关键设计
1. 本工作 shows that ignoring source dynamics yields inconsistent trajectories that suppress or merge semantic cues, leading to "semantic over-smoothing" artifacts.
2. As such, we reformulate text-to-3D optimization as mapping a dynamically evolving source distribution to a fixed target distribution.
3. Building on this insight, we introduce AnchorDS, an improved score distillation mechanism that provides state-anchored guidance with image conditions and stabilizes generation.
4. We further penalize erroneous source estimates and design a lightweight filter strategy and 微调 strategy that refines the anchor with negligible overhead.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
- 大量实验 show that our method 超越 先前方法 in both quality and efficiency.

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 被AAAI 2026接收
- 待深读后补充

## 局限性 / 可改进方向
- 待深读后补充

## 与相关工作的对比
待深读后补充。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 对我的价值: ⭐⭐⭐
