# Multimodal classification of Radiation-Induced Contrast Enhancements and tumor recurrence using deep learning

**会议**: arXiv 2026  
**arXiv**: [2603.11827](https://arxiv.org/abs/2603.11827)  
**作者**: Robin Peretzke, Marlin Hanstein, Maximilian Fischer, Lars Badhi Wessel, Obada Alhalabi
**代码**: 待确认  
**领域**: 医学图像  
**关键词**: multimodal, classification, radiation-induced, contrast, enhancements  

## 一句话总结
胶质母细胞瘤患者治疗后的肿瘤复发和放射引起的对比增强之间的区别仍然是主要的临床挑战。

## 背景与动机
The differentiation between tumor recurrence and radiation-induced contrast enhancements in post-treatment glioblastoma patients remains a major clinical challenge.. Existing approaches rely on clinically sparsely available diffusion MRI or do not consider radiation maps, which are gaining increasing interest in the tumor board for this differentiation.

## 核心问题
胶质母细胞瘤患者治疗后的肿瘤复发和放射引起的对比增强之间的区别仍然是主要的临床挑战。

## 方法详解

### 整体框架
- We introduce RICE-NET, a multimodal 3D deep learning model that integrates longitudinal MRI data with radiotherapy dose distributions for automated lesion classification using conventional T1-weighted MRI data.

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
- 该模型使用 92 名患者组成的队列，在独立测试集上取得了 0.92 的 F1 分数。
- 在广泛的消融实验中，我们量化了每个时间点和模式的贡献，并表明可靠的分类很大程度上取决于辐射图。

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
