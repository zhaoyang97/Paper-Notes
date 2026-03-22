# Hybrid eTFCE-GRF: Exact Cluster-Size Retrieval with Analytical p-Values for Voxel-Based Morphometry

**会议**: arXiv 2026  
**arXiv**: [2603.11344](https://arxiv.org/abs/2603.11344)  
**作者**: Don Yin, Hao Chen, Takeshi Miki, Boxing Liu, Enyu Yang
**代码**: 待确认  
**领域**: 医学图像  
**关键词**: hybrid, etfce-grf, exact, cluster-size, retrieval  

## 一句话总结
无阈值聚类增强 (TFCE) 跨阈值集成聚类范围，以改进体素神经影像推理，但排列测试使其对于大型数据集来说速度过慢。

## 背景与动机
Threshold-free cluster enhancement (TFCE) integrates cluster extent across thresholds to improve voxel-wise neuroimaging inference, but permutation testing makes it prohibitively slow for large datasets.. Probabilistic TFCE (pTFCE) uses analytical Gaussian random field (GRF) p-values but discretises the threshold grid.

## 核心问题
无阈值聚类增强 (TFCE) 跨阈值集成聚类范围，以改进体素神经影像推理，但排列测试使其对于大型数据集而言速度过慢。概率 TFCE (pTFCE) 使用分析高斯随机场 (GRF) p 值，但离散化阈值网格。

## 方法详解

### 整体框架
- Probabilistic TFCE (pTFCE) uses analytical Gaussian random field (GRF) p-values but discretises the threshold grid.
- Exact TFCE (eTFCE) eliminates discretisation via a union-find data structure but still requires permutations.
- We combine eTFCE's union-find for exact cluster-size retrieval with pTFCE's analytical GRF inference.

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
- 无阈值聚类增强 (TFCE) 跨阈值集成聚类范围，以改进体素神经影像推理，但排列测试使其对于大型数据集来说速度过慢。

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
