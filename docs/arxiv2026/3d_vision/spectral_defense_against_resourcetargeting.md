# Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting

**会议**: arXiv 2026  
**arXiv**: [2603.12796](https://arxiv.org/abs/2603.12796)  
**作者**: Yang Chen, Yi Yu, Jiaming He, Yueqi Duan, Zheng Zhu et al.
**代码**: 待确认  
**领域**: 3D视觉 / 隐私/安全/公平  
**关键词**: spectral, defense, against, resource-targeting, attack  

## 一句话总结
3D 高斯泼溅 (3DGS) 的最新进展可提供高质量渲染，但高斯表示暴露了新的攻击面，即资源定位攻击。

## 背景与动机
Recent advances in 3D Gaussian Splatting (3DGS) deliver high-quality rendering, yet the Gaussian representation exposes a new attack surface, the resource-targeting attack.. This attack poisons training images, excessively inducing Gaussian growth to cause resource exhaustion.

## 核心问题
3D 高斯分布 (3DGS) 的最新进展提供了高质量的渲染，但高斯表示暴露了一个新的攻击面，即资源定位攻击。这种攻击会毒害训练图像，过度诱导高斯增长导致资源耗尽。

## 方法详解

### 整体框架
- Recent advances in 3D Gaussian Splatting (3DGS) deliver high-quality rendering, yet the Gaussian representation exposes a new attack surface, the resource-targeting attack.
- As a result, poisoned inputs introduce abnormal high-frequency amplifications that mislead 3DGS into interpreting noisy patterns as detailed structures, ultimately causing unstable Gaussian overgrowth and degraded scene fidelity.
- To address this, we propose \textbf{Spectral Defense} in Gaussian and image fields.
- We first design a 3D frequency filter to selectively prune Gaussians exhibiting abnormally high frequencies.

### 关键设计
1. **关键组件1**: We first design a 3D frequency filter to selectively prune Gaussians exhibiting abnormally high frequencies.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 实验表明，我们的防御构建了强大、准确且安全的 3DGS，在攻击下将过度生长抑制高达 $5.92\times$，将内存减少高达 $3.66\times$，并将速度提高高达 $4.34\times$。

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
