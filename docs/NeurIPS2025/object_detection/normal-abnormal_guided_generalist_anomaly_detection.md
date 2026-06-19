---
title: >-
  [论文解读] Normal-Abnormal Guided Generalist Anomaly Detection
description: >-
  [NeurIPS 2025][目标检测][通用异常检测] NAGL 框架首次在通用异常检测（GAD）中引入正常+异常混合参考样本，通过残差挖掘（RM）和异常特征学习（AFL）两个注意力模块，在残差空间中学习可迁移的异常模式，仅用 1 个异常样本即可在跨域场景中大幅超越仅使用正常参考的方法。 视觉异常检测在工业质检和医学诊断中…
tags:
  - "NeurIPS 2025"
  - "目标检测"
  - "通用异常检测"
  - "跨域迁移"
  - "残差学习"
  - "正常-异常参考"
  - "元学习"
---

# Normal-Abnormal Guided Generalist Anomaly Detection

**会议**: NeurIPS 2025  
**arXiv**: [2510.00495](https://arxiv.org/abs/2510.00495)  
**代码**: [GitHub](https://github.com/JasonKyng/NAGL)  
**领域**: LLM评测  
**关键词**: 通用异常检测, 跨域迁移, 残差学习, 正常-异常参考, 元学习

## 一句话总结

NAGL 框架首次在通用异常检测（GAD）中引入正常+异常混合参考样本，通过残差挖掘（RM）和异常特征学习（AFL）两个注意力模块，在残差空间中学习可迁移的异常模式，仅用 1 个异常样本即可在跨域场景中大幅超越仅使用正常参考的方法。

## 研究背景与动机

视觉异常检测在工业质检和医学诊断中至关重要。通用异常检测（GAD）旨在在原始域训练统一模型，然后直接迁移到新目标域进行异常检测，解决目标域数据稀缺和隐私限制的问题。

现有 GAD 方法（如 InCTRL、ResAD）存在关键局限：

**仅使用正常样本作为参考**：模型缺乏对异常特征的直接认知，判别能力有限

**忽视现实中可用的异常样本**：实际场景中通常能获得少量异常样本（如缺陷零件、已诊断病例），这些样本包含宝贵的异常特征信息但未被利用

直觉上，引入异常参考应该有帮助，但简单方法行不通：
- **KNN 方法**（远离正常 + 靠近异常 = 异常）：训练无关，缺乏适应性，且实验发现存在严重的**误激活问题**——正常区域被错误地标记为异常
- 现有 GAD 方法基于查询与正常参考的残差来保证迁移性，但无法直接扩展到正常-异常混合参考场景

核心挑战在于：如何有效利用异常参考样本的信息而不引入噪声？NAGL 的关键洞察是——在**残差空间**中操作可以同时保持跨域迁移性和异常判别力。

## 方法详解

### 整体框架

NAGL 的推理流程：给定查询图像和正常+异常参考集 →
1. 预训练 ViT 骨干提取特征
2. 查询与正常参考做最近邻搜索 → 正常引导分数图 $\mathcal{S}_n$
3. RM 模块从正常-异常参考残差中挖掘异常模式 → 残差代理
4. AFL 模块将残差代理映射到查询图像 → 异常代理 → 异常引导分数图 $\mathcal{S}_a$
5. 合并两个分数图 $\mathcal{S} = \mathcal{S}_n + \mathcal{S}_a$ 得到最终异常定位

### 关键设计

1. **正常引导异常分数（Normal-Guided Score）**：采用 PatchCore 式的最近邻搜索。对查询特征 $\mathcal{F}^q$ 中的每个 patch $f_i^q$，在正常参考特征 $\mathcal{F}^n$ 中找到最近邻 $f_*^n$，异常分数为 $\mathcal{S}_n^i = \mathbf{d}(f_i^q, f_*^n)$（余弦距离）。这提供了基础的异常定位能力，但仅依赖正常参考的判别力有限。

2. **残差挖掘模块（Residual Mining, RM）**：核心目标是从异常参考中提取可迁移的异常模式表示。首先计算异常参考与其最近正常参考之间的残差：
    $\text{Res}(\mathcal{F}^a, \mathcal{F}^n) = \mathcal{F}^a - \mathcal{F}_*^n$
   然后设计一个交叉注意力层，以可学习代理 $\mathcal{P} \in \mathbb{R}^{M \times C}$ 为 Query，异常特征 $\mathcal{F}^a$ 为 Key，残差为 Value：
    $\widetilde{\mathcal{P}} = \mathbf{SA}_1\left(\text{Softmax}\left(\frac{\mathbf{Q}_1 \mathbf{K}_1^T}{\sqrt{d}} + \mathcal{M}'\right) \mathbf{V}_1\right)$
   关键设计是注意力掩码 $\mathcal{M}' = \alpha(1 - \mathcal{M}^a)$（$\alpha$ 为大负值），确保注意力仅聚焦于异常区域的残差。输出的残差代理 $\widetilde{\mathcal{P}}$ 捕获了异常样本在残差空间中的变化模式——在残差空间操作保证了跨域迁移性，因为不同域的残差特征呈现相似分布。

3. **异常特征学习模块（Anomaly Feature Learning, AFL）**：将残差代理应用于查询图像以发现潜在异常。以残差代理 $\widetilde{\mathcal{P}}$ 为 Query，查询-正常残差 $\text{Res}(\mathcal{F}^q, \mathcal{F}^n)$ 为 Key，查询特征 $\mathcal{F}^q$ 为 Value：
    $\widehat{\mathcal{P}} = \mathbf{SA}_2\left(\text{Softmax}\left(\frac{\mathbf{Q}_2 \mathbf{K}_2^T}{\sqrt{d}}\right) \mathbf{V}_2\right)$
   核心思路是：通过比较**参考异常-正常残差**与**查询-正常残差**的相似性来发现查询中的异常区域。若查询的某区域残差模式与已知异常残差相似，对应的视觉特征就可能是异常。输出的异常代理 $\widehat{\mathcal{P}} \in \mathbb{R}^{M \times C}$ 包含查询图像中最具判别性的特征。最终异常引导分数为：
    $\mathcal{S}_a^i = \frac{1}{M} \sum_{m=1}^M 1 - \mathbf{d}(f_i^q, \widehat{\mathcal{P}}_m)$

### 损失函数 / 训练策略

在原始域采用元学习策略训练，每个 episode 包含正常-异常参考集和查询图像。损失函数组合分割和分类任务：
$$\mathcal{L} = \mathcal{L}_{cls} + \lambda \mathcal{L}_{seg}$$
其中：
- $\mathcal{L}_{seg} = \text{Focal}(\mathcal{S}, \mathcal{M}^q) + \text{Dice}(\mathcal{S}, \mathcal{M}^q)$：像素级异常定位
- $\mathcal{L}_{cls} = \text{BCE}(s, y^q)$：图像级异常分类，$s = \mathcal{T}_{0.01}(\mathcal{S})$（取分数图前 1% 均值）
- $\lambda = 1.0$

实现细节：ViT-S 骨干（21M 参数，冻结），仅训练注意力模块（总 24.4M 参数）。AdamW 优化器，初始学习率 $10^{-5}$，在 epoch 10 和 15 降低 10 倍。20 epoch 收敛，每 epoch 500 个 episode。输入分辨率 $448 \times 448$，可学习代理数 $M=25$。正常参考 $K_1 \in [1,2,4]$，异常参考 $K_2 = 1$。

## 实验关键数据

### 主实验

跨域评估：VisA 训练 → MVTecAD 测试，MVTecAD 训练 → VisA/BraTS 测试。

| 设置 | 方法 | MVTecAD Image AUROC | MVTecAD Pixel AUROC | VisA Image AUROC | VisA Pixel AUROC |
|------|------|-------------------|-------------------|-----------------|-----------------|
| $N^1$ | PatchCore | 83.4 | 92.0 | 79.9 | 95.4 |
| $N^1$ | WinCLIP | 93.1 | 95.2 | 83.8 | 96.4 |
| $N^1$ | ResAD | 84.8 | 93.4 | 80.9 | 95.9 |
| $N^1+A^1$ | **Ours** | **95.8** | **96.6** | **88.5** | **97.5** |
| $N^4$ | WinCLIP | 95.2 | 96.2 | 87.3 | 97.2 |
| $N^4+A^1$ | **Ours** | **97.1** | **97.0** | **91.2** | **97.8** |

跨域医学数据集（MVTecAD 训练 → BraTS 测试）：

| 设置 | 方法 | Image AUROC | Pixel AUROC | Mean |
|------|------|------------|------------|------|
| $N^2$ | ResAD | 66.2 | 91.5 | 78.9 |
| $N^2+A^1$ | **Ours** | **82.1** | **96.8** | **89.5** |
| $N^4$ | PatchCore | 71.2 | 95.9 | 83.6 |
| $N^4+A^1$ | **Ours** | **84.9** | **97.1** | **91.0** |

### 消融实验

| 配置 | MVTecAD Image | MVTecAD Pixel | VisA Image | VisA Pixel | 平均 |
|------|-------------|-------------|-----------|-----------|------|
| i: 仅正常 NN | 93.2 | 94.5 | 81.5 | 95.3 | 91.1 |
| ii: 仅异常 NN | 70.1 | 83.3 | 58.8 | 84.5 | 74.2 |
| iii: 正常+异常 NN（无 NAGL） | 90.7 | 92.1 | 77.2 | 93.5 | 88.4 |
| iv: 正常+异常+NAGL（完整） | **95.8** | **96.6** | **88.5** | **97.5** | **94.6** |

效率对比：

| 方法 | 参数量 (M) | 训练时间 (H) | 推理速度 (FPS) |
|------|----------|------------|-------------|
| InCTRL | 117.5 | 0.7 | 1.2 |
| ResAD | 59.2 | 20.6 | 7.8 |
| **Ours** | **24.4** | **0.3** | **17.1** |

### 关键发现

- **1 个异常样本带来巨大提升**：$N^1+A^1$ 设置超越了使用 $N^4$ 正常参考的最佳基线（MVTecAD: +0.6 Image AUROC, VisA: +1.2 Image AUROC），用更少参考获得更好效果
- **误激活问题验证**：简单融合正常+异常参考（设置 iii）反而比只用正常参考（设置 i）更差（91.1→88.4），证实了朴素方法引入误导性噪声
- **NAGL 有效解决误激活**：设置 iv 比设置 iii 提升 6.2 平均 AUROC，证明残差空间操作能有效过滤噪声
- **跨域泛化优秀**：在工业→医学跨域场景（BraTS）中，NAGL 以 $N^2+A^1$ 超越 ResAD $N^2$ 10.6 个百分点
- **效率卓越**：参数量仅 InCTRL 的 1/5，训练速度是 ResAD 的 69 倍，推理速度是 InCTRL 的 14 倍

## 亮点与洞察

- **任务定义创新**：首次将正常-异常混合参考引入 GAD，更贴近实际应用场景
- **残差空间是关键**：直接在视觉空间融合异常参考会引入误激活，但在残差空间操作天然具有跨域迁移性，因为残差特征的分布在不同领域间更稳定
- **设计极简高效**：冻结骨干 + 仅训练 2 个注意力模块，24.4M 参数实现 SOTA
- RM→AFL 的两阶段设计精巧：先在残差空间提取异常模式原型，再将原型映射回视觉空间定位具体异常

## 局限与展望

- 异常参考限制为 $K_2=1$，更多异常参考是否能更进一步提升有待探索
- 假设每种异常类型有 1 个参考样本，若异常类型未覆盖则效果未知
- 可学习代理数 $M=25$ 的选择缺乏自适应机制
- 未来可探索语言描述作为正常/异常参考的补充
- 可扩展到 3D 异常检测和视频异常检测场景

## 相关工作与启发

- 与 PatchCore 的对比说明：好的表示学习 + 异常指导 > 大量正常参考
- 残差特征的跨域一致性值得在其他迁移学习任务中进一步验证
- RM-AFL 架构可推广到其他 few-shot 检测任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次在 GAD 中利用异常参考，任务和方法都有创新
- **实验充分度**: ⭐⭐⭐⭐⭐ 工业+医学跨域评估，消融详尽，效率对比全面
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，图表配合好，RM-AFL 的解释到位
- **价值**: ⭐⭐⭐⭐ 实用导向强，效率优势明显，适合实际部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Reimagining Anomalies: What if Anomalies Were Normal?](../../AAAI2026/object_detection/reimagining_anomalies_what_if_anomalies_were_normal.md)
- [\[NeurIPS 2025\] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)
- [\[AAAI 2026\] PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixing of Experts](../../AAAI2026/object_detection/promptmoe_generalizable_zero-shot_anomaly_detection_via_visually-guided_prompt_m.md)
- [\[ICLR 2026\] Complexity- and Statistics-Guided Anomaly Detection in Time Series Foundation Models](../../ICLR2026/object_detection/complexity-_and_statistics-guided_anomaly_detection_in_time_series_foundation_mo.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)

</div>

<!-- RELATED:END -->
