---
title: >-
  [论文解读] AugGen: Synthetic Augmentation using Diffusion Models Can Improve Recognition
description: >-
  [NeurIPS 2025][图像生成][合成数据增强] 提出 AugGen，一种自包含的合成数据增强方法：在目标数据集上训练类条件扩散模型，通过混合不同类别的条件向量生成新的"混合类"样本，增强判别模型训练，在人脸识别基准上实现 1-12% 的性能提升，无需任何外部数据或辅助模型。 机器学习对大规模数据集的依赖带来了隐私和…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "合成数据增强"
  - "类条件扩散模型"
  - "人脸识别"
  - "类别混合"
  - "自包含框架"
---

# AugGen: Synthetic Augmentation using Diffusion Models Can Improve Recognition

**会议**: NeurIPS 2025  
**arXiv**: [2503.11544](https://arxiv.org/abs/2503.11544)  
**代码**: [https://parsa-ra.github.io/auggen/](https://parsa-ra.github.io/auggen/) (项目页面)  
**领域**: 图像生成 / 合成数据增强  
**关键词**: 合成数据增强, 类条件扩散模型, 人脸识别, 类别混合, 自包含框架

## 一句话总结

提出 AugGen，一种自包含的合成数据增强方法：在目标数据集上训练类条件扩散模型，通过混合不同类别的条件向量生成新的"混合类"样本，增强判别模型训练，在人脸识别基准上实现 1-12% 的性能提升，无需任何外部数据或辅助模型。

## 研究背景与动机

机器学习对大规模数据集的依赖带来了隐私和伦理挑战，尤其在人脸识别等敏感领域。合成数据生成是替代方案，但现有方法（如 DCFace、IDiff-Face）严重依赖外部数据集和预训练的大模型，这一方面增加了复杂度和资源需求，另一方面使用的外部数据本身也可能存在隐私问题（如 MS1Mv2）。

AugGen 的出发点是：能否仅用原始训练数据本身，通过智能采样策略，让生成模型成为判别模型的"增强器"而非"替代品"？这在数据有限的场景下特别有价值——用有限的真实数据弥合小规模和大规模训练之间的性能差距。

两个关键动机：(1) 扩散模型生成的合成数据集经常泄漏训练数据，单纯替代真实数据无明确优势；(2) 负责任的 FR 数据集稀缺难收集，应将生成模型用作增强工具而非数据替代品。

## 方法详解

### 整体框架

AugGen 流程分四步：(a) 用标注数据集 $D_{\text{orig}}$ 同时训练一个类条件生成模型 $G$ 和一个判别模型 $M_{\text{orig}}$；(b) 用原始条件向量再生成样本 $D_{\text{repro}}$，模拟原始数据；(c) 通过网格搜索找到最优混合权重 $(\alpha^*, \beta^*)$，用于生成新类；(d) 生成增强数据 $D_{\text{aug}}$，与原始数据混合后重新训练判别模型。

### 关键设计

1. **类条件混合采样策略**: 给定两个类 $i, j$ 的独热条件向量 $\mathbf{c}^i, \mathbf{c}^j$，构造新条件 $\mathbf{c}^* = \alpha \mathbf{c}^i + \beta \mathbf{c}^j$。使用 $\mathbf{c}^*$ 从训练好的扩散模型中采样，生成融合两个"源类"特征的新样本。关键在于选择合适的 $\alpha, \beta$：太小则新样本与源类无差异，太大则脱离有效分布。设计动机是生成"硬负样本"——在特征空间中靠近源类但可区分的新类，迫使判别模型学到更紧凑的类内表示和更大的类间分离。

2. **基于相似度的混合权重搜索**: 设计组合度量 $m^{\text{total}} = m_s^{\text{total}} + m_d^{\text{total}}$，其中 $m_d$ 衡量新类与源类的特征差异（越大越好），$m_s$ 衡量同一 $\mathbf{c}^*$ 多次采样结果的类内一致性（越高越好）。通过网格搜索 $\alpha, \beta \in [0.1, 1.1]$，最大化该度量。CASIA-WebFace 最优为 $(0.7, 0.7)$，WebFace160K 为 $(0.8, 0.8)$。搜索效率高——单张 RTX 3090 Ti 不到 2 GPU-day。

3. **自包含设计原则**: 整个流程仅使用目标数据集，不引入任何外部数据、预训练生成器或辅助分类器。生成模型使用 EDM/EDM2 框架从零训练，判别模型使用 IR50/IR101 + AdaFace，两者共享同一训练数据。这保证了隐私合规性和方法的可复现性。

### 损失函数 / 训练策略

- 生成模型采用 Karras 等人的 EDM/EDM2 扩散目标（去噪 score matching），支持潜空间和像素空间两种变体
- 判别模型使用 AdaFace margin loss（ArcFace 的改进版本）
- 标准数据增强（光照变换、裁剪、低分辨率模拟）应用于所有模型
- 多次随机种子训练取均值和标准差，确保结果可靠

## 实验关键数据

### 主实验

以 CASIA-WebFace 为基础数据集，IR50 骨干，评估 8 个 FR 基准：

| 方法 | 辅助数据 | IJB-B 1e-6 | IJB-C 1e-6 | TinyFace R1 | Avg-H |
|------|---------|-----------|-----------|------------|-------|
| CASIA-WebFace（真实） | 无 | 1.02±0.26 | 0.73±0.19 | 58.12±0.31 | 94.21 |
| CASIA-WebFace IR101† | 无 | 0.74±0.31 | 0.38±0.13 | 59.64±0.49 | 94.84 |
| DCFace（合成） | 有 | 22.48±4.35 | 35.27±10.78 | 45.94 | 91.56 |
| IDiff-Face（合成） | 有 | 26.84±2.03 | 41.75±1.04 | 45.98 | 84.68 |
| **AugGen $D_{\text{aug}}$（混合）** | **无** | **2.61±0.91** | **4.36±1.41** | 59.82 | 94.66 |

WebFace160K 上，AugGen 在 IJB-B/C 的严格阈值下超过更大网络（IR101）的纯真实数据训练：

| 方法 | IJB-B 1e-6 | IJB-B 1e-5 | IJB-C 1e-6 | IJB-C 1e-5 |
|------|-----------|-----------|-----------|-----------|
| WebFace160K IR50 | 32.13±1.87 | 72.18±0.18 | 70.37±0.75 | 78.81±0.32 |
| WebFace160K IR101† | 34.84±0.49 | 74.10±0.24 | 72.56±0.02 | 81.26±0.14 |
| **AugGen + WebFace160K** | **36.62±0.77** | **78.32±0.33** | **78.58±0.15** | **85.02±0.15** |

### 消融实验

等效真实数据实验：添加 60 万 AugGen 样本的性能提升约等于添加 11 万真实样本（1.69x 真实数据比例），展示了合成增强的数据效率。

### 关键发现

- AugGen 是唯一在所有基准上一致超过基线的方法——DCFace 和 IDiff-Face 在混合训练时对 IJB-B/C 低阈值反而不如纯真实数据
- 合成增强的效果常超过架构升级（IR50 + AugGen > IR101 纯真实数据）
- 现有生成质量指标（FD、KD）与下游判别性能相关性差，强调需要更好的代理指标
- AugGen 生成的"混合类"在特征空间中表现为靠近源类但可区分的新身份，增加了训练的区分性难度

## 亮点与洞察

- **反直觉价值**: 与追求大规模预训练+微调的主流趋势相反，证明仅用有限真实数据训练的小型生成模型也能显著提升下游判别任务
- **隐私合规**: 完全自包含设计避免了外部数据引入的隐私问题，在责任 AI 的方向上具有重要意义
- 混合条件向量的思想类似于 Mixup 在特征空间的扩展，但在生成模型的条件空间中操作，生成的是语义上合理的新样本而非简单插值
- 合成增强 > 架构升级这一发现，凸显了数据质量/多样性对判别任务的核心作用

## 局限与展望

- 仅在人脸识别任务上验证，未扩展到通用图像分类或细粒度识别
- 混合权重搜索虽高效但仍需要一定计算资源和已训练的判别模型
- 仅探索了两类混合（$\alpha \mathbf{c}^i + \beta \mathbf{c}^j$），更多类的混合或连续插值可能带来更多增益
- 生成质量受限于原始数据集的规模和多样性——数据太少时扩散模型本身质量难以保证
- 未探索文本条件扩散模型（如 Stable Diffusion）在该框架中的适用性

## 相关工作与启发

- **DCFace / IDiff-Face**: 依赖外部数据和模型，AugGen 证明这种依赖并非必要
- **Mixup**: AugGen 的类混合策略是 Mixup 在生成模型条件空间的自然推广
- **DigiFace1M**: 3D 渲染方法，AugGen 证明学习式生成在无3D先验的情况下也可行
- 核心启发：生成模型不应只与真实数据竞争，应作为增强工具与真实数据协同

## 评分

⭐⭐⭐⭐ — 思路简洁实用（自包含+无外部依赖），实验在8个FR基准上充分验证。概念虽简单但有效，是合成数据增强的范式转变。主要局限是应用范围暂限于人脸识别。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Synthetic Perception: Can Generated Images Unlock Latent Visual Prior for Text-Centric Reasoning?](../../ICML2025/image_generation/synthetic_perception_can_generated_images_unlock_latent_visual_prior_for_text-ce.md)
- [\[NeurIPS 2025\] UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation](utilgen_utility-centric_generative_data_augmentation_with_dual-level_task_adapta.md)
- [\[NeurIPS 2025\] Scaling Can Lead to Compositional Generalization](scaling_can_lead_to_compositional_generalization.md)
- [\[CVPR 2025\] Can Generative Video Models Help Pose Estimation?](../../CVPR2025/image_generation/can_generative_video_models_help_pose_estimation.md)
- [\[ICCV 2025\] Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!](../../ICCV2025/image_generation/attention_to_neural_plagiarism_diffusion_models_can_plagiarize_your_copyrighted_.md)

</div>

<!-- RELATED:END -->
