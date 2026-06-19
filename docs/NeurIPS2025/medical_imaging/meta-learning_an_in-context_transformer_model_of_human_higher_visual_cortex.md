---
title: >-
  [论文解读] Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex
description: >-
  [NeurIPS 2025][医学图像][fMRI编码模型] 提出BraInCoRL（Brain In-Context Representation Learning），一种基于Transformer的元学习框架，通过上下文学习（in-context learning）从少量刺激-响应样本直接预测新被试的体素级神经响应，无需微调即可适应新被试和新刺激，仅用100张图片就接近在9000张图片上完整训练的参考模型的性能。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "fMRI编码模型"
  - "元学习"
  - "上下文学习"
  - "高级视觉皮层"
  - "超网络"
---

# Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex

**会议**: NeurIPS 2025  
**arXiv**: [2505.15813](https://arxiv.org/abs/2505.15813)  
**代码**: [GitHub](https://github.com/leomqyu/BraInCoRL)  
**领域**: 3D视觉  
**关键词**: fMRI编码模型, 元学习, 上下文学习, 高级视觉皮层, 超网络

## 一句话总结

提出BraInCoRL（Brain In-Context Representation Learning），一种基于Transformer的元学习框架，通过上下文学习（in-context learning）从少量刺激-响应样本直接预测新被试的体素级神经响应，无需微调即可适应新被试和新刺激，仅用100张图片就接近在9000张图片上完整训练的参考模型的性能。

## 研究背景与动机

理解高级视觉皮层的功能表征是计算神经科学的核心问题。现有的视觉皮层编码模型通常将预训练深度特征线性回归到被试特定的体素响应上，但面临关键瓶颈：

**数据获取瓶颈**：为每个被试拟合编码模型需要数小时昂贵的fMRI扫描（如NSD数据集需要~10000张图片），在临床场景中收集大量数据往往不可行

**个体差异**：虽然视觉皮层在粗尺度上的功能组织在个体间一致（如人脸选择性区域FFA），但精细水平上存在显著的解剖位置、空间范围和响应特征差异

**跨被试泛化困难**：现有方法（如Adeli等人的auto-decoder Transformer）虽然使用多被试数据，但仍需要对新被试进行微调

核心思想：受语言模型上下文学习（ICL）能力的启发，将体素编码视为函数推理问题——给定新个体的少量刺激-响应对，直接构建体素编码模型而无需任何训练。

## 方法详解

### 整体框架

BraInCoRL将每个体素的响应函数视为一个元训练任务。训练时，从多个被试中随机采样体素，通过Transformer学习跨被试共享的功能原则。推理时，给定新被试的少量上下文样本直接生成体素编码器参数（超网络范式）。

三阶段训练流程：(1) 预训练——用合成体素数据训练，固定上下文500张；(2) 上下文扩展——上下文大小从Uniform(30, 500)随机采样，获得长度鲁棒性；(3) 监督微调——在真实fMRI数据上训练。

### 关键设计

1. **体素级元学习**：不同于将整个视觉皮层作为建模单元的传统方法，BraInCoRL以单个体素为基本单位。对support set中每个体素定义上下文token $c_i = [x_i; \beta_i]$（图像嵌入与体素响应拼接）。Transformer $T$ 直接输出编码器参数 $\omega$：

$$\omega = T_\theta(c_1, c_2, \ldots, c_p), \quad \hat{\beta} = f_\omega(\mathcal{I})$$

优化目标：$\theta^* = \arg\min_\theta \mathbb{E}_{(I_q, \beta_q)} \|f_\omega(\mathcal{I}) - \beta_{\text{True}}\|_2^2$

设计动机：体素级粒度自然处理了跨被试体素数量不同的问题，且无需假设跨被试刺激重叠。

2. **测试时上下文缩放（Logit Scaling）**：为处理推理时不同长度的上下文，采用对数缩放注意力：

$$\alpha_{\text{scaled}} = \frac{\log(l) \cdot q \cdot k}{\sqrt{d_k}}$$

其中 $l$ 是上下文长度。配合训练时随机采样的上下文大小，实现对任意上下文长度的鲁棒推理。

3. **超网络生成线性编码器**：遵循神经科学中线性编码模型的传统，最终的体素响应预测为简单线性映射：$\hat{\beta} = f(\phi(\mathcal{I}); \omega) = \text{matmul}(x, \omega)$。Transformer生成的 $\omega$ 即为线性权重向量，保持了可解释性的同时获得了元学习的数据效率优势。

4. **网络架构细节**：20层自注意力编码器（SwiGLU激活 + pre-normalization），10个注意力头。CLIP版本(E=512)约97.2M参数，DINO (E=768)约112M，SIGLIP (E=1152)约130M。[CLS] token通过MLP输出最终超权重。

### 损失函数 / 训练策略

- MSE损失 + AdamW优化器（初始学习率 $10^{-3}$，衰减至 $10^{-5}$）
- ReduceLROnPlateau调度器（factor 0.1, patience 5, cooldown 2）
- 预训练阶段使用合成data-by-synthesis方法（合成权重 + 正态噪声）
- 微调阶段严格留出测试被试数据
- 批大小80，早停patience 5个epoch

## 实验关键数据

### 主实验（NSD数据集，5个类别选择性区域的explained variance）

| 方法 | Faces (S1/S2) | Places (S1/S2) | Bodies (S1/S2) | Words (S1/S2) | Food (S1/S2) | Mean (S1/S2) |
|------|-------------|---------------|---------------|--------------|-------------|-------------|
| Fully Trained (9000张) | 0.19/0.16 | 0.20/0.27 | 0.28/0.24 | 0.11/0.11 | 0.16/0.17 | 0.18/0.19 |
| Ridge-100 | 0.10/0.07 | 0.08/0.14 | 0.16/0.12 | 0.02/0.03 | 0.05/0.07 | 0.07/0.08 |
| Ridge-300 | 0.13/0.10 | 0.13/0.20 | 0.22/0.16 | 0.06/0.06 | 0.10/0.11 | 0.11/0.12 |
| FsAverage map | 0.13/0.06 | 0.11/0.19 | 0.09/0.08 | 0.06/0.03 | 0.14/0.18 | 0.08/0.06 |
| **BraInCoRL-100** | **0.16/0.13** | **0.16/0.23** | **0.25/0.21** | **0.07/0.08** | **0.12/0.13** | **0.13/0.15** |

### 消融 / 跨数据集实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| BraInCoRL（保留测试图像） | Mean EV ≈ 0.15 | 默认设置 |
| BraInCoRL（不保留测试图像） | 性能略高 | 无过拟合风险 |
| 仅预训练（无fMRI微调） | 性能下降明显 | 真实fMRI数据微调重要 |
| BOLD5000跨数据集泛化 | 优于Ridge回归 | 不同扫描仪(3T vs 7T)、不同参数 |
| 文本prompt分类准确率 | Bodies 63%/54%，Places 81%/88% | 语言驱动的皮层功能映射可行 |

### 关键发现

- **极高的数据效率**：仅100张上下文图片就接近在9000张上完整训练的参考模型性能（约72-89%），比Ridge-100提升约87%
- **强test-time scaling行为**：随着上下文增加从0到1000张，性能稳步提升并接近完整训练的参考模型
- **跨数据集泛化**：模型在NSD(7T)上训练后可直接泛化到BOLD5000(3T)，不同被试、不同刺激呈现时间、不同试验结构
- 对BraInCoRL预测权重做UMAP降维，揭示了与已知视觉区域（EBA、FFA、RSC、OPA、PPA等）一致的语义聚类
- 反直觉发现：过于特异性的上下文支持集反而降低性能，随机采样的多样化图片效果更好

## 亮点与洞察

- **范式转变**：从"为每个被试收集大量数据 → 拟合线性模型"转变为"跨被试元学习 → 少样本上下文适应"，有望大幅降低神经影像研究的数据需求
- **语言驱动的皮层映射**：CLIP特征 + BraInCoRL实现了从自然语言查询到体素选择性的零样本映射，这对理解视觉皮层的语义组织非常强大
- **三阶段训练策略**的巧妙设计：先在合成数据上学习元学习能力，再适应真实数据，有效解决了大规模fMRI数据稀缺的问题

## 局限与展望

- 目前仅处理静态自然图像，扩展到动态刺激（视频）需要重新设计编码器骨干
- 训练主要依赖NSD——当前最大的7T fMRI数据集，数据集多样性仍可能不足
- 体素级建模不考虑体素间的空间关系，未利用皮层的拓扑结构信息
- Word-selective区域的预测准确率相对较低，可能与该区域更强的个体差异有关

## 相关工作与启发

- 与大语言模型的上下文学习能力有深层联系——"Transformer隐式学习了元学习算法"这一假说的神经科学实证
- 超网络（HyperNetwork）范式在元学习中的应用：输出函数参数而非预测值
- 对脑机接口（BCI）和神经假体的个性化适配具有潜在应用价值

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ ICL + 元学习 + 超网络三者融合应用于神经科学，极具创新性
- 实验充分度: ⭐⭐⭐⭐⭐ NSD + BOLD5000双数据集验证，多骨干(CLIP/DINO/SIGLIP)，跨域泛化和语义分析均充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，可视化丰富
- 价值: ⭐⭐⭐⭐⭐ 解决了神经影像学中的核心数据效率瓶颈，跨领域启发性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] NEURONS: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction](../../ICCV2025/medical_imaging/neurons_emulating_the_human_visual_cortex_improves_fidelity_and_interpretability.md)
- [\[ICLR 2026\] LaVCa: LLM-assisted Visual Cortex Captioning](../../ICLR2026/medical_imaging/lavca_llm-assisted_visual_cortex_captioning.md)
- [\[NeurIPS 2025\] SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning](smmile_an_expert-driven_benchmark_for_multimodal_medical_in-context_learning.md)
- [\[NeurIPS 2025\] SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning](synbrain_enhancing_visual-to-fmri_synthesis_via_probabilistic_representation_lea.md)
- [\[CVPR 2025\] Show and Segment: Universal Medical Image Segmentation via In-Context Learning](../../CVPR2025/medical_imaging/show_and_segment_universal_medical_image_segmentation_via_in-context_learning.md)

</div>

<!-- RELATED:END -->
