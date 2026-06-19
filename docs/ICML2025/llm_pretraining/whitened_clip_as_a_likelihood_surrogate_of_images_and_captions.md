---
title: >-
  [论文解读] Whitened CLIP as a Likelihood Surrogate of Images and Captions
description: >-
  [ICML 2025][预训练][CLIP] 提出 Whitened CLIP (W-CLIP)，通过对 CLIP 嵌入做可逆 PCA 白化变换使其近似 i.i.d. 标准正态分布，从而用欧氏范数的平方直接估计图像和文本的对数似然，在伪影检测、域偏移分析和全圆 SLERP 图像操控中展现了有效性。 图像似然估计的困难：计算图…
tags:
  - "ICML 2025"
  - "预训练"
  - "CLIP"
  - "白化变换"
  - "似然代理"
  - "各向同性"
  - "OOD检测"
---

# Whitened CLIP as a Likelihood Surrogate of Images and Captions

**会议**: ICML 2025  
**arXiv**: [2505.06934](https://arxiv.org/abs/2505.06934)  
**代码**: 有（论文中提供链接）  
**领域**: LLM预训练  
**关键词**: CLIP, 白化变换, 似然代理, 各向同性, OOD检测

## 一句话总结

提出 Whitened CLIP (W-CLIP)，通过对 CLIP 嵌入做可逆 PCA 白化变换使其近似 i.i.d. 标准正态分布，从而用欧氏范数的平方直接估计图像和文本的对数似然，在伪影检测、域偏移分析和全圆 SLERP 图像操控中展现了有效性。

## 研究背景与动机

**图像似然估计的困难**：计算图像的似然 $P(X)$ 是计算机视觉的基础问题，但现有方法（如扩散模型）只能近似得分函数 $\nabla_x \log P(X)$，GAN/VAE/EBM 等生成模型也仅隐式估计分布，无法直接获得 $P(X)$。

**CLIP 空间的结构性问题**：CLIP 将图像和文本嵌入共享空间，广泛用于图文匹配。但其嵌入空间存在两个已知缺陷——**Narrow Cone Effect**（嵌入集中在狭窄角度范围）和 **Modality Gap**（图像与文本嵌入分布不相交），限制了其作为概率估计器的使用。

**核心 idea**：对 CLIP 嵌入应用白化变换（零均值 + 单位协方差），将原始椭球形空间转化为超球面。在标准正态假设下，对数似然可直接由白化空间中的欧氏范数平方估计：$\ell(x) = -\frac{1}{2}(d\log(2\pi) + \|x\|^2)$。该变换完全无需训练，仅依赖预计算的白化矩阵。这是首次为图像提供基于高层语义的直接概率计算方法。

## 方法详解

### 整体框架

W-CLIP 的 pipeline 极简：(1) 在代表性数据集（如 MS-COCO 验证集 5000 张图）上计算 CLIP 嵌入的均值 $\mu$ 和协方差矩阵 $\Sigma$；(2) 通过 PCA 分解 $\Sigma = V\Lambda V^\top$ 得到白化矩阵 $W = \Lambda^{-1/2}V^\top$；(3) 对任意新样本的 CLIP 嵌入 $x$，计算白化嵌入 $y = W(x - \mu)$；(4) 利用 $\ell(x) = -\frac{1}{2}(d\log(2\pi) + \|y\|^2)$ 估计似然。图像和文本模态分别独立白化处理，白化矩阵预计算一次即可复用。使用 CLIP ViT-L/14 模型（$d=768$）。

### 关键设计

1. **PCA 白化变换**:
    - 功能：将 CLIP 嵌入从各向异性椭球分布转化为各向同性超球分布
    - 核心思路：给定协方差矩阵 $\Sigma = V\Lambda V^\top$，白化矩阵 $W = \Lambda^{-1/2}V^\top$，白化后 $y = W\hat{x}$ 满足 $\mu_Y = 0, \Sigma_Y = I$。变换可逆，原始空间可通过 $x = W^{-1}y + \mu$ 恢复。Diagonal Score（对角度量）验证白化后协方差近乎完美对角化，在正态假设下不相关等价于独立
    - 设计动机：白化是唯一同时实现零均值、单位方差、去相关的线性变换；纯数据驱动无超参数，计算开销极低，可逆性保证 CLIP 原有功能不受影响

2. **范数-似然映射与正态性验证**:
    - 功能：统计验证白化嵌入近似标准正态分布，建立范数与似然的精确对应
    - 核心思路：使用 Anderson-Darling（侧重尾部偏差）和 D'Agostino-Pearson（结合偏度和峰度）两种检验。图像嵌入 >98% 特征通过正态检验，文本 >90%。范数服从 chi 分布 $\chi_d$，期望 $\mathbb{E}[S] = \sqrt{2}\frac{\Gamma(\frac{d+1}{2})}{\Gamma(\frac{d}{2})} \approx \sqrt{d - 1/2}$，$d=768$ 时理论值 27.7，实测图像嵌入均值 27.43（偏差仅 0.98%）
    - 设计动机：只有验证了正态假设的有效性，范数才能作为可靠的似然代理。经验与理论值的高度吻合确认了方法的统计基础

3. **全圆球面线性插值 (Full-Circle SLERP)**:
    - 功能：将标准 SLERP 从 $t \in [0,1]$ 扩展到全 $360°$，实现图像间的插值与外推
    - 核心思路：设插值角度 $\omega$，令 $t = \omega/\theta$，代入 SLERP 公式 $\text{SLERP}(t; E_1, E_2) = \frac{\sin((1-t)\theta)}{\sin\theta}E_1 + \frac{\sin(t\theta)}{\sin\theta}E_2$。在原始 CLIP 空间中 $180°$ 处生成噪声，而在 W-CLIP 中所有角度均生成自然图像。$180°$ 处的"对立嵌入"仅由源图决定，是源图的固定对称对应物
    - 设计动机：CLIP 的 Narrow Cone 效应导致嵌入偏离超球面，SLERP 超出插值区间时失效。白化使嵌入均匀分布在超球面上，所有方向均在分布内

### 损失函数 / 训练策略

W-CLIP 完全免训练。白化矩阵 $W$ 和均值 $\mu$ 在代表性数据集上一次性预计算。跨数据集泛化验证表明 MS-COCO 和 Flickr8k 交换白化/测试角色后似然相关性仍达 0.69-0.88。

## 实验关键数据

### 主实验：正态分布检验

| 检验方法 | 模态 | 平均分数 | 通过比例 | 阈值 |
|---------|------|---------|---------|------|
| Anderson-Darling | 图像 | 0.489 | 98.3% | < 0.752 |
| Anderson-Darling | 文本 | 0.593 | 90.1% | < 0.752 |
| D'Agostino-Pearson | 图像 | 0.362 | 99.3% | > 0.05 |
| D'Agostino-Pearson | 文本 | 0.257 | 99.2% | > 0.05 |

### 经验值与理论值对比（$d=768$）

| 模态 | 均值 (经验/理论) | 标准差 (经验/理论) |
|------|----------------|-------------------|
| 图像 | 27.43 / 27.7 (偏差 0.98%) | 3.94 / 3.96 (偏差 0.55%) |
| 文本 | 28.49 / 27.7 (偏差 2.85%) | 5.72 / 6.60 (偏差 13.24%) |

### Full-Circle SLERP 对立图像质量

| 方法 | Total Variation | Entropy | 饱和像素占比 |
|------|----------------|---------|-------------|
| MS-COCO 真实图像 | 222.3 | 7.3 | 4.2% |
| CLIP 对立图像 | 156.7 | 4.8 | 55.5% |
| W-CLIP 对立图像 | 215.9 | 7.2 | 6.4% |

### 似然分离能力对比 (AUC)

| 模型 | 类型分离 (Caption vs 通用文本) | 去名词分离 |
|------|---------------------------|-----------|
| GPT-2 (LLM) | 0.80 | 0.43 |
| OPT (LLM) | 0.80 | 0.58 |
| NEO (LLM) | 0.77 | 0.58 |
| BLIP (VLM) | 0.92 | 0.66 |
| GIT (VLM) | 0.97 | 0.69 |
| **W-CLIP (本文)** | **0.999** | **0.94** |

### 消融实验：跨数据集泛化

| 测试集 | 白化数据集 | Avg. AD | 似然相关性 (图像/文本) |
|--------|----------|---------|---------------------|
| COCO | COCO | 0.489 | 基线 |
| COCO | Flickr8k | 0.466 | 0.69 / 0.74 |
| Flickr8k | COCO | 0.641 | 0.77 / 0.88 |
| Flickr8k | Flickr8k | 0.522 | 基线 |

### 关键发现

- W-CLIP 范数能有效区分真实与含伪影 AI 生成图像（SynArtifact 数据集中所有生成图像似然均低于真实对应物）
- ImageNet-C 噪声级别与 W-CLIP 范数呈单调正相关（噪声越强似然越低），ImageNet-R 各风格偏移量有序：涂鸦最接近真实、电子游戏渲染偏移最大
- W-CLIP 对语法错误（去名词）极其敏感（AUC=0.94），远超所有 LLM（≤0.58）和 VLM（≤0.69）
- 文本复杂度与似然负相关：移除具体词（人名/地点）→似然升高，添加具体词→似然降低
- 生成模型（UnCLIP）存在系统性似然偏差，迭代生成中嵌入范数逐步增大导致退化，归一化到 $\sqrt{d}$ 可缓解

## 亮点与洞察

- **零成本后处理**：白化矩阵预计算一次，推理时仅需一次矩阵乘法，内存和计算需求极低
- **与语言模型的互补性**：W-CLIP 对文本长度不敏感但对语义变化（语法、caption vs 通用文本）高度敏感，而语言模型恰好相反
- **均匀性提升**：白化后余弦相似度集中在零附近（标准差极小），解决了原始 CLIP 中相似度集中在 0.5 附近的问题
- **可逆性保证兼容性**：所有 CLIP 下游应用可与 W-CLIP 无缝集成

## 局限与展望

- 文本模态正态近似不如图像精确（标准差偏差 13.24%），文本侧似然估计精度受限
- 白化依赖代表性数据集计算协方差，对域偏移敏感
- 仅验证了 CLIP ViT-L/14 模型，其他架构（如 ViT-B/32、OpenCLIP）的适用性未系统评估
- 与语言模型似然的相关仅 0.33-0.48，说明 W-CLIP 捕捉的是不同维度的"似然"
- 生成图像检测仅做初步定性分析，缺乏大规模定量评估和专用检测器对比

## 相关工作与启发

- Liang et al. (2022) 发现 Modality Gap，Schrodi et al. (2024) 发现 Narrow Cone Effect——白化同时解决了这两个问题
- Levi & Gilboa (2025) 的双椭球体几何分析与本文从概率视角互补
- UnCLIP 迭代实验揭示生成模型的系统性似然偏差，启发将 W-CLIP 用于生成质量监控
- 可扩展方向：将 W-CLIP 似然作为图像生成的质量评估指标、OOD 检测基线、或条件生成中的采样引导信号

## 评分

- 新颖性: ⭐⭐⭐⭐ 将白化这一经典操作与 CLIP 概率解释结合，视角独特，理论自洽
- 实验充分度: ⭐⭐⭐⭐ 涵盖正态性检验、跨域泛化消融、多应用场景、与多种 LLM/VLM 对比
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，图表丰富，结构清晰
- 价值: ⭐⭐⭐ 方法极简实用，但应用深度有限，更接近初步探索而非成熟工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] The Double-Ellipsoid Geometry of CLIP](the_double-ellipsoid_geometry_of_clip.md)
- [\[CVPR 2026\] Reconstructing CLIP for Open-Vocabulary Dense Perception](../../CVPR2026/llm_pretraining/reconstructing_clip_for_open-vocabulary_dense_perception.md)
- [\[CVPR 2025\] Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection](../../CVPR2025/llm_pretraining/seeing_what_matters_empowering_clip_with_patch_generation-to-selection.md)
- [\[ICLR 2026\] CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images](../../ICLR2026/llm_pretraining/chammi-75_pre-training_multi-channel_models_with_heterogeneous_microscopy_images.md)
- [\[AAAI 2026\] Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment](../../AAAI2026/llm_pretraining/beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass.md)

</div>

<!-- RELATED:END -->
