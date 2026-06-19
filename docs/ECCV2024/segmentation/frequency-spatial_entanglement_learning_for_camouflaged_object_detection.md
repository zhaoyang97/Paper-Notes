---
title: >-
  [论文解读] Frequency-Spatial Entanglement Learning for Camouflaged Object Detection
description: >-
  [ECCV 2024][语义分割][伪装目标检测] 提出频率-空间纠缠学习（FSEL）框架，通过在频率域和空间域之间进行纠缠学习（entanglement learning），利用全局频率特征弥补空间特征的局部性和敏感性限制，在三个COD基准上超越21个SOTA方法。 伪装目标检测（COD）的核心挑战在于目标与背景在空间域中…
tags:
  - "ECCV 2024"
  - "语义分割"
  - "伪装目标检测"
  - "频率域学习"
  - "空间-频率纠缠"
  - "Transformer"
  - "傅里叶变换"
---

# Frequency-Spatial Entanglement Learning for Camouflaged Object Detection

**会议**: ECCV 2024  
**arXiv**: [2409.01686](https://arxiv.org/abs/2409.01686)  
**代码**: [有](https://github.com/CSYSI/FSEL)  
**领域**: 图像分割  
**关键词**: 伪装目标检测, 频率域学习, 空间-频率纠缠, Transformer, 傅里叶变换

## 一句话总结

提出频率-空间纠缠学习（FSEL）框架，通过在频率域和空间域之间进行纠缠学习（entanglement learning），利用全局频率特征弥补空间特征的局部性和敏感性限制，在三个COD基准上超越21个SOTA方法。

## 研究背景与动机

伪装目标检测（COD）的核心挑战在于目标与背景在空间域中的高度相似性，使得识别极其困难。现有方法存在的主要问题：

**空间域特征的固有缺陷**：现有COD方法主要依赖单一空间特征，这些特征基于像素级信息，关注局部强度和空间位置，具有**局部性**和**敏感性**——像素仅与周围像素相关，难以区分伪装目标与背景的微妙差异。当面对复杂背景时，空间特征容易受到干扰。

**现有频率方法的不足**：一些方法开始引入频率线索，但分为两类缺陷：
   - **第一类**（如FDNet、EVP）：直接对输入图像做频率变换提取特征，但**伪装图像含大量背景噪声**，获得的频率特征不可靠，与空间特征聚合时会引入不必要的背景噪声
   - **第二类**（如FPNet、FEDER）：在编码器初始特征上操作，但仅关注高频和低频信息，**忽略了中间频率段包含的丰富信息**，遗漏频率域中的重要信息

**频率特征的全局优势**：通过傅里叶变换生成的频率特征具有全局特性，能捕获整幅图像的频率分布，有助于突破空间特征的局部性瓶颈。

基于上述分析，作者提出核心想法：不应该简单地将频率和空间特征拼接，而应该让两个域的特征进行**纠缠学习**（类似量子纠缠的隐喻），使全局频率特征和局部空间特征相互学习和优化，形成更强大的综合表示。

## 方法详解

### 整体框架

FSEL模型包含三个核心组件：
1. **Joint Domain Perception Module (JDPM)**：捕获高层语义特征引导定位
2. **Entanglement Transformer Block (ETB)**：在频率和空间域进行纠缠学习生成判别性特征
3. **Dual-domain Reverse Parser (DRP)**：在双域中聚合多层级特征流

输入图像经backbone（PVTv2/ResNet50/Res2Net）编码为4层初始特征$\{\mathcal{P}_i\}_{i=1}^4$，经JDPM生成粗定位$\mathcal{P}_5$，经ETB生成判别性特征$\{\mathcal{X}_i\}_{i=1}^4$，最后通过DRP输出预测图$\{\mathcal{N}_i\}_{i=1}^4$。

### 关键设计

#### 1. Joint Domain Perception Module (JDPM)

JDPM利用分层结构提取频率-空间的多感受野信息。以最高层特征$\mathcal{P}_4$为输入：

- 先通过1×1卷积降维到128通道
- 用一组不同膨胀率的3×3空洞卷积（$z = 2n+1$）获取局部多尺度空间特征$\{\mathcal{J}_n^s\}_{n=1}^4$
- 将空间特征通过FFT→权重过滤→IFFT→取模得到全局频率特征$\{\mathcal{J}_n^f\}_{n=1}^4$：
$$\mathcal{J}_n^f = \Phi\|ifft(\sigma(fft(\mathcal{J}_n^s)) * fft(\mathcal{J}_n^s))\|$$
- 空间+频率特征逐元素相加：$\mathcal{J}_n = \mathcal{J}_n^s + \mathcal{J}_n^f$
- 拼接所有尺度特征并引入残差连接生成1通道粗定位图$\mathcal{P}_5$

设计动机：空间域的卷积感受野有限，通过FFT引入频率变换可以实现全局感知，同时多尺度空洞卷积覆盖不同范围的上下文。

#### 2. Entanglement Transformer Block (ETB)

ETB是本文核心，包含三个子组件实现频率-空间纠缠：

**Frequency Self-Attention (FSA)**：对输入特征做FFT得到频率域的Q/K/V，构建频率注意力图。由于频率注意力图是复数类型，不能直接用Softmax激活，因此将其分解为实部和虚部分别激活再合并：

$$a\Lambda_f = \Theta(Sof(\Lambda_f^{re}), Sof(\Lambda_f^{im}))$$

通过IFFT和取模操作得到频率注意力特征$\mathcal{X}_f^1$。设计动机：建模不同频段之间的依赖关系和重要性权重，而非仅关注高/低频。

**Spatial Self-Attention (SSA)**：使用深度可分离卷积（3×3和5×5）嵌入多尺度空间上下文，生成Q/K/V进行标准自注意力操作。

**Entanglement Feed-Forward Network (EFFN)**：两阶段纠缠学习：
- **第一阶段**：将融合后特征分别映射到频率域（FFT→权重过滤→GELU门控）和空间域（深度可分离卷积→GELU门控），得到$\hat{\mathcal{X}}_f^2$和$\hat{\mathcal{X}}_s^2$
- **第二阶段**：将两个域的特征再次纠缠交互——频率和空间特征互相拼接后分别在各自域中优化，最终聚合并残差连接

$$\hat{\mathcal{X}}_f^3 = \Phi\|ifft(\sigma(fft(Cat(\hat{\mathcal{X}}_f^2, \hat{\mathcal{X}}_s^2))) * fft(Cat(\hat{\mathcal{X}}_f^2, \hat{\mathcal{X}}_s^2)))\|$$
$$\hat{\mathcal{X}}_s^3 = \mathcal{DC}_3 Cat(\hat{\mathcal{X}}_f^2, \hat{\mathcal{X}}_s^2)$$

设计动机：频率域关注全局能量分布和信号变化，空间域作用于局部像素级细节，两者互补。纠缠学习让不同状态的特征相互适应，形成更鲁棒的表示。

#### 3. Dual-domain Reverse Parser (DRP)

DRP设计为双分支结构，在频率和空间双域中优化和聚合多层级特征：

- **分支1**：将辅助特征$\mathcal{P}_5$扩展通道后与ETB输出$\mathcal{X}_4$拼接融合，分别在频率域（FFT→过滤→IFFT）和空间域（卷积序列）中优化，相加得到$\mathcal{N}_4^1$
- **分支2**：生成混合反向注意力图$\mathcal{A}_r$（同时包含空间和频率域的反转信息），用它加权特征获取聚焦于难区分区域的反向特征$\mathcal{N}_4^2$
- 两分支拼接融合得到最终输出，以密集连接方式逐级优化低层特征

### 损失函数 / 训练策略

使用加权BCE和加权IoU的多层级监督：

$$\mathcal{L}_{all} = \sum_{i=1}^5 \frac{1}{2^{i-1}} (\mathcal{L}_{bce}^w(\mathcal{N}_i, G) + \mathcal{L}_{iou}^w(\mathcal{N}_i, G))$$

- 5层输出按$1, 1/2, 1/4, 1/8, 1/16$权重递减监督
- 训练设置：Adam优化器，初始lr=1e-4，每60 epoch衰减10倍，输入416×416，batch=40，共180 epochs
- 4块NVIDIA GTX 4090训练

## 实验关键数据

### 主实验：三大COD基准（PVTv2骨干 Ours-Pvt）

| 方法 | CAMO $\mathcal{M}$↓ | CAMO $F_\varphi^m$↑ | CAMO $S_m$↑ | COD10K $\mathcal{M}$↓ | COD10K $F_\varphi^m$↑ | COD10K $S_m$↑ | NC4K $\mathcal{M}$↓ | NC4K $S_m$↑ |
|------|-----|-----|-----|------|------|-----|------|------|
| SINet (CVPR'20) | .100 | .762 | .751 | .051 | .708 | .770 | .058 | .807 |
| FEDER (CVPR'23) | .071 | .824 | .802 | .032 | .788 | .820 | .044 | .846 |
| FSPNet (CVPR'23) | .050 | .869 | .855 | .026 | .816 | .847 | .035 | .878 |
| HiNet (AAAI'23) | .055 | .857 | .849 | .023 | .850 | .868 | .037 | .874 |
| FPNet (MM'23) | .056 | .863 | .851 | .029 | .817 | .847 | — | — |
| **Ours-Pvt** | **.040** | **.891** | **.885** | **.021** | **.853** | **.873** | **.030** | **.892** |

相较之前最优方法：CAMO的$\mathcal{M}$从0.050降至0.040（提升20%），COD10K的$\mathcal{M}$从0.023降至0.021，全面SOTA。

### 消融实验：各模块贡献（ResNet50骨干）

| 配置 | Baseline | ETB | DRP | JDPM | CAMO $\mathcal{M}$↓ | CAMO $S_m$↑ | COD10K $\mathcal{M}$↓ | COD10K $S_m$↑ |
|------|---------|-----|-----|------|-----|-----|------|------|
| (a) | ✓ | | | | .093 | .767 | .046 | .778 |
| (b) | ✓ | ✓ | | | .076 | .801 | .034 | .821 |
| (c) | ✓ | | ✓ | | .074 | .810 | .034 | .826 |
| (d) | ✓ | | | ✓ | .081 | .787 | .039 | .804 |
| (h) | ✓ | ✓ | ✓ | ✓ | **.067** | **.821** | **.031** | **.830** |

ETB内部频率vs空间消融（ETB-S仅含空间, ETB-F仅含频率, ETB完整）表明两个域的纠缠学习互补不可或缺。

### 效率分析

| 方法 | 参数量(M) | FLOPs(G) |
|------|-----------|----------|
| SINet | 48.95 | 38.75 |
| FEDER | 37.37 | 23.98 |
| FSPNet | 273.79 | 283.31 |
| **Ours-R50** | **29.15** | **35.64** |
| Ours-Pvt | 67.13 | 54.73 |

### 关键发现

1. FSEL在三个数据集上全面超越21个SOTA方法，尤其在CAMO数据集上$\mathcal{M}$指标提升显著
2. 频率和空间特征缺一不可——仅用频率（ETB-F）或仅用空间（ETB-S）均不及纠缠学习的完整ETB
3. R50版本的参数量和FLOPs均处于中等偏低水平（29.15M / 35.64G），但性能远超同体量方法
4. 模型泛化到显著性目标检测和息肉分割任务同样有效

## 亮点与洞察

- **频率域自注意力的实数/虚数分解**：频率注意力图是复数类型，需要分离实部和虚部分别Softmax再合并，这个处理方式值得借鉴
- **纠缠学习的隐喻**：借用量子纠缠的概念描述频率-空间特征的双向交互，使两种"状态"的特征通过信息交换形成更强表示
- **全频段关注而非仅高低频**：与FPNet、FEDER等仅关注高低频的方法不同，通过频段间的自注意力建模所有频率的关系和重要性

## 局限与展望

- FFT/IFFT操作的计算开销随分辨率升高而增大，大分辨率场景下效率有待优化
- 纠缠学习的层数和交互方式较为固定（两阶段），可探索自适应的交互策略
- 仅在COD任务上验证，虽展示了SOD和息肉分割的泛化性，但更多分割任务（如实例分割）的效果未知
- 缺少对不同频率变换方法（DCT vs FFT vs 小波等）的系统对比

## 相关工作与启发

- 频率域特征在COD中的作用被低估，本文证明了全频段交互的重要性
- 纠缠学习思路可推广到其他需要多域特征融合的任务（如医学影像、遥感）
- 频率自注意力处理复数的技巧（实部/虚部分解+分别激活）可作为通用模块

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 频率-空间纠缠学习的框架设计新颖，频率自注意力的复数处理有技术创新
- **实验充分度**: ⭐⭐⭐⭐⭐ — 21个方法对比+3个数据集+3个backbone+详尽消融+扩展应用
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，公式完整，图表丰富
- **实用价值**: ⭐⭐⭐⭐ — COD全面SOTA，ETB模块可即插即用到其他方法中

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Learning Camouflaged Object Detection from Noisy Pseudo Label](learning_camouflaged_object_detection_from_noisy_pseudo_label.md)
- [\[ECCV 2024\] Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales](self-supervised_co-salient_object_detection_via_feature_correspondences_at_multi.md)
- [\[CVPR 2026\] Beyond Appearance: Camouflaged Object Detection via Geometric Structure](../../CVPR2026/segmentation/beyond_appearance_camouflaged_object_detection_via_geometric_structure.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](../../CVPR2026/segmentation/sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[ICCV 2025\] Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection](../../ICCV2025/segmentation/beyond_single_images_retrieval_self-augmented_unsupervised_camouflaged_object_de.md)

</div>

<!-- RELATED:END -->
