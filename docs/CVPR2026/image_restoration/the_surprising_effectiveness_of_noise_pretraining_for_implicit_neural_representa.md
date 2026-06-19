---
title: >-
  [论文解读] The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations
description: >-
  [CVPR 2026][图像恢复][隐式神经表示] 本文通过系统的实验分析发现：用非结构化噪声（均匀/高斯分布）预训练 INR 可在图像拟合中达到惊人的 ~80dB PSNR，远超所有数据驱动初始化方法；而具有自然图像 $1/|f^\alpha|$ 频谱结构的噪声则在信号拟合和去噪之间实现最佳平衡，无需任何真实数据即可匹配 SOTA 数据驱动初始化性能。
tags:
  - "CVPR 2026"
  - "图像恢复"
  - "隐式神经表示"
  - "噪声预训练"
  - "参数初始化"
  - "信号拟合"
  - "去噪"
---

# The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations

**会议**: CVPR 2026  
**arXiv**: [2603.29034](https://arxiv.org/abs/2603.29034)  
**代码**: 有（项目页面已公开）  
**领域**: 图像恢复 / 隐式神经表示  
**关键词**: 隐式神经表示、噪声预训练、参数初始化、信号拟合、去噪

## 一句话总结

本文通过系统的实验分析发现：用非结构化噪声（均匀/高斯分布）预训练 INR 可在图像拟合中达到惊人的 ~80dB PSNR，远超所有数据驱动初始化方法；而具有自然图像 $1/|f^\alpha|$ 频谱结构的噪声则在信号拟合和去噪之间实现最佳平衡，无需任何真实数据即可匹配 SOTA 数据驱动初始化性能。

## 研究背景与动机

**领域现状**：隐式神经表示（INR）用 MLP 将空间坐标映射为信号值，在压缩、逆成像、神经渲染等领域广泛应用。INR 的收敛性能高度依赖参数初始化策略——数据驱动方法（如 meta-learning、Strainer）已被证明远优于标准随机初始化。

**现有痛点**：尽管数据驱动初始化效果显著，其成功的根本原因仍然不清楚——是编码了经典的统计信号先验？还是更复杂的数据特定特征？这种不清楚限制了在缺乏领域数据的科学成像等场景中的应用。

**核心矛盾**：数据驱动初始化需要先验真实信号，但许多应用领域（如科学成像）缺乏充足的领域数据。如果能理解初始化成功的根本机制，可能找到不依赖真实数据的高效替代方案。

**本文目标** (1) 数据驱动 INR 初始化的性能增益来源于什么层面的信号特性？(2) 是否可以用不含真实数据的噪声来替代数据驱动初始化？(3) 不同类型噪声的预训练对信号拟合和逆问题求解有何不同影响？

**切入角度**：受噪声预训练视觉分类网络的工作启发，用不同类别噪声预训练 INR，通过对比实验揭示初始化成功的底层机制——每种噪声类别有精确定义的性质，可作为控制变量分析。

**核心 idea**：用噪声代替真实数据预训练 INR，发现非结构化噪声是信号拟合之王而频谱噪声是全能选手，无需真实数据即可高效初始化 INR。

## 方法详解

### 整体框架

这篇论文要回答的不是"怎么把 INR 训得更好"，而是"数据驱动初始化为什么有用，能不能不用真实数据"。它的实验载体直接沿用 Strainer 的 INR 框架：一个 6 层 MLP（sine 激活，$\omega=30$），前 5 层是所有信号共享的编码器，最后一层是每个信号各自的解码器头。预训练时在 $N=10$ 个样本上联合训练 5000 步，把编码器调成一个好的"起点"；到测试时冻结编码器、给新信号随机初始化一个新的解码器头，再训练 2000 步去拟合。整套流程不变，唯一被替换的是预训练用的数据——把真实图像换成各类噪声，这个变体被命名为 Snp（Strainer Noise Pretraining）。换句话说，初始化方法本身是现成的，论文真正的设计在于"用什么噪声、由此发现了什么"。

### 关键设计

**1. 多类噪声作为受控变量：把"初始化先验"拆成可分离的成分**

数据驱动初始化效果好却说不清原因，是因为真实图像里同时混着低层统计、频谱结构、语义特征等一堆东西，没法归因。本文的做法是用统计性质精确已知的噪声当受控变量：一端是**非结构化噪声**（均匀 $\mathcal{U}(0,1)$、高斯 $\mathcal{N}(0.5,0.2)$），几乎不含任何空间结构；另一端是**结构化噪声**，包括 Dead Leaves 系列（方块/有向/混合/纹理）和统计模型系列（频谱 $1/|f^\alpha|$、频谱+颜色、小波边缘模型），后者通过在随机噪声上施加自然图像的频谱或边缘统计生成。每类噪声只要 10 个样本就能完成预训练。由于每种噪声"含有什么先验"是事先定义清楚的，对比它们的初始化效果就等于在逐项剥离不同层面的信号特性对收敛的贡献——这是整篇分析能成立的方法论基础。

**2. 同时量信号拟合与去噪：暴露容量与先验不可兼得的权衡**

如果只看拟合 PSNR，很容易得出"噪声越没结构越好"的错觉，但这会漏掉 INR 作为逆问题求解器的另一半能力。本文坚持在两个维度上同时评估，结果拉出一条清晰的权衡曲线：非结构化噪声把信号拟合推到惊人的 ~80dB PSNR，却在去噪上垫底（23–24dB）；而结构化频谱噪声 $1/|f^\alpha|$ 落在中间最甜的位置——拟合逼近 Strainer（56.4 vs 57.8 dB），去噪逼近 Siren（27.6 vs 28.3 dB）。这条曲线说明 INR 初始化里存在一个根本性的不可兼得：**功能容量**（能多快多准地拟合任意目标）和**深度先验强度**（对干净信号的归纳偏好）是此消彼长的，没有哪种噪声能同时最大化两者，而频谱噪声之所以"全能"恰恰是因为它两边都不极端。

**3. NTK 与局部复杂度：解释两类噪声为何行为迥异**

现象之外，论文进一步追问机制。神经正切核（NTK）分析显示 Snp:Uniform 只用极少的特征值就能覆盖目标信号的大部分能量，这解释了它为何收敛得又快又准。局部复杂度分析则揭示，Snp:Uniform 的前几层把输入空间切成高度非线性、近乎伪随机的划分——这在功能几何上和 Instant-NGP 的哈希编码异曲同工：容量极高，但完全不带结构先验，所以拟合无敌、去噪却差。与之相对，Snp:Spectrum 和真实数据训练的 Strainer 呈现出几乎相同的损失地形，这也就从机理上解释了为什么频谱噪声能在不碰任何真实数据的情况下复现数据驱动初始化的表现。

### 损失函数 / 训练策略

预训练和拟合阶段统一用 L2 损失、Adam 优化器（lr=1e-4），预训练 5000 步、测试拟合 2000 步。去噪任务靠早停（early stopping）选最佳迭代次数，避免 INR 过拟合到噪声上。视频实验换用 ResFields 框架，在每帧上叠加可学习的残差更新，训练 100k 步——结构化频谱噪声在这里依然适用。

## 实验关键数据

### 主实验

图像拟合 PSNR (dB), T=2000 步：

| 方法 | CelebA-HQ | AFHQ | OASIS-MRI |
|------|----------|------|-----------|
| Siren | 44.9 | 45.1 | 53.0 |
| Strainer | 57.8 | 58.0 | 62.8 |
| TransINR | 51.9 | 49.0 | 55.5 |
| IPC | 49.7 | 47.2 | 51.4 |
| **Snp: Uniform** | **85.7** | **79.9** | **79.3** |
| Snp: Gaussian | 80.0 | 77.0 | 79.1 |
| Snp: Spectrum | 56.4 | 56.2 | 60.0 |

### 消融实验

去噪 PSNR (dB) 与最佳迭代步数 (CelebA-HQ)：

| 方法 | PSNR | 最佳步数 |
|------|------|---------|
| Siren | **28.3** | 139 |
| Strainer | 27.3 | 70 |
| Snp: Spectrum | 27.6 | 78 |
| Snp: Uniform | 23.0 | 73 |
| Snp: Gaussian | 23.8 | 70 |

视频拟合与去噪 (Pexels, 100k 步)：

| 任务 | 方法 | 平均 PSNR |
|------|------|----------|
| 拟合 | Vanilla ResFields | 29.5 |
| 拟合 | **ResFields + Snp:Spectrum** | **31.1** |
| 去噪 | Vanilla ResFields | 27.0 |
| 去噪 | **ResFields + Snp:Spectrum** | **28.0** |

### 关键发现

- **非结构化噪声的惊人拟合能力**：Snp:Uniform 达到 85.7dB，比最强数据驱动方法 Strainer（57.8dB）高出近 28dB，这是一个极其出人意料的发现
- **信号拟合 vs 去噪的权衡**：非结构化噪声拟合能力最强但去噪最差，频谱噪声在二者间最佳平衡
- **频谱噪声≈真实数据**：$1/|f^\alpha|$ 噪声预训练性能几乎匹配用真实人脸数据预训练的 Strainer
- **与哈希编码的类比**：Snp:Uniform 的功能几何类似于 Instant-NGP 的哈希编码，高容量来自伪随机的输入空间划分
- **视频领域趋势不同**：频谱噪声在视频拟合中略优于均匀噪声，可能因为视频的时空连续性更受益于结构化先验

## 亮点与洞察

- **发现性贡献极强**：通过精心设计的控制实验揭示了 INR 初始化的核心机制，这是少有的"解释为什么"而非"做什么"的工作
- **实用价值明确**：$1/|f^\alpha|$ 噪声易于生成，可完全替代真实数据进行 INR 初始化——对数据缺乏的科学成像领域尤其有价值
- **权衡洞察深刻**：信号拟合容量和深度先验强度的不可兼得性是对 INR 社区的重要警示
- **跨域泛化**：发现从图像到视频、从人脸到 MRI 均成立，说明是 INR 的普遍性质而非数据特定现象
- **NTK/损失地形分析**：提供了理论层面的解释框架，不停留在经验观察

## 局限与展望

- 仅考察了正弦激活函数和固定层数，不同激活函数（ReLU、Gaussian、Wavelet）的结论可能不同
- 未探索不同网络深度与噪声类型的交互效应
- 去噪实验较简单（仅高斯噪声），更复杂的逆问题（如超分辨率、CT重建）值得验证
- 可探索自适应地混合不同噪声类型的预训练以同时优化拟合和先验

## 相关工作与启发

- Strainer 证明了跨域 INR 初始化的有效性（人脸→动物→MRI），本文进一步证明甚至不需要真实数据
- Looking at Noise 数据集提供了结构化噪声的系统分类，是本文控制实验的基础
- Instant-NGP 的哈希编码与 Snp:Uniform 的功能几何的类比为理解两种方法提供了统一视角

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 用噪声预训练这个反直觉的发现极具启发性，打破了 INR 社区的常规认知
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖多种噪声类型、多个数据集、多个任务、多种可解释性分析，实验设计极为系统
- **写作质量**: ⭐⭐⭐⭐⭐ — 叙事流畅、图表直观、从实验到机理解释的逻辑链完整
- **价值**: ⭐⭐⭐⭐ — 对 INR 社区有深远的理论指导意义，实用价值在数据缺乏的科学成像场景尤为突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)
- [\[CVPR 2026\] Convexity-Aware Noise Calibration: A Self-Supervised Framework for Noise-Level-Unknown Image Denoising](convexity-aware_noise_calibration_a_self-supervised_framework_for_noise-level-un.md)
- [\[CVPR 2026\] Event-Based Motion Deblurring Using Task-Oriented 3D Gaussian Event Representations](event-based_motion_deblurring_with_unpaired_data.md)
- [\[NeurIPS 2025\] Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution](../../NeurIPS2025/image_restoration/implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio.md)
- [\[CVPR 2026\] Towards Generalized Representations for Low-Light Understanding: When Signal Constancy Meets Semantic Enrichment](towards_generalized_representations_for_low-light_understanding_when_signal_cons.md)

</div>

<!-- RELATED:END -->
