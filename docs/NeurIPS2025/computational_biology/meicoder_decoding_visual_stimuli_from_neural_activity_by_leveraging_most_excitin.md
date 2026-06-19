---
title: >-
  [论文解读] MEIcoder: Decoding Visual Stimuli from Neural Activity by Leveraging Most Exciting Inputs
description: >-
  [NeurIPS 2025][计算生物][视觉解码] 提出 MEIcoder，利用神经元特异性的最激励输入(MEI)作为生物学先验、SSIM 损失和对抗训练，从初级视觉皮层(V1)的神经群体活动中实现 SOTA 级别的视觉刺激重建，尤其在小数据集和少量神经元场景下表现突出。 从大脑活动解码视觉刺激是理解大脑的关键课题…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "视觉解码"
  - "最激励输入(MEI)"
  - "初级视觉皮层"
  - "神经群体活动"
  - "对抗训练"
---

# MEIcoder: Decoding Visual Stimuli from Neural Activity by Leveraging Most Exciting Inputs

**会议**: NeurIPS 2025  
**arXiv**: [2510.20762](https://arxiv.org/abs/2510.20762)  
**代码**: [https://github.com/Johnny1188/meicoder](https://github.com/Johnny1188/meicoder)  
**领域**: 计算神经科学 / 脑机接口  
**关键词**: 视觉解码, 最激励输入(MEI), 初级视觉皮层, 神经群体活动, 对抗训练

## 一句话总结

提出 MEIcoder，利用神经元特异性的最激励输入(MEI)作为生物学先验、SSIM 损失和对抗训练，从初级视觉皮层(V1)的神经群体活动中实现 SOTA 级别的视觉刺激重建，尤其在小数据集和少量神经元场景下表现突出。

## 研究背景与动机

从大脑活动解码视觉刺激是理解大脑的关键课题，也是脑机接口的重要应用。但面临若干根本性挑战：

**数据稀缺**: 单被试数据量有限，从头训练机器学习模型往往得到低保真重建

**生成式AI幻觉**: 利用预训练生成模型(如扩散模型)虽能产生高分辨率图像，但常包含虚构内容——先前研究已正式证明基于扩散的解码存在"输出维度坍塌"，限制了可解码特征

**逆问题的病态性**: 从少量神经元(高度压缩和含噪)反推高信息量图像本身极为困难

现有方法要么重建保真度低，要么不可靠(幻觉)，而且多数方法不在像素空间直接优化重建质量。因此需要一种平衡生物学先验与数据信号的新方法。

## 方法详解

### 整体框架

MEIcoder 由两部分组成：(1) **Readin 模块**——每个被试/数据集独立训练，将神经活动嵌入到核心模块的潜在空间；(2) **Core 模块**——跨被试共享的六层 CNN，从潜在空间映射回图像。这种拆分设计允许在异构多被试数据间复用学习信号。

### 关键设计

1. **MEI 驱动的 Readin 模块**: 这是核心创新。每个神经元的响应 $r_i$ 与其可学习嵌入 $\mathbf{e}_i$ 一起通过单层网络 $g_\psi$ 映射为上下文表示 $\mathbf{C}_i \in \mathbb{R}^{h \cdot w}$。然后将预计算的 MEI（最激励输入，即最大化该神经元响应的图像）$\mathbf{M}_i$ 与上下文表示逐元素相乘，得到神经映射 $\mathbf{H} = \mathbf{M} \odot \mathbf{C}$。最后通过逐点卷积压缩到固定通道数 $d_c$。

    - **直觉**: MEI 包含神经元感受野信息。简单的线性解码方法是将所有神经元的 MEI 按响应强度叠加来重建图像。MEIcoder 将这种组合交给非线性 CNN core 处理
    - MEI 仅需从训练数据中生成一次（约 70 分钟 vs 解码器训练 11 小时）

2. **SSIM 重建损失**: 使用负对数 SSIM 损失替代传统 MSE，$\mathcal{L}_{SSIM} = -\log(\frac{SSIM(\mathbf{y}, \hat{\mathbf{y}}) + 1}{2} + \epsilon)$。SSIM 关注亮度、对比度和结构三个感知维度，比 MSE 更能引导生成感知准确的重建。VGG 感知损失在训练中不稳定且产生高频伪影。

3. **对抗训练**: 辅助 CNN 判别器区分重建图像与真实图像，通过 LS-GAN 损失推动重建走向自然图像流形。与标准 GAN 不同，解码器以神经响应为条件，且直接优化空间准确重建。判别器训练引入目标噪声(类似单侧标签平滑)以稳定训练。损失权重 $\lambda_{SSIM}=0.9$，$\lambda_{ADV}=0.1$。

### 损失函数 / 训练策略

- 总损失: $\mathcal{L} = 0.9 \cdot \mathcal{L}_{SSIM} + 0.1 \cdot \mathcal{L}_{ADV}$
- AdamW 优化器，300 epochs，基于验证集 Alex(5) 分数选择最佳 checkpoint
- Core 模块参数高效：6 层 CNN，通道数 480→256→256→128→64→1，dropout 0.35
- 多被试训练：共享 core，为不同被试训练独立 readin，参数量减少三倍

## 实验关键数据

### 主实验

在三个数据集上评估(Brainreader/SENSORIUM 2022: 真实小鼠 V1; Synthetic Cat V1: 高仿真猫 V1 模型)。

| 方法 | Brainreader SSIM | Brainreader Alex(5) | SENSORIUM SSIM | SENSORIUM Alex(5) |
|------|-----------------|--------------------|-----------------|--------------------|
| InvEnc | .321 | .896 | .288 | .720 |
| EGG | .256 | .659 | .256 | .755 |
| MonkeySee | .232 | .826 | .185 | .523 |
| MindEye2 | .277 | .878 | .210 | .762 |
| **MEIcoder** | **.400** | **.990** | **.331** | **.896** |
| **MEIcoder (FT)** | **.424** | .977 | .318 | **.908** |

| 方法 | Synthetic Cat V1 SSIM | PixCorr | Alex(2) | Alex(5) |
|------|----------------------|---------|---------|---------|
| InvEnc | **.771** | **.833** | .986 | .978 |
| MEIcoder | **.774** | .777 | **.994** | **.987** |

### 消融实验

| 消融设置 | 相对影响(对比无消融) | 说明 |
|---------|--------------------|----|
| 去除 MEI | **影响最大** (所有指标显著下降) | MEI 是性能主要驱动力 |
| 去除神经元嵌入 | 中等影响 | 丢失额外编码属性 |
| 用 MSE 替代 SSIM | 中等影响 | 感知质量下降 |

### 缩放实验关键发现

| 条件 | 关键观察 |
|------|---------|
| 训练数据量 → 1000 | 已超越第二好方法全量训练的 Alex(2) |
| 神经元数 → 1000~2500 | Alex(2) 达到 95%+，可区分手写数字 |
| 46875 神经元 | PixCorr 仍未饱和，说明更多神经元仍有收益 |

### 关键发现

- MEI 是性能的最大驱动因素——即使 MEI 有大量高斯噪声(std=1)，性能仍与 baseline 持平或更好
- MEIcoder 的方差显著小于 MindEye2，表明结果更可靠，避免了 GenAI 方法的幻觉问题
- 仅需 1000 个训练样本即可超越其他方法的全量训练表现
- 1000~2500 个 V1 神经元足以进行精细重建
- 多被试预训练 + 单被试微调在 Brainreader 上进一步提升，但在 SENSORIUM 上未见提升
- 概念分析发现：(1) 解码层级从粗到细，(2) 许多神经元编码全局亮度条件，(3) 存在与"黑色主导"OFF 神经元一致的响应模式

## 亮点与洞察

- **MEI 作为解码先验是核心创新**: 不同于 GenAI 方法使用与脑信号无关的图像先验，MEI 直接来自编码模型对实际神经元的建模，提供了生物学对齐的先验
- 数据效率极高：1000 样本 + 1000 神经元即可有意义地解码，这对实际 BCI 应用至关重要
- Core-Readin 分离架构简洁实用，解决了多被试异构数据的复用问题
- MindEye2 等基于扩散模型的方法虽然视觉上看起来sharp，但空间不准确且存在幻觉——本文提供了正式证据
- 统一 benchmark(16万+样本)的贡献对社区有持续价值
- 概念分析展示了 MEIcoder 作为科学发现工具的潜力

## 局限与展望

- 仅在 V1(编码低层特征)上验证，更高级视觉区域(V4等)的适用性未测试
- 跨被试迁移效果不一致(Brainreader 有效，SENSORIUM 无效)
- 合成猫数据虽然验证了方法，但与真实灵长类数据的差距仍然存在
- MEI 质量依赖于编码模型的准确性，编码模型本身的局限可能传递到解码
- 重建分辨率仍较低（36×64, 22×36 等），高分辨率图像重建有待探索
- 未在人类数据上验证，从小鼠/猫到人类的跨物种泛化是开放问题

## 相关工作与启发

- InvEnc (Cobos et al., 2022) 通过编码器反转进行解码的思路影响了本文
- Energy Guided Diffusion (Pierzchlewicz et al., 2023) 用扩散先验引导解码，但优化在神经活动空间而非像素空间
- MonkeySee (Le et al., 2024) 的同胚解码器和 U-Net 架构是直接对比对象
- Shirakawa et al. (2025) 关于"虚假重建"的正式分析支撑了本文避免 GenAI 先验的选择
- 对其他模态(fMRI, EEG)的解码研究有方法论参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Synthetic Visual Genome](../../CVPR2025/computational_biology/synthetic_visual_genome.md)
- [\[NeurIPS 2025\] SpecMER: Fast Protein Generation with K-mer Guided Speculative Decoding](specmer_fast_protein_generation_with_k-mer_guided_speculative_decoding.md)
- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/computational_biology/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)

</div>

<!-- RELATED:END -->
