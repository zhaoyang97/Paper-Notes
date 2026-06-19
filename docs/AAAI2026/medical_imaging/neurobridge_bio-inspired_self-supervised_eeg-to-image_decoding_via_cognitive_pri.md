---
title: >-
  [论文解读] NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding via Cognitive Priors and Bidirectional Semantic Alignment
description: >-
  [AAAI 2026][医学图像][EEG解码] 提出NeuroBridge框架，通过认知先验增强（CPA，非对称增广模拟感知变异性）和共享语义投影器（SSP，双向对齐到统一语义空间），在THINGS-EEG数据集200类零样本EEG-图像检索任务上达到63.2% Top-1（+12.3%）和89.9% Top-5（+10.2%），大幅超越现有SOTA。
tags:
  - "AAAI 2026"
  - "医学图像"
  - "EEG解码"
  - "跨模态对比学习"
  - "认知先验增强"
  - "共享语义投影"
  - "零样本检索"
  - "脑机接口"
---

# NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding via Cognitive Priors and Bidirectional Semantic Alignment

**会议**: AAAI 2026  
**arXiv**: [2511.06836](https://arxiv.org/abs/2511.06836)  
**作者**: Wenjiang Zhang, Sifeng Wang, Yuwei Su, Xinyu Li, Chen Zhang, Suyu Zhong  
**代码**: [GitHub](https://github.com/feroooooo/NeuroBridge)  
**领域**: 自监督  
**关键词**: EEG解码, 跨模态对比学习, 认知先验增强, 共享语义投影, 零样本检索, 脑机接口  

## 一句话总结

提出NeuroBridge框架，通过认知先验增强（CPA，非对称增广模拟感知变异性）和共享语义投影器（SSP，双向对齐到统一语义空间），在THINGS-EEG数据集200类零样本EEG-图像检索任务上达到63.2% Top-1（+12.3%）和89.9% Top-5（+10.2%），大幅超越现有SOTA。

## 研究背景与动机

### 问题背景
视觉神经解码旨在从脑活动重建或推断感知到的视觉刺激，在脑机接口和AI领域有重要应用。EEG因高时间分辨率、低成本、高便携性成为非侵入式神经成像的重要手段。跨模态对比学习是当前EEG视觉解码的主流范式。

### 已有工作的不足

**动态变异性鸿沟**：同一被试观看同一图像的EEG响应因注意力波动、心理状态、生理噪声等因素而存在显著变异——不同人关注同一张猫图片的不同语义区域，产生差异化的EEG响应

**静态固有鸿沟**：EEG是时序、低维、含噪信号，图像是空间结构化的高维语义密集数据，模态差异根本性存在

**单向对齐局限**：已有方法（如NICE、ATM）多采用单向对齐策略，将EEG对齐到冻结的CLIP空间，但CLIP的语义空间由视觉-语言数据塑造，与EEG反映的感知/认知过程存在语义不匹配

**增广策略不足**：UBP仅在视觉侧引入模糊先验，Neural-MCRL仅在EEG侧做语义补全，缺乏综合性双模态增强框架

**数据稀缺**：EEG配对数据规模远小于视觉-语言领域的大规模数据集

### 核心动机
受生物系统的感知变异性（Perceptual Variability）和协同适应策略（Co-adaptive Strategy）启发：用非对称增广模拟人类认知变异，用双向投影实现EEG和图像在统一语义空间的协同对齐。

## 方法详解

### 整体框架
训练阶段：配对EEG-图像数据 → CPA认知先验增广 → 编码（冻结CLIP图像编码器 + 可训练EEG编码器）→ SSP投影到共享语义空间 → 双向对比损失优化。推理阶段：EEG嵌入通过余弦相似度在视觉概念池中检索匹配图像。

### 模块1：认知先验增强（CPA）

CPA模拟人类视觉感知过程中的认知变异性，核心设计是**非对称增广策略**：

- **图像侧**：应用K种增广策略（高斯模糊、高斯噪声、低分辨率、马赛克），生成多视图 $X'_{I,k} = t_{I,k}(X_I)$，编码后取平均 $H_I = \frac{1}{K}\sum_{k=1}^{K} H_{I,k}$ 获得语义聚合表征
- **EEG侧**：仅用单一增广（平滑处理） $X'_E = t_E(X_E)$
- **非对称设计动机**：CLIP图像编码器预训练充分，多增广可充分挖掘其能力；EEG编码器从头训练，过多增广反而损坏信号结构

关于增广选择的发现：高斯模糊/噪声/低分辨率/马赛克有效，因保留高层语义同时削弱低层像素变化；颜色抖动和灰度化效果消极，表明人类感知对颜色信息敏感（符合神经科学研究）；随机裁剪可能移除关键语义区域。EEG侧仅平滑处理有效，因EEG信噪比低，平滑可降噪；时间偏移会扰乱时序动态。

### 模块2：共享语义投影器（SSP）

SSP将两个模态的特征映射到统一的可训练语义空间：

$$Z_I = p_I(H_I), \quad Z_E = p_E(H_E)$$

其中 $p_I$ 和 $p_E$ 是两个投影网络（默认512维线性投影效果最优）。关键区别：**$p_I$ 也是可训练的**（而非固定不变），实现双向对齐而非单向将EEG对齐到CLIP空间。这种共享空间学习比直接利用CLIP固定空间更灵活。

### 模块3：模态感知对比学习

采用双向InfoNCE损失，关键设计是**非对称归一化**：仅对图像特征做 $\ell_2$ 归一化到单位超球面，EEG特征幅度保持自由。这样利用特征方向做语义对齐，幅度作为可学习的置信度。实验表明（Table 7）这种非对称策略（Asym: 63.2%）显著优于对称归一化（Sym: 46.4%）和无归一化（Plain: 54.4%）。

### 训练细节
- 数据集：THINGS-EEG（10个被试，RSVP范式，训练集1654概念×10图×4重复=16540样本/被试，测试集200概念×1图×80重复）
- 默认配置：RN50图像编码器 + EEGProject脑电编码器（2.44M参数），batch=1024，epoch=50，lr=1e-4，τ=0.07
- Intra-subject用17个顶枕区电极（P7/P5/P3/P1/Pz/P2/P4/P6/P8/PO7/PO3/POz/PO4/PO8/O1/Oz/O2），Inter-subject用全部63通道+TSConv编码器

## 实验关键数据

### 表1：THINGS-EEG 200类零样本检索主实验（Intra-Subject平均）

| 方法 | Top-1 (%) | Top-5 (%) |
|------|-----------|-----------|
| BraVL | 5.8 | 17.5 |
| NICE | 16.1 | 43.6 |
| ATM | 27.1 | 58.1 |
| CognitionCapturer | 33.3 | 60.6 |
| Neural-MCRL | 32.4 | 64.1 |
| VE-SDN | 37.2 | 70.0 |
| UBP | 50.9 | 79.7 |
| **NeuroBridge** | **63.2** | **89.9** |

NeuroBridge相比前SOTA（UBP）提升+12.3% Top-1和+10.2% Top-5。Sub 10上最高达73.6% Top-1和97.1% Top-5。Inter-subject设定下同样SOTA（19.0% Top-1 vs UBP 12.4%，45.9% Top-5 vs UBP 33.4%）。

### 表2：消融实验（核心组件贡献）

| Image Prior | EEG Prior | SSP | Top-1 (%) | Top-5 (%) |
|:-----------:|:---------:|:---:|-----------|-----------|
| ✗ | ✗ | ✗ | 40.5 | 72.2 |
| ✓ | ✗ | ✗ | 60.0 | 89.1 |
| ✗ | ✓ | ✗ | 40.8 | 72.7 |
| ✗ | ✗ | ✓ | 41.5 | 73.5 |
| ✓ | ✗ | ✓ | 62.1 | 89.8 |
| ✓ | ✓ | ✗ | 60.8 | 89.8 |
| ✓ | ✓ | ✓ | 63.2 | 89.9 |

Image Prior贡献最大（+19.5% Top-1），是性能飞跃的关键；EEG Prior和SSP各自带来约1%的增益；三者组合达到最优。

### 补充实验
- **融合变换数量**：从1到4个图像增广融合效果递增（50.9%→62.1%），5个以上开始下降（58.5%），4个是最优平衡点
- **投影器设计**：512维线性投影最优，MLP和更高维度反而过拟合
- **归一化策略**：非对称归一化（仅归一化图像）63.2%，对称归一化46.4%，无归一化54.4%，反向非对称（仅归一化EEG）38.6%
- **Batch Size**：1024最优（63.2%），比32提升+8.6%，2048略降至62.2%
- **温度参数**：τ=0.5时最优（63.6%），NeuroBridge对温度变化比标准对比学习更鲁棒
- **THINGS-MEG验证**：在MEG数据上同样SOTA，Intra-subject 32.2% Top-1 vs UBP 26.7%，Inter-subject 3.4% vs UBP 2.2%
- **编码器通用性**：在所有图像编码器（RN50到ViT-bigG-14）和EEG编码器（EEGNet/TSConv/ATM/EEGProject）组合上均带来一致提升

## 亮点

- **生物启发的非对称增广**：直接模拟人类感知变异性，图像多视图模拟不同注意力聚焦，EEG轻增广保留时序动态，有明确认知科学依据
- **双向对齐突破**：SSP打破了将EEG强行对齐到CLIP固定空间的惯例，学习新的共享空间更灵活
- **非对称归一化洞察**：仅归一化图像特征让EEG幅度编码置信度，简单但关键（+16.8% vs对称归一化）
- **大幅度SOTA提升**：在竞争激烈的THINGS-EEG基准上Top-1提升12.3%，且在所有10个被试上均优于前SOTA
- **编码器通用性**：框架可插拔不同图像/EEG编码器组合，均带来一致提升
- **代码开源**：完整代码公开可复现

## 局限与展望

- **手工设计增广**：CPA的增广策略（高斯模糊、低分辨率等）是人工选定的，可能不足以完整捕获认知变异性，自适应/可学习增广是未来方向
- **依赖预训练视觉编码器**：冻结的CLIP编码器可能引入视觉-语言偏差，EEG反映的语义与语言驱动的语义可能存在本质差异
- **仅限零样本检索任务**：未验证在图像生成/重建等更具挑战性的解码任务上的效果
- **数据规模受限**：仅在THINGS-EEG（1654概念）和THINGS-MEG上验证，更大规模数据集上的表现未知
- **EEG增广贡献有限**：EEG Prior仅提升0.3% Top-1，说明在EEG增广策略上仍有很大探索空间
- **非对称归一化缺乏理论解释**：为何允许EEG幅度自由编码置信度有效，文中仅有推测缺乏理论分析

## 与相关工作的对比

- **UBP（Wu et al. 2025）**：当时SOTA，引入模糊先验模拟早期视觉感知，仅在视觉侧增强。NeuroBridge在所有被试上均超越，平均+12.3% Top-1
- **Neural-MCRL（Li et al. 2024）**：引入EEG模态内语义补全，仅在EEG侧增强。NeuroBridge综合双模态，+30.8% Top-1
- **VE-SDN（Chen et al. 2024）**：视觉增强解码网络，37.2% → 63.2%，+26.0% Top-1
- **CognitionCapturer（Zhang et al. 2025）**：认知捕获方法，33.3% → 63.2%，+29.9% Top-1
- **NICE（Song et al. 2024）**：TSConv编码器+简单对比学习基线，16.1% → 63.2%
- **CLIP/ALIGN/BLIP等**：大规模视觉-语言预训练依靠海量配对数据成功，但EEG-图像缺乏此量级数据，因此需要更强的架构先验（CPA+SSP）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 非对称增广+双向SSP组合设计有创新性，但各模块单独看较为直觉
- 实验充分度: ⭐⭐⭐⭐⭐ — 10个被试全面评估、多编码器验证、详尽消融、THINGS-MEG泛化、超参敏感性分析
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，生物启发叙事连贯，消融实验设计合理
- 价值: ⭐⭐⭐⭐ — 在EEG视觉解码方向取得显著进展，但增广策略的手工设计限制了方法论深度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)
- [\[CVPR 2026\] D$^2$-FOSA: Dual-Diffusion Guided EEG-to-Image Reconstruction with Frequency-Oriented Semantic Alignment](../../CVPR2026/medical_imaging/d2-fosa_dual-diffusion_guided_eeg-to-image_reconstruction_with_frequency-oriente.md)
- [\[AAAI 2026\] Self-supervised Multiplex Consensus Mamba for General Image Fusion](self-supervised_multiplex_consensus_mamba_for_general_image_fusion.md)
- [\[AAAI 2026\] CAT-Net: A Cross-Attention Tone Network for Cross-Subject EEG-EMG Fusion Tone Decoding](cat-net_a_cross-attention_tone_network_for_cross-subject_eeg-emg_fusion_tone_dec.md)
- [\[AAAI 2026\] Shrinking the Teacher: An Adaptive Teaching Paradigm for Asymmetric EEG-Vision Alignment](shrinking_the_teacher_an_adaptive_teaching_paradigm_for_asymmetric_eeg-vision_al.md)

</div>

<!-- RELATED:END -->
