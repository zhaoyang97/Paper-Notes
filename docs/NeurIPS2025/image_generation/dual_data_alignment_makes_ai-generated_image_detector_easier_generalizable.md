---
title: >-
  [论文解读] Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable
description: >-
  [NeurIPS 2025 Spotlight][图像生成][AI生成图像检测] 提出 Dual Data Alignment (DDA)，通过像素域和频域双重对齐生成训练用合成图像，消除数据集偏置导致的虚假相关性，使检测器仅学习伪造相关特征，在11个基准上平均准确率达到90.7%，大幅超越现有方法。
tags:
  - "NeurIPS 2025 Spotlight"
  - "图像生成"
  - "AI生成图像检测"
  - "数据对齐"
  - "频域对齐"
  - "泛化性"
  - "VAE重建"
  - "数据偏置"
---

# Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.14359](https://arxiv.org/abs/2505.14359)  
**代码**: [roy-ch/Dual-Data-Alignment](https://github.com/roy-ch/Dual-Data-Alignment)  
**领域**: 图像生成  
**关键词**: AI生成图像检测, 数据对齐, 频域对齐, 泛化性, VAE重建, 数据偏置  

## 一句话总结

提出 Dual Data Alignment (DDA)，通过像素域和频域双重对齐生成训练用合成图像，消除数据集偏置导致的虚假相关性，使检测器仅学习伪造相关特征，在11个基准上平均准确率达到90.7%，大幅超越现有方法。

## 研究背景与动机

**AI生成图像 (AIGI) 威胁日益严峻**：扩散模型、自回归模型等生成模型快速演进，伪造图像在虚假信息、欺诈、版权侵权等场景中构成严重安全威胁，亟需可靠的检测方法。

**现有检测器泛化性不足**：当前检测器在训练集上表现良好，但在跨数据集、跨生成模型的零样本场景中性能显著下降，尤其面对未见过的生成范式时尤为明显。

**数据集偏置是泛化性差的根源**：现有数据集中真实图像与合成图像在格式（JPEG vs PNG）、尺寸（多样 vs 固定为128倍数）、语义内容等非因果属性上存在系统性差异，检测器容易学到这些虚假相关性而非真正的伪造痕迹。

**重建式对齐方法已被提出但不够完善**：DRCT、AlignedForensics 等方法通过扩散重建或 VAE 重建使合成图像在像素级与真实图像对齐，试图减少内容偏置。

**频域偏置被忽视**：作者发现即使经过像素级对齐，重建图像在频域仍存在显著差异——VAE 重建会恢复真实 JPEG 图像中因压缩丢失的高频细节，使合成图像高频能量远强于真实图像，形成新的虚假线索。

**频域偏置可被检测器利用**：实验表明频域检测器 SAFE 可以93%的成功率检测出视觉上几乎完全相同的 VAE 重建图像，但只要轻微遮蔽高频信息，检测率就骤降，说明检测器确实过拟合于高频偏置而非真正的伪造特征。

## 方法详解

### 整体框架

DDA 包含三个步骤：(1) VAE 重建实现像素级对齐；(2) 高频融合消除频域偏置；(3) 像素级 Mixup 进一步在像素域对齐。整个流程生成与真实图像在像素域和频域都高度对齐的合成图像作为训练数据，配合 DINOv2 + LoRA 微调的检测器实现强泛化性检测。

### 关键设计 1：VAE 重建（像素级对齐）

- **功能**：使用 Stable Diffusion 的 VAE 编解码器对真实图像进行 $\hat{x} = \text{Decoder}(\text{Encoder}(x))$ 重建，不修改隐空间，生成像素级高度相似的合成图像。
- **核心思路**：VAE 解码器是扩散生成器的最后阶段，其引入的 artifact 具有跨生成器通用性——学到这些 artifact 对应的决策边界可推广到更"远"的合成图像（如 text-to-image 生成的图像）。
- **设计动机**：相比扩散重建（修改隐空间导致语义偏移）或 text-to-image 生成（缺乏精确监督），纯 VAE 重建生成的图像与真实图像像素差最小，是最紧密的真-假图像对。

### 关键设计 2：频域对齐（JPEG 压缩匹配）

- **功能**：估计每张真实图像的 JPEG 质量因子，在训练时以50%概率对其 VAE 重建图像施加相同质量因子的 JPEG 压缩，使两者高频能量分布一致。
- **核心思路**：频域偏置的根本原因是真实图像经历了 JPEG 压缩（高频被削弱），而 VAE 重建图像未经压缩（高频保留完整）。对重建图像施加同等压缩即可消除这一差异。
- **设计动机**：消除检测器利用"高频丰富=合成"的虚假捷径，迫使其学习真正的伪造 artifact 而非压缩差异。

### 关键设计 3：像素级 Mixup

- **功能**：将真实图像与频域对齐后的合成图像进行像素混合：$x_{\text{mix}} = r_{\text{pixel}} \cdot x_{\text{real}} + (1 - r_{\text{pixel}}) \cdot x_{\text{syn}}$，其中 $r_{\text{pixel}} \sim \mathcal{U}(0, R_{\text{pixel}})$。
- **核心思路**：通过可控的像素混合进一步缩小真-假间距，让合成图像位于真实数据流形边界附近，促使模型学到更紧凑、更可迁移的决策边界。
- **设计动机**：t-SNE 可视化表明 DDA 生成的合成图像聚类中心距真实图像最近（远近顺序：DDA < VAE Rec. < Diff. Rec. < T2I），更紧的决策边界意味着更强的泛化性。

### 关键设计 4：两个新评估基准

- **DDA-COCO**：5K 真实 MSCOCO 图像 + 25K 由5种 VAE 重建并频域对齐的合成图像，用于测试检测器是否依赖伪造特征而非偏置线索。
- **EvalGEN**：包含 FLUX、GoT、Infinity、NOVA、OmniGen 五种最新生成器（含自回归模型）生成的2765张图像，评估对未见生成器的泛化能力。

## 损失函数与训练策略

- 骨干网络：DINOv2 + LoRA（rank=8）微调
- 输入分辨率：336×336，训练随机裁剪，验证中心裁剪
- 训练数据：**仅使用 MSCOCO 图像及其 DDA 对齐版本**（118K 真实 + 118K 合成）
- 训练用 VAE：SD 2.1 的 VAE
- 频域对齐概率：训练时50%概率施加 JPEG 压缩
- 所有评估使用**单一模型**，无数据集特定微调或阈值调整

## 实验关键数据

### 表1：11个基准总览对比（Balanced Accuracy）

| 方法 | 平均准确率 | 最低准确率 | 标准差特点 |
|------|-----------|-----------|-----------|
| DDA (本文) | **90.7%** | **81.4%** | 最小 (±5.3) |
| AlignedForensics | 75.0% | 53.9% | ±11.1 |
| DRCT | 70.1% | 50.6% | ±14.6 |
| C2P-CLIP | 62.1% | 38.9% | ±15.6 |
| FatFormer | 59.6% | 45.6% | ±14.6 |

DDA 平均准确率超第二名 **15.7%**，最低准确率超第二名 **27.5%**，标准差仅为其他方法的约一半。

### 表2：EvalGEN 新生成器泛化（Balanced Accuracy）

| 方法 | Flux | GoT | Infinity | NOVA | OmniGen | 平均 |
|------|------|-----|----------|------|---------|------|
| DDA | 89.9 | 99.5 | 97.8 | 99.5 | 99.5 | **97.2 ±4.2** |
| DRCT | 72.5 | 81.4 | 77.9 | 84.6 | 72.5 | 77.8 ±5.4 |
| AlignedForensics | 32.0 | 72.3 | 74.0 | 84.8 | 77.0 | 68.0 ±20.7 |
| C2P-CLIP | 8.7 | 49.6 | 35.3 | 86.4 | 14.5 | 38.9 ±31.2 |

DDA 在包含自回归模型的最新生成器上表现出极强的跨架构泛化能力。

### 其他关键结果

- **In-the-wild 数据集**：Chameleon 82.4%（第二名71.0%），WildRF 90.3%（第二名80.1%），BFree-Online 95.1%（第二名68.5%）
- **鲁棒性**：在 JPEG 60、RESIZE 2.0、BLUR 2.0 后处理下分别超第二名 10.5%、4.1%、5.7%
- **数据生成效率**：DDA 全集构建仅需 5.9h，远低于 DRCT 的 64.6h 和 B-Free 的 258.79h
- **消融实验**：$P_{\text{pixel}}$ 和 $R_{\text{pixel}}$ 在 0.2-0.8 范围内性能稳定；SD 2.1 的 VAE 效果最佳

## 亮点

1. **问题洞察深刻**：首次发现并系统验证了像素级对齐方法中被忽略的频域偏置问题，从频率角度解释了现有对齐方法的不足
2. **方案简洁有效**：DDA 仅需 VAE 重建 + JPEG 压缩匹配 + 像素 Mixup 三步，无需复杂训练流程或额外模型
3. **实验极其充分**：11个基准数据集（4个 in-the-wild）、9种对比方法、鲁棒性分析、消融实验、可视化分析，覆盖面在该领域首屈一指
4. **效率优势明显**：仅用 MSCOCO 118K 图像训练，数据构建仅 5.9h，远低于竞品
5. **贡献了两个高质量评估基准**：DDA-COCO 和 EvalGEN 填补了 AIGI 检测评估的空白

## 局限性

1. **对严重后处理场景仍有差距**：作者坦承在真实世界中图像经历大量后处理（社交媒体压缩等）时仍有提升空间
2. **智能手机 AI 增强干扰**：现代手机拍照管线内嵌 AI 增强，可能使真实照片呈现类合成 artifact，增加检测复杂度
3. **依赖 VAE 架构假设**：DDA 的泛化性建立在 VAE 是扩散生成器最终阶段的假设上，对非 VAE 架构（如纯 Transformer 生成器）的适用性需进一步验证
4. **ForenSynths 上表现不佳**：对早期 GAN（如 ProGAN、CycleGAN）生成的图像检测效果相对较弱

## 与相关工作的对比

| 对比维度 | DDA (本文) | DRCT (ICML'24) | AlignedForensics (ICLR'25) | B-Free |
|----------|-----------|----------------|--------------------------|--------|
| 对齐方式 | 像素+频域双对齐 | 扩散重建 | 纯 VAE 重建 | 扩散重建+修复 |
| 频域处理 | JPEG 压缩匹配 | 无 | 无 | 无 |
| 格式对齐 | ✓ | ✓ | ✗ | ✗ |
| 训练数据量 | 118K/118K | 118K/354K | 179K/179K | 51K/309K |
| 构建时间 | 5.9h | 64.6h | 8.73h | 258.79h |
| 11基准平均 | 90.7% | 70.1% | 75.0% | N/A |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 频域偏置的发现是重要洞察，DDA 方案虽简单但切中要害
- 实验充分度: ⭐⭐⭐⭐⭐ — 11个基准、9种对比方法、消融-鲁棒性-可视化全面覆盖
- 写作质量: ⭐⭐⭐⭐ — 动机推导清晰，图示丰富，论证链完整
- 价值: ⭐⭐⭐⭐ — 对 AIGI 检测领域的数据偏置问题给出了系统性解决方案，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[CVPR 2025\] A Bias-Free Training Paradigm for More General AI-generated Image Detection](../../CVPR2025/image_generation/a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)
- [\[NeurIPS 2025\] Is Artificial Intelligence Generated Image Detection a Solved Problem?](is_artificial_intelligence_generated_image_detection_a_solved_problem.md)
- [\[AAAI 2026\] Aggregating Diverse Cue Experts for AI-Generated Image Detection](../../AAAI2026/image_generation/aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)
- [\[ECCV 2024\] Zero-Shot Detection of AI-Generated Images](../../ECCV2024/image_generation/zero-shot_detection_of_ai-generated_images.md)

</div>

<!-- RELATED:END -->
