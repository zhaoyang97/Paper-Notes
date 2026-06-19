---
title: >-
  [论文解读] CardiacNet: Learning to Reconstruct Abnormalities for Cardiac Disease Assessment from Echocardiogram Videos
description: >-
  [ECCV2024][医学图像][echocardiogram] 提出基于重建的心脏疾病评估框架 CardiacNet，通过 Consistency Deformation Codebook (CDC) 和 Consistency Deformation Discriminator (CDD) 学习正常与异常心脏超声视频之间的结构和运动差异，在射血分数预测、肺动脉高压和房间隔缺损分类三个任务上达到 SOTA。
tags:
  - "ECCV2024"
  - "医学图像"
  - "echocardiogram"
  - "cardiac disease assessment"
  - "video reconstruction"
  - "量化"
  - "optimal transport"
---

# CardiacNet: Learning to Reconstruct Abnormalities for Cardiac Disease Assessment from Echocardiogram Videos

**会议**: ECCV2024  
**arXiv**: [2410.20769](https://arxiv.org/abs/2410.20769)  
**代码**: [xmed-lab/CardiacNet](https://github.com/xmed-lab/CardiacNet)  
**领域**: 医学图像  
**关键词**: echocardiogram, cardiac disease assessment, video reconstruction, vector quantization, optimal transport

## 一句话总结

提出基于重建的心脏疾病评估框架 CardiacNet，通过 Consistency Deformation Codebook (CDC) 和 Consistency Deformation Discriminator (CDD) 学习正常与异常心脏超声视频之间的结构和运动差异，在射血分数预测、肺动脉高压和房间隔缺损分类三个任务上达到 SOTA。

## 背景与动机

超声心动图视频是心脏医学中最常用的成像模式，可用于评估多种心脏疾病。现有方法（如 EchoNet）主要利用全局时空特征进行分类/回归，但忽略了心脏局部结构的周期性运动特征。作者观察到心脏疾病存在两类异常模式：

1. **局部结构异常**：单帧可见的明显形态异常，如房间隔缺损（ASD）中心房间隔上的孔洞
2. **心脏运动异常**：单帧不易察觉，但通过视频中局部心脏结构的运动异常可以检测，如肺动脉高压（PAH）

现有基于分类的方法关注全局信息，难以捕捉局部表示；现有基于重建的方法主要针对 CT/MRI/X-ray 模态，处理超声心动图中复杂的时空异常效果不佳。

## 核心问题

如何从超声心动图视频中同时学习局部心脏结构异常和运动异常的表征，以实现对多种心脏疾病的准确评估？核心假设：如果模型能准确地从正常样本重建出异常样本（反之亦然），则它能更好地理解疾病在局部结构和运动变化上的特征。

## 方法详解

### 整体架构：双向重建网络

CardiacNet 包含两个独立网络 $\phi^A(\cdot)$ 和 $\phi^B(\cdot)$，分别负责"正常→异常"和"异常→正常"的重建过程。输入超声视频先分割为 patch 并随机 mask，经编码器-codebook-解码器管线重建为目标类别视频，再通过反向网络重建回原始类别。使用 L1 重建损失监督：

$$\mathcal{L}_{\text{recon}}(X, X^R) = \|X - X^R\|_1$$

### Consistency Deformation Codebook (CDC)

CDC 的核心思想是通过 vector quantization 方式离散化连续特征，以区域化方式建模心脏结构的形变模式：

- **一致性形变编码**：编码器提取特征 $F$ 后，通过可学习 codebook $\mathcal{Z}=\{Z_k\}_{k=1}^K$ 进行向量量化。为保持时序一致性，在时间维度添加可学习位置编码 $\mathcal{P}$，量化过程为最近邻匹配
- **最优传输距离优化**：为区分两个网络学到的正常/异常分布，构建 memory bank $\mathcal{M}^A$、$\mathcal{M}^B$ 存储编码特征，使用 Wasserstein 距离（Sinkhorn 迭代）最大化两个分布间的传输距离 $\mathcal{L}_{\text{OT}}$，同时最小化量化特征与对应 memory bank 质心的距离 $\mathcal{L}_{\text{dis}}$
- Codebook 通过 EMA 更新，更新权重 $\omega=0.01$

### Consistency Deformation Discriminator (CDD)

CDD 包含两个判别器确保重建结果的时空一致性：

- **时序判别器** $\eta^T(\cdot)$：以整段视频为输入，判别全局时序一致性
- **空间判别器** $\eta^S(\cdot)$：逐帧判别空间一致性
- **局部区域判别**：将视频切分为 non-overlap patches，对每个 patch 区域用时序判别器进行局部判别，确保心脏各区域的重建质量

总损失为：$\mathcal{L}_{all} = \mathcal{L}_{\text{CDC}} + \mathcal{L}_{\text{CDD}} + \mathcal{L}_{\text{recon}}(X, X^R) + \mathcal{L}_{\text{recon}}(Y, Y^R)$

### 推理阶段

冻结 $\phi^A$ 的特征提取器参数，将特征展平后接单层 Linear 进行分类或回归微调。输入 16 帧，采样间隔为 4。

## 实验关键数据

### 数据集

| 数据集 | 视频数 | 任务 |
|--------|--------|------|
| CardiacNet-PAH（自建） | 496 | PAH 分类 |
| CardiacNet-ASD（自建） | 231 | ASD 分类 |
| CAMUS | 500 | EF 回归/分类 |
| EchoNet-Dynamic | 10,300 | EF 回归/分类 |

### PAH 分类（CardiacNet-PAH）

| 方法 | AUC-ROC | ACC | FID |
|------|---------|-----|-----|
| HiFuse | 84.11% | 83.67% | - |
| EchoNet | 81.63% | 80.95% | - |
| CyTran | 72.69% | 69.38% | 16.40 |
| **CardiacNet** | **89.32%** | **85.71%** | **14.73** |

### ASD 分类（CardiacNet-ASD）

| 方法 | AUC-ROC | ACC | DICE |
|------|---------|-----|------|
| DeepGuide | 85.02% | 84.79% | - |
| CyTran | 74.35% | 72.41% | 70.21% |
| **CardiacNet** | **91.24%** | **89.63%** | **73.52%** |

### EF 预测（CAMUS / EchoNet）

| 方法 | CAMUS MAE | EchoNet MAE |
|------|-----------|-------------|
| HiFuse | 6.34 | 4.08 |
| EchoNet | 6.30 | 4.22 |
| **CardiacNet** | **5.97** | **3.83** |

### 消融实验（CardiacNet-PAH）

- CDC 单独启用：AUC 从 52.37% 提升到 80.27%，FID 从 18.90 降到 16.82
- CDD 单独启用：AUC 仅 52.46%，表明 CDD 需要 CDC 配合
- CDC 中位置编码贡献：分类准确率提升约 20%，FID 改善 1.34
- CDC 中最优传输贡献：分类准确率提升约 30%，FID 改善 0.84
- CDD 中全局+局部判别器组合达到最佳性能

## 亮点

1. **重建驱动的疾病评估范式**：不直接分类，而是通过学习正常↔异常的双向重建来理解疾病表征，思路新颖
2. **CDC 设计精巧**：vector quantization + 时序位置编码 + 最优传输距离的组合，有效捕捉心脏局部结构和运动模式
3. **自建基准数据集**：提供 PAH 和 ASD 两个新基准，填补超声心动图视频疾病评估数据集的空白
4. **跨任务泛化**：在分类（PAH、ASD）和回归（EF）三种不同任务上均达到 SOTA

## 局限与展望

1. **推理效率**：推理时间 4.523s，虽远快于扩散模型方法但仍有优化空间
2. **数据规模有限**：自建数据集仅 496+231 条视频，来自 4 家医院，泛化性有待验证
3. **双向重建计算开销**：需要训练两个独立网络，参数量和计算量翻倍
4. **缺少更多疾病类型验证**：仅验证了三种心脏疾病，是否适用于更复杂的多类别诊断场景未知
5. **分辨率限制**：训练时 resize 到 144×144 再裁剪到 112×112，可能损失细节信息

## 与相关工作的对比

- **vs EchoNet (R2+1D)**：EchoNet 使用全局时空特征，忽略局部心脏结构；CardiacNet 通过重建学习局部异常表征，PAH 分类 AUC 提升约 8%
- **vs 现有重建方法 (CyTran, Wolleb)**：它们缺乏心脏先验知识约束，直接重建超声图像质量差；CardiacNet 的 CDC+CDD 引入结构一致性约束，FID 显著更优
- **vs DiffMIC (扩散模型)**：扩散模型推理需 1000 步去噪（1182s），CardiacNet 仅需 4.5s 且性能更优
- **vs 注意力方法 (AGXNet)**：注意力机制依赖分类骨干网，易受噪声影响，定位异常区域精度不足

## 启发与关联

- 双向重建+codebook 的范式可推广到其他需要捕捉正常/异常分布差异的医学影像任务
- 最优传输距离优化 codebook 分布的想法在对比学习、域适应等场景也有借鉴价值
- 局部+全局判别器的设计对视频生成/翻译任务中保持时空一致性有参考意义

## 评分
- 新颖性: 8/10 — 重建驱动评估+CDC+最优传输的组合较新颖
- 实验充分度: 8/10 — 四个数据集、三种任务、详细消融、可视化分析
- 写作质量: 7/10 — 整体清晰但数学符号较多，部分描述冗长
- 价值: 8/10 — 提供新基准数据集和有效方法，对心脏超声AI诊断有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals](brain_netflix_scaling_data_to_reconstruct_videos_from_brain_signals.md)
- [\[NeurIPS 2025\] Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment](../../NeurIPS2025/medical_imaging/care-pd_a_multi-site_anonymized_clinical_dataset_for_parkinsons_disease_gait_ass.md)
- [\[CVPR 2025\] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](../../CVPR2025/medical_imaging/deep_learning-based_assessment_of_the_relation_between_the_third_molar_and_mandi.md)
- [\[AAAI 2026\] From Policy to Logic for Efficient and Interpretable Coverage Assessment](../../AAAI2026/medical_imaging/from_policy_to_logic_for_efficient_and_interpretable_coverage_assessment.md)
- [\[NeurIPS 2025\] Multimodal Bayesian Network for Robust Assessment of Casualties in Autonomous Triage](../../NeurIPS2025/medical_imaging/multimodal_bayesian_network_for_robust_assessment_of_casualties_in_autonomous_tr.md)

</div>

<!-- RELATED:END -->
