---
title: >-
  [论文解读] From Pixels to Views: Learning Angular-Aware and Physics-Consistent Representations for Light Field Microscopy
description: >-
  [NeurIPS 2025][3D视觉][Light Field Microscopy] 提出 XLFM-Former，通过 **视角级 Masked View Modeling（MVM-LF）** 自监督预训练学习 XLFM 的角度–空间先验，并设计基于 PSF 可微渲染的 **光学渲染一致性损失（ORC Loss）** 约束重建体积的物理合理性，在自建的首个 XLFM-Zebrafish 标准化基准上，平均 PSNR 达 54.04 dB，较最佳基线 ConvNeXt（50.16 dB）提升 **7.7%**。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "Light Field Microscopy"
  - "XLFM"
  - "Masked View Modeling"
  - "ORC Loss"
  - "PSF"
  - "自监督学习"
---

# From Pixels to Views: Learning Angular-Aware and Physics-Consistent Representations for Light Field Microscopy

**会议**: NeurIPS 2025  
**arXiv**: [2510.22577](https://arxiv.org/abs/2510.22577)  
**代码**: [https://github.com/hefengcs/XLFM-Former](https://github.com/hefengcs/XLFM-Former)  
**作者**: Feng He, Guodong Tan, Qiankun Li, Jun Yu, Quan Wen (中国科学技术大学)
**领域**: 3D 视觉 — 光场显微镜 3D 重建  
**关键词**: Light Field Microscopy, XLFM, Masked View Modeling, ORC Loss, PSF, Self-supervised Pretraining

## 一句话总结

提出 XLFM-Former，通过 **视角级 Masked View Modeling（MVM-LF）** 自监督预训练学习 XLFM 的角度–空间先验，并设计基于 PSF 可微渲染的 **光学渲染一致性损失（ORC Loss）** 约束重建体积的物理合理性，在自建的首个 XLFM-Zebrafish 标准化基准上，平均 PSNR 达 54.04 dB，较最佳基线 ConvNeXt（50.16 dB）提升 **7.7%**。

## 研究背景与动机

- **领域现状**: 扩展光场显微镜（XLFM）可在 100 Hz 下单次曝光实现完整光场采集，是神经科学中大规模活体体积成像的关键工具（斑马鱼、小鼠）。但基于深度学习的 XLFM 3D 重建严重滞后——既缺乏标准化数据集和可复现的评估协议，也缺少能高效建模角度–空间结构且物理接地的方法。
- **现有痛点**: ① 每帧 XLFM 数据通过微透镜阵列编码密集角度采样的 3D 场景，产生高度纠缠的多视角观测，传统 CNN 难以建模视角间依赖；② 高质量体积 GT（通过 Richardson-Lucy 反卷积生成）计算代价昂贵，大规模监督学习成本高；③ 现有方法（XLFMNet、FNet）要么仅重建稀疏神经信号忽略完整形态、要么 Fourier 域卷积导致显存爆炸需多 GPU。
- **核心矛盾**: XLFM 原始采集廉价且量大，但标注昂贵且缺标准——数据丰富但标签稀缺。纯像素级损失训练的模型可能生成视觉合理但光学不一致的"幻觉"结构，损害科学可信性。
- **本文目标**: 如何在标签稀缺条件下，高效学习 XLFM 的角度先验并保证重建体积的物理一致性？
- **切入角度**: 将 XLFM 重建重新定义为 **结构化预测问题**——以"视角"而非"像素"为原子建模单元进行自监督预训练，并引入基于已知 PSF 的可微前向渲染约束。
- **核心 idea**: 视角级掩码自监督预训练学角度先验 + PSF 可微渲染损失保物理一致性 = 数据高效且物理可信的 XLFM 全体积重建。

## 方法详解

### 整体框架

XLFM-Former 采用 **Swin Transformer 编码器 + CNN 解码器** 的层次化架构，用于从 XLFM 光场数据渐进式重建 3D 体积。整体流程分两阶段：

1. **预训练阶段**：在 XLFM 无标注光场数据上执行 MVM-LF 自监督预训练，对 27 个子孔径视角随机遮挡 70% 并预测被遮挡视角，用轻量 CNN 解码器 + $\ell_2$ 损失训练 250 epochs
2. **微调阶段**：丢弃预训练解码器，保留编码器权重初始化 XLFM-Former，用多 loss 组合（MS-SSIM + Edge + PSNR + MSE + ORC）进行有监督全体积重建

### 关键设计

1. **Masked View Modeling for Light Fields（MVM-LF，视角级掩码自监督预训练）**

    **功能**：让编码器在无标注条件下学习 XLFM 的角度先验和视角间依赖关系，提升数据效率和特征泛化性。

    **核心思路**：从 XLFM 的 27 个子孔径视角 $\mathcal{U} = \{U_1, U_2, \dots, U_{27}\}$ 中，以 $r_m = 0.7$（70%）的比例随机抽取子集 $\mathcal{U}_{\text{mask}}$，将对应视角零填充但保留位置信息，训练模型从未遮挡视角重建被遮挡视角：$\hat{\mathcal{U}}_{\text{mask}} = f_\theta(\mathcal{U} \setminus \mathcal{U}_{\text{mask}})$，损失为 MSE：$\mathcal{L}_{\text{MVM-LF}} = \sum_{U_i \in \mathcal{U}_{\text{mask}}} \|U_i - \hat{U}_i\|_2^2$。

    **设计动机**：XLFM 视角不独立——它们展现遮挡模式、空间冗余和角度连续性，类似自然语言或多视角立体系统中的依赖关系。以"视角"（而非像素）为掩码单元，与 XLFM 物理采样结构天然匹配，迫使模型学习全局场景结构和视角间的角度相关性。相比像素级 MAE 预训练，视角级掩码能捕获光场数据的本质结构（实验证明 MVM-LF 比 Random mask 高 1.07 dB）。

2. **Optical Rendering Consistency Loss（ORC Loss，光学渲染一致性损失）**

    **功能**：确保重建的 3D 体积不仅在结构上匹配 GT，还在 XLFM 成像前向模型下保持物理一致（光学可信）。

    **核心思路**：将预测体积 $\mathcal{V}_{\text{pred}}$ 和 GT 体积 $\mathcal{V}_{\text{GT}}$ 分别通过已知的系统 PSF（点扩散函数）进行 3D 卷积前向渲染，得到合成光场图像 $\mathbf{I}_{\text{pred}} = h * \mathcal{V}_{\text{pred}}$ 和 $\mathbf{I}_{\text{GT}} = h * \mathcal{V}_{\text{GT}}$，最小化二者的 MSE：$\mathcal{L}_{\text{ORC}} = \|h * \mathcal{V}_{\text{pred}} - h * \mathcal{V}_{\text{GT}}\|_2^2$。

    **设计动机**：纯像素级损失（MSE/SSIM）只约束体积空间的逐点匹配，不保证重建体积在光学前向模型下的一致性。直接用原始测量做约束不可行——原始数据含传感器噪声、暗电流和散射伪影，会引入非物理梯度。ORC Loss 以 GT 体积的 PSF 前向投影为干净监督信号，桥接数据驱动学习与波光学一致性。实验还验证了 ORC Loss 对 PSF 误差具有鲁棒性（±10% FWHM 扰动仅 ±0.12 dB 波动）。

3. **XLFM-Zebrafish 标准化基准数据集**

    **功能**：填补 XLFM 领域缺乏标准化数据集和可复现评估协议的空白，为系统性推进该方向提供基础设施。

    **核心思路**：收集 22,581 张光场图像，覆盖 3 条自由游泳斑马鱼 + 13 条固定斑马鱼（训练/验证 7 条 + 测试 6 条 unseen），设置双采样率（10 fps 高时间分辨 + 1 fps 长期跟踪），提供标准化的训练/测试划分和评估流程。

    **设计动机**：此前 XLFM 重建领域的方法间比较往往是轶闻式的（anecdotal），缺乏可复现基准导致进展碎片化。不同运动状态（自由游泳 vs 固定）和采样条件的组合确保了数据多样性，unseen 测试鱼保证泛化评估的可信度。

### 损失函数 / 训练策略

**预训练阶段**：仅用 $\ell_2$ 损失，batch size = 8，初始 lr = 1e-4 + ReduceLROnPlateau 调度器，训练 250 epochs，4×A100-80GB GPU。

**微调阶段**：多损失组合：

$$\mathcal{L}_{\text{total}} = \frac{1}{\lambda_1}\mathcal{L}_{\text{MS\_SSIM}} + \frac{1}{\lambda_2}\mathcal{L}_{\text{Edge}} + \frac{1}{\lambda_3}\mathcal{L}_{\text{PSNR}} + \frac{1}{\lambda_4}\mathcal{L}_{\text{MSE}} + \frac{1}{\lambda_5}\mathcal{L}_{\text{ORC}}$$

batch size = 1（体积重建需求），继承预训练编码器权重。五项损失分别约束结构相似性、边缘锐利度、峰值信噪比、逐点误差和物理一致性。

## 实验关键数据

### 主实验

XLFM-Zebrafish 测试集（6 个 unseen 样本），各方法使用相同 Swin-XLFM 或对应标准架构：

| 方法 | #1 PSNR | #2 PSNR | #3 PSNR | #4 PSNR | #5 PSNR | #6 PSNR | Avg PSNR↑ | Avg SSIM↑ |
|------|---------|---------|---------|---------|---------|---------|-----------|-----------|
| ConvNeXt | 49.48 | 53.88 | 44.87 | 51.38 | 51.52 | 49.79 | 50.16 | 0.9876 |
| ViT | 49.38 | 52.67 | 45.29 | 51.09 | 51.35 | 45.90 | 49.28 | 0.9876 |
| PVT | 47.21 | 47.93 | 44.50 | 49.46 | 48.32 | 46.60 | 47.34 | 0.9829 |
| EfficientNet | 45.04 | 54.68 | 42.13 | 49.56 | 48.63 | 27.16 | 44.53 | 0.9296 |
| ResNet-50 | 46.46 | 54.89 | 41.46 | 49.47 | 48.82 | 39.98 | 46.85 | 0.9634 |
| ResNet-101 | 47.20 | 54.90 | 41.33 | 49.47 | 49.09 | 39.50 | 46.91 | 0.9554 |
| U-Net | 48.81 | 57.23 | 44.41 | 52.61 | 52.06 | 41.47 | 49.43 | 0.9847 |
| **XLFM-Former** | **53.97** | **59.83** | **49.31** | **54.55** | **54.65** | **51.95** | **54.04** | **0.9944** |

XLFM-Former 在全部 6 个测试样本上均取得最高 PSNR 和 SSIM，平均 PSNR 超越第二名 ConvNeXt **3.88 dB**（7.7%），SSIM 提升 0.0068。

### 消融实验

| 配置 | 说明 | PSNR↑ | SSIM↑ |
|------|------|-------|-------|
| Baseline | 无 ORC Loss, 无 MVM-LF | 52.14 | 0.9924 |
| + ORC Loss only | 仅加物理损失 | 52.96 | 0.9931 |
| + MVM-LF only | 仅加视角预训练 | 53.38 | 0.9938 |
| **Full (Ours)** | **ORC + MVM-LF** | **54.04** | **0.9944** |
| ImageNet-1K 预训练 | 视觉领域权重 | 52.70 | 0.9931 |
| ImageNet-22K 预训练 | 大规模视觉权重 | 52.38 | 0.9923 |
| Random mask 预训练 | 像素级 MAE | 52.97 | 0.9934 |
| MAE (ViT backbone) | 标准 MAE | 46.55 | 0.9752 |
| **MVM-LF 预训练** | **视角级掩码** | **54.04** | **0.9944** |

视角缺失鲁棒性（使用 MVM-LF 预训练）：

| 可用视角比例 | PSNR↑ | SSIM↑ |
|-------------|-------|-------|
| 100%（scratch, 无预训练） | 52.14 | 0.9924 |
| 90%（w/ MVM-LF） | 52.97 | 0.9933 |
| 80% | 53.26 | 0.9936 |
| 70% | 52.67 | 0.9928 |
| 60% | 52.54 | 0.9928 |

### 关键发现

1. **ORC Loss 和 MVM-LF 互补性强**：单独使用各提升 +0.82 dB 和 +1.24 dB，组合后提升 +1.90 dB，接近累加效果，说明二者约束了不同维度（物理一致性 vs 角度先验）。
2. **视角级 > 像素级掩码**：MVM-LF（54.04）比 Random mask（52.97）高 1.07 dB，比 ImageNet-1K/22K 预训练高 1.34/1.66 dB，证明任务特异性预训练的必要性。标准 MAE 仅 46.55 dB，严重不适配 XLFM 数据。
3. **预训练显著提升数据效率**：仅 10% 标注数据 + MVM-LF 预训练（51.92 dB）即超过 100% 标注 scratch（52.14 dB 左右），在低标注场景优势尤为明显。
4. **缺失视角鲁棒性**：预训练模型在仅 60% 视角输入下（52.54 dB）仍超过 scratch 模型的 100% 视角（52.14 dB），说明 MVM-LF 使模型学会了从不完整视角推断全局结构。
5. **ORC Loss 对 PSF 误差鲁棒**：PSF 的 FWHM ±10% 扰动仅导致 ±0.12 dB 波动，可容忍实际成像系统中的轻微校准偏差。
6. **跨域泛化**：在 H2B-Nemos 数据集上，XLFM-Former 的零样本推理（53.72 dB）超过 ResNet-101 有监督基线（51.42 dB）+2.29 dB，体现强跨域能力。

## 亮点与洞察

1. **"视角是原子单元"的认知飞跃**：论文的核心洞察在于将光场数据的建模粒度从像素提升到视角——XLFM 的 27 个子孔径视角类似于多视角立体系统的视图，它们之间的遮挡、冗余和角度连续性构成了结构化依赖，这一认知催生了 MVM-LF 设计，比通用的 MAE 策略更匹配光场数据的本质。

2. **可微渲染约束的科学严谨性**：ORC Loss 不只是一个额外正则项——它将 XLFM 的成像物理（PSF 前向模型）显式嵌入学习过程，使网络被约束在物理可行解空间内，这对科学成像至关重要（重建不能只好看，还必须光学自洽）。巧妙之处在于用 GT 的 PSF 投影替代噪声原始测量作为监督，避免了非物理梯度。

3. **基准数据集的社区贡献**：作为首个标准化 XLFM 基准，XLFM-Zebrafish 不仅支撑了本文实验，更为整个领域提供了可复现评估的基础设施，这类"铺路"工作的长期价值不可低估。

4. **跨域零样本的惊喜结果**：XLFM-Former 在 H2B-Nemos 数据集上的零样本（53.72 dB）甚至超过同数据集上有监督微调（52.34 dB），说明 MVM-LF 预训练学到的角度先验具有强迁移性。

5. **全体积重建 vs 稀疏信号提取**：区别于 XLFMNet 等仅重建稀疏神经信号的方法，XLFM-Former 重建完整体积结构（功能信号 + 形态信息），对需要同时分析神经活动和解剖结构的生物学应用意义重大。

## 局限与展望

1. **生物多样性有限**：仅在斑马鱼（幼体）上验证，未涉及小鼠、果蝇等更大或更复杂组织，XLFM 在不同生物上的 PSF 特性和成像条件差异可能影响泛化。

2. **计算资源需求高**：训练需 4×A100-80GB GPU，体积重建的 batch size 仅为 1，限制了方法的可及性和在更大规模数据上的扩展。

3. **未评估功能性追踪提取**：论文聚焦 3D 体积重建质量（PSNR/SSIM），但神经科学最终关心的是神经活动轨迹提取准确率——从重建体积到功能性分析的 pipeline（配准→分割→聚类→轨迹提取）未被端到端评估。

4. **ORC Loss 对 PSF 的依赖**：虽然对 ±10% FWHM 扰动鲁棒，但未探索更大 PSF 偏差或完全未知 PSF 的场景——实际部署中光学系统老化、温度变化等可能导致更大偏移。

5. **预训练策略的超参数**：70% 掩码率通过实验搜索确定，不同成像系统/样本类型是否需要重新搜索未讨论；预训练与微调的最优 epoch 比例也缺乏理论指导。

## 相关工作与启发

| 方法 | 核心技术 | 重建类型 | 全体积? | 自监督? | 物理约束? | 核心局限 |
|------|---------|---------|--------|---------|---------|---------|
| XLFMNet | SLNet + XLFMNet 稀疏分解 | 稀疏神经信号 | ✗ | ✗ | ✗ | 忽略完整形态结构 |
| CWFA | 条件归一化流 | 稀疏活动 | ✗ | ✗ | ✗ | 同上，仅神经信号 |
| FNet | Fourier 全局卷积 | 全体积 | ✓ | ✗ | ✓（端到端） | 显存爆炸，需多 GPU |
| MLFM | Transformer + 像素级掩码 | 光场超分 | - | ✓ | ✗ | 像素级掩码不匹配视角结构 |
| MAE | ViT + 随机 patch 掩码 | 通用视觉 | - | ✓ | ✗ | 不适配光场数据（46.55 dB） |
| **XLFM-Former** | **Swin-T + MVM-LF + ORC** | **全体积** | **✓** | **✓** | **✓** | 仅斑马鱼验证 |

**启发方向**：

- MVM-LF 的视角级自监督范式可推广到其他多视角成像系统：光场相机、多视角 CT、NeRF 数据采集中的稀疏视角补全
- ORC Loss 的"已知前向模型 + 可微渲染约束"框架适用于任何逆问题——天文成像、MRI 重建、超声成像等只要前向模型可微即可复用
- "数据廉价但标注昂贵"的场景下，视角/视图级自监督优于通用 MAE 的发现，启示科学成像领域应设计与物理采样结构对齐的预训练任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 视角级掩码自监督预训练是光场领域的新范式，ORC Loss 的 PSF 可微渲染约束设计精巧；但各组件（Swin-T / MAE / 可微渲染）本身非全新概念，创新在组合与适配
- 实验充分度: ⭐⭐⭐⭐⭐ 首个标准化基准 + 7 个 SOTA 架构对比 + 详细消融（组件/预训练策略/掩码率/标注比例/缺失视角/PSF 鲁棒性）+ 跨域泛化（H2B-Nemos） + 定性可视化，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 四个设计 insight 驱动的叙事结构清晰，物理动机与方法设计对应关系明确；部分架构细节放补充材料略有不便
- 价值: ⭐⭐⭐⭐ 对计算神经科学社区有重要基础设施价值（数据集 + 基准 + 方法），全体积重建 + 物理一致性对实际科学应用至关重要；受众相对小众，但在该细分领域影响力大
---
title: >-
  [论文解读] From Pixels to Views: Learning Angular-Aware and Physics-Consistent Representations for Light Field Microscopy
description: >-
  [NEURIPS2025][3D视觉][光场显微镜] 提出XLFM-Former用于扩展光场显微镜(XLFM)的3D重建：构建首个XLFM-Zebrafish标准化基准，设计Masked View Modeling (MVM-LF)自监督预训练学习角度先验，引入光学渲染一致性损失(ORC Loss)确保物理可信性，PSNR较SOTA提升7.7%（54.04 vs 50.16 dB）。
tags:
  - NEURIPS2025
  - 3D视觉
  - 光场显微镜
  - XLFM
  - 3D重建
  - Masked View Modeling
  - 物理一致性
---


<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)
- [\[CVPR 2025\] ProbeSDF: Light Field Probes for Neural Surface Reconstruction](../../CVPR2025/3d_vision/probesdf_light_field_probes_for_neural_surface_reconstruction.md)
- [\[NeurIPS 2025\] MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)
- [\[ICML 2025\] PhysicsNeRF: Physics-Guided 3D Reconstruction from Sparse Views](../../ICML2025/3d_vision/physicsnerf_physics-guided_3d_reconstruction_from_sparse_views.md)
- [\[ICLR 2026\] LiTo: Surface Light Field Tokenization](../../ICLR2026/3d_vision/lito_surface_light_field_tokenization.md)

</div>

<!-- RELATED:END -->
