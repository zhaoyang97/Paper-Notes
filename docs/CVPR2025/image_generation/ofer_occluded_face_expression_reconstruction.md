---
title: >-
  [论文解读] OFER: Occluded Face Expression Reconstruction
description: >-
  [CVPR 2025][图像生成][遮挡人脸重建] OFER 使用两个条件扩散模型分别生成 FLAME 参数模型的形状和表情系数，结合一个排序网络从多个候选中选出最优形状，实现了遮挡条件下多样且真实的 3D 人脸表情重建。 领域现状：单图 3D 人脸重建是一个经典的逆问题，通常通过回归或优化 3DMM（如 FLAME）参数来…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "遮挡人脸重建"
  - "扩散模型"
  - "多假设重建"
  - "排序机制"
  - "FLAME参数模型"
---

# OFER: Occluded Face Expression Reconstruction

**会议**: CVPR 2025  
**arXiv**: [2410.21629](https://arxiv.org/abs/2410.21629)  
**代码**: [https://ofer.is.tue.mpg.de/](https://ofer.is.tue.mpg.de/)  
**领域**: 图像生成  
**关键词**: 遮挡人脸重建, 扩散模型, 多假设重建, 排序机制, FLAME参数模型

## 一句话总结

OFER 使用两个条件扩散模型分别生成 FLAME 参数模型的形状和表情系数，结合一个排序网络从多个候选中选出最优形状，实现了遮挡条件下多样且真实的 3D 人脸表情重建。

## 研究背景与动机

**领域现状**：单图 3D 人脸重建是一个经典的逆问题，通常通过回归或优化 3DMM（如 FLAME）参数来恢复几何。现有方法如 MICA、DECA、EMOCA 等在轻度遮挡下表现尚可，但多数为确定性方法，只能输出单一解。

**现有痛点**：在严重遮挡（口罩、墨镜、头发、大角度侧脸等）下，面部信息极度不完整，被遮挡区域可以对应无穷多合理的 3D 形态。确定性方法缺乏生成多样性，无法捕捉问题的多假设本质。Diverse3D 虽用 DPP 采样多解，但其生成结果常常夸张失真。

**核心矛盾**：遮挡带来的歧义性要求方法能建模输出分布而非点估计，但已有多假设方法（如基于 VAE+DPP）无法充分学习面部几何的真实分布，生成质量差。

**本文目标**：(1) 在遮挡下生成多个真实、多样的 3D 人脸；(2) 从多个形状候选中选出最优身份形状，确保跨表情的一致性。

**切入角度**：扩散模型天然能学习数据分布，适合建模多模态输出。作者观察到形状（identity）的变异远小于表情，因此可以先确定一个最优形状，再在其上叠加多样表情。

**核心 idea**：用两个 DDPM 分别生成 FLAME 的形状和表情系数，再用一个排序网络从形状候选中挑选最优解。

## 方法详解

### 整体框架

输入是一张可能有遮挡的人脸图像，输出是一组多样的 3D 人脸重建。整个流程分三步：(1) IdGen 生成 N=100 个 FLAME 形状系数候选；(2) IdRank 对候选排序选出最优身份；(3) ExpGen 生成 N 个表情系数，与选定的形状组合成最终多样重建。

### 关键设计

1. **Identity Generative Network (IdGen)**:

    - 功能：从高斯噪声出发，条件于输入图像，生成一组 FLAME 形状系数
    - 核心思路：使用 1D U-Net 结构的 DDPM，输入为 300 维噪声，条件为 ArcFace 编码的 512 维图像特征。通过 1000 步去噪生成形状参数 $S \in \mathbb{R}^{300}$。采样 N=100 个不同初始噪声即可得到 100 个形状假设。训练损失为标准 DDPM 噪声预测损失
    - 设计动机：扩散模型能学习形状参数的真实分布，不同噪声起点自然产生在遮挡区域的合理变化，避免了 DPP 等采样方法的分布建模缺陷

2. **Identity Ranking Network (IdRank)**:

    - 功能：从 IdGen 输出的 N 个形状候选中选出最匹配输入图像的最优形状
    - 核心思路：将每个形状系数通过 FLAME 解码为 mesh，去掉头后部分只保留正面顶点，减去均值后得到残差 mesh。一个 5 层 MLP 以 ArcFace+FaRL 联合编码（1024 维）为条件，对每个候选打分。训练时在线生成排序标签——计算每个候选与 GT 的 L1 距离，经 softmax 后作为排序目标，用交叉熵损失训练
    - 设计动机：实验发现 100 个候选中的最小误差已接近 SOTA，关键是如何选出来。shape 比 expression 更稳定，一个好的排序机制可以过滤掉低质量样本。这是首次在扩散模型中引入排序选择机制

3. **Expression Generative Network (ExpGen)**:

    - 功能：生成多样的 FLAME 表情系数
    - 核心思路：结构同 IdGen，但生成 50 维表情参数（FLAME 的前 50 个表情成分），条件编码为 ArcFace+FaRL 联合特征（1024 维）。推理时生成 N 个表情候选，与 IdRank 选出的形状组合
    - 设计动机：表情在遮挡下歧义更大，适合用扩散模型建模多种可能性；将形状和表情解耦后可以保持身份一致性

### 损失函数 / 训练策略

- IdGen 训练损失：标准 DDPM L1 噪声预测损失
- IdRank 训练损失：预测排序分布与 GT 距离排序分布之间的交叉熵
- ExpGen 训练损失：同 IdGen 的 DDPM 损失
- 三个网络独立训练，IdRank 训练时冻结 IdGen 梯度，在线生成候选
- 训练数据：IdGen/IdRank 用 Stirling、FaceWarehouse、LYHM、Florence 四个 2D-3D 配对数据集；ExpGen 用 FaMoS 运动人脸数据集

## 实验关键数据

### 主实验

| 数据集 | 指标 | OFER-rank | FOCUS (MP) | MICA (8DS) | Diverse3D |
|--------|------|-----------|------------|------------|-----------|
| NoW (全集) Med | MSE↓ | **0.98** | 1.03 | 0.90 | 1.41 |
| NoW (遮挡) Med | MSE↓ | **1.01** | 1.08 | N/A | N/A |
| CO-545 (遮挡) | SE RMSE↓ | **1.95** | - | - | 2.16 |
| CO-545 | CSE RMSE↓ | **0.17** | - | - | 0.30 |

| 数据集 | 指标 | OFER | Diverse3D |
|--------|------|------|-----------|
| Erakiotan (mask) | STD-S↑ | **34.04** | 11.81 |
| Erakiotan (sunglasses) | STD-S↑ | **34.38** | 21.28 |
| Erakiotan (mask) | ODE↓ | **0.002** | 0.95 |

### 消融实验

| 配置 | Ranked Med↓ | Avg Med↓ | Ideal Min Med↓ |
|------|------------|---------|----------------|
| OFER (100 samples) | **0.98** | 1.02 | 0.81 |
| FLAME(50)+OFER(50) | 1.08 | 1.33 | 0.84 |
| FLAME(80)+OFER(20) | 1.34 | 1.60 | 0.86 |
| Random FLAME(1000) | N/A | 1.75 | 0.90 |

### 关键发现

- IdGen 生成的候选搜索空间明显优于随机 FLAME 采样（ideal min: 0.81 vs 0.90），说明扩散模型确实学到了更好的形状分布
- 排序机制能有效从候选中选出低误差样本，Ranked 结果（0.98）远优于平均（1.02）且接近理想最小（0.81）
- OFER 在遮挡区域生成的表情多样性（STD-S）远超 Diverse3D（34+ vs 11-21），且几乎无分布外异常（ODE ≈ 0）
- 在 NoW 遮挡子集上，OFER-rank 超越专门设计的遮挡方法 FOCUS

## 亮点与洞察

- **形状-表情解耦加排序**的设计非常巧妙：利用了"身份比表情在遮挡下更确定"的先验，先锁定形状再生成多样表情，既保证了一致性又保留了多样性
- **首次将排序机制引入扩散模型**的样本选择流程，借鉴信息检索的 listwise ranking 思想，对其他需要从多样本中选优的生成任务有通用价值
- 排序网络的去均值残差输入设计减少冗余，加速收敛
- CO-545 数据集填补了遮挡人脸重建定量评估的空白

## 局限与展望

- 训练数据量有限（仅数千对 2D-3D 配对），表情生成可能缺少精细细节
- 排序网络使用 softmax 在候选数量大时分数区分度下降，未必能选出绝对最优
- 推理需生成 100 个候选并逐一排序，计算开销较大
- 排序网络固定训练在 N=100 样本上，无法直接迁移到其他数量
- 未来方向：(1) 将排序作为扩散训练的反馈信号，在生成时就约束质量；(2) 融合 2D 和 3D 监督扩大训练数据；(3) 更快的采样策略（如 DDIM）

## 相关工作与启发

- **vs MICA**: MICA 用 ArcFace 特征回归 FLAME 形状，性能强但确定性且不支持表情。OFER 复用了 MICA 的编码器设计思路但改为生成式
- **vs Diverse3D**: 两者都做多假设遮挡重建，但 Diverse3D 用 VAE+DPP 采样导致分布建模差，生成结果夸张；OFER 用扩散模型直接学分布 + 排序选优
- **vs EMOCA**: 确定性表情回归在遮挡下退化严重，无法提供多样假设
- RankGAN 将排序引入 GAN 生成，OFER 将此思路迁移到扩散模型中

## 评分

- 新颖性: ⭐⭐⭐⭐ 排序机制选优+扩散模型多假设的组合新颖，但各组件本身都是已有技术
- 实验充分度: ⭐⭐⭐⭐ 提出新数据集 CO-545，消融充分，多指标评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机链条流畅
- 价值: ⭐⭐⭐⭐ 排序选优思路有通用迁移价值，遮挡重建场景实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Realistic Face Reconstruction from Facial Embeddings via Diffusion Models](../../AAAI2026/image_generation/realistic_face_reconstruction_from_facial_embeddings_via_diffusion_models.md)
- [\[CVPR 2025\] SVFR: A Unified Framework for Generalized Video Face Restoration](svfr_a_unified_framework_for_generalized_video_face_restoration.md)
- [\[CVPR 2025\] FDeID-Toolbox: Face De-Identification Toolbox](fdeid-toolbox_face_de-identification_toolbox.md)
- [\[CVPR 2025\] GIF: Generative Inspiration for Face Recognition at Scale](gif_generative_inspiration_for_face_recognition_at_scale.md)
- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](osdface_one-step_diffusion_model_for_face_restoration.md)

</div>

<!-- RELATED:END -->
