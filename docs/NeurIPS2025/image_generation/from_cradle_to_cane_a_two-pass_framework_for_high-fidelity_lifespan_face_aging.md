---
title: >-
  [论文解读] From Cradle to Cane: A Two-Pass Framework for High-Fidelity Lifespan Face Aging
description: >-
  [NeurIPS 2025][图像生成][人脸老化] 提出 Cradle2Cane 两阶段人脸老化框架：第一阶段通过自适应噪声注入（AdaNI）实现精准年龄控制，第二阶段通过 SVR-ArcFace 和 Rotate-CLIP 双身份嵌入（IDEmb）强化身份一致性，在全寿命跨度（0-80岁）人脸老化中实现年龄精度与身份保持的最优平衡。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "人脸老化"
  - "扩散模型"
  - "身份保持"
  - "自适应噪声注入"
  - "SDXL-Turbo"
  - "年龄-身份权衡"
---

# From Cradle to Cane: A Two-Pass Framework for High-Fidelity Lifespan Face Aging

**会议**: NeurIPS 2025  
**arXiv**: [2506.20977](https://arxiv.org/abs/2506.20977)  
**代码**: [https://github.com/byliutao/Cradle2Cane](https://github.com/byliutao/Cradle2Cane)  
**领域**: 图像生成  
**关键词**: 人脸老化, 扩散模型, 身份保持, 自适应噪声注入, SDXL-Turbo, 年龄-身份权衡

## 一句话总结

提出 Cradle2Cane 两阶段人脸老化框架：第一阶段通过自适应噪声注入（AdaNI）实现精准年龄控制，第二阶段通过 SVR-ArcFace 和 Rotate-CLIP 双身份嵌入（IDEmb）强化身份一致性，在全寿命跨度（0-80岁）人脸老化中实现年龄精度与身份保持的最优平衡。

## 研究背景与动机

人脸老化（Face Aging）是计算机视觉中的重要任务，在娱乐、医疗、安防等领域有广泛应用。其核心目标是在保持人物身份一致的前提下，生成视觉真实的年龄变化效果。

**现有方法的核心瓶颈 — Age-ID Trade-off：**

- GAN 系方法（SAM、CUSP、Lifespan、HRFAE）和扩散模型方法（FADING、IPFE）普遍面临一个根本矛盾：年龄变化精度与身份保持的跷跷板效应
- 当年龄跨度较大时（如从20岁变到70岁），要麽年龄变化不够逼真，要麽身份信息严重丢失
- 现有方法对所有年龄跨度使用统一变换策略，忽视了一个自然规律：**小年龄差异只需微调外表，大年龄差异则需要更显著的结构和纹理变化**
- 在极端头部姿态、遮挡等 in-the-wild 场景下，现有方法常常彻底失败

**本文的核心洞察：** 在少步扩散模型（如 SDXL-Turbo）中，前向扩散过程注入的噪声水平直接控制了编辑强度 — 高噪声带来更强的年龄变化但损害身份，低噪声保持身份但限制老化效果。这启发了将年龄精度和身份保持解耦为两个独立阶段的思路。

## 方法详解

### 整体框架

Cradle2Cane 基于 SDXL-Turbo（4步生成）构建两阶段（two-pass）流水线：

1. **第一阶段（Pass 1）**：自适应噪声注入（AdaNI）— 专注年龄精度
2. **第二阶段（Pass 2）**：身份感知嵌入（IDEmb）— 专注身份保持

两个阶段端到端联合训练，第一阶段的输出图像经过再次加噪后作为第二阶段的输入。

### 关键设计一：自适应噪声注入（AdaNI）

核心思想：年龄跨度越大，需要注入的噪声越多以允许更强的编辑。

具体做法：
- 使用 CLIP 文本编码器将年龄 prompt（包含年龄和性别描述）编码为文本嵌入，通过交叉注意力引导生成
- 根据年龄变化幅度 $|\Delta \text{age}|$ 将变换分为三档，以年龄差 5 和 20 为分界点：
    - $|\Delta \text{age}| \leq 5$：注入低噪声 $z_1$，偏向身份保持
    - $5 < |\Delta \text{age}| \leq 20$：注入中噪声 $z_2$，平衡两者
    - $|\Delta \text{age}| > 20$：注入高噪声 $z_3$，偏向强老化效果
- 分界点 5 和 20 基于定量分析得出（年龄精度在这些阈值后显著下降）
- 解码后得到中间结果 $\hat{x}_b$，年龄精度高但身份保持较弱

### 关键设计二：SVR-ArcFace 嵌入

**目标**：从 ArcFace 人脸识别嵌入中去除年龄相关成分，提取纯净的身份特征。

**问题**：ArcFace 特征天然纠缠了年龄和身份信息 — 不同年龄的同一人会产生差异较大的嵌入。

**解决方案 — 奇异值重加权（SVR）**：

1. 从源图 $x_a$ 和第一阶段产生的多个不同噪声级别的老化图像中提取 ArcFace 嵌入
2. 拼接为矩阵 $U \in \mathbb{R}^{D \times (n+1)}$
3. 对 $U$ 做 SVD 分解：$U = \mathbf{U} \Sigma \mathbf{V}^T$
4. 对奇异值做非线性重加权：$\hat{\sigma}_i = \beta e^{\alpha \sigma_i} \cdot \sigma_i$
5. 重建后取第一列作为精炼身份嵌入 $\hat{u}_a$

**直觉**：同一人的多个年龄变体，其主要奇异值编码的是共享的身份信息，次要奇异值编码年龄差异。指数重加权放大主要成分、抑制次要成分，从而实现身份-年龄解纠缠。

### 关键设计三：Rotate-CLIP 嵌入

**目标**：在 CLIP 空间中将源图像的年龄语义平滑地移向目标年龄，同时保留其他身份信息。

**方法**：

1. 提取源图像的 CLIP 图像嵌入 $i_a$ 和源/目标年龄的文本嵌入 $t_a, t_b$
2. 使用球面线性插值（slerp）替代简单向量减法计算年龄偏移向量：$\Delta' = \text{slerp}(t_b, t_a, \lambda)$
3. 得到 Rotate-CLIP 嵌入：$\hat{i}_a = i_a + \Delta'$

**为何不用简单减法**：CLIP 对年龄的表示比较粗粒度，直接减法可能引入语义不一致，slerp 提供更平滑的语义过渡。

两个嵌入分别经过 MLP 投影后拼接，注入 SDXL-Turbo 的交叉注意力模块。

### 损失函数

总损失由三部分组成：$\mathcal{L}_{total} = \mathcal{L}_{id} + \mathcal{L}_{age} + \mathcal{L}_{per}$

| 损失 | 组成 | 作用 |
|------|------|------|
| 身份损失 $\mathcal{L}_{id}$ | MS-SSIM（结构相似性）+ ArcFace 余弦距离 | 保持源图与生成图的身份一致 |
| 年龄损失 $\mathcal{L}_{age}$ | MiVOLO 特征余弦距离 + 预测年龄 L2 误差 | 确保年龄变换精度 |
| 质量损失 $\mathcal{L}_{per}$ | LPIPS 感知距离 + GAN 对抗损失 | 提升感知质量和真实感 |

训练时冻结 ArcFace 和 CLIP 编码器，仅优化 MLP 和 UNet-LoRA 模块。

## 实验关键数据

### 主实验（Face++ 评估 + Qwen-VL 评估，CelebA-HQ）

| 方法 | 类型 | Age Diff.↓ | ID Sim.↑ | Img Quality↑ | HCS↑ | 推理时间(s) | 训练数据 |
|------|------|-----------|----------|-------------|------|------------|---------|
| Lifespan | GAN | ±22.07 | 79.80 | 66.68 | 57.40 | 0.95 | 70K |
| HRFAE | GAN | ±15.12 | 94.32 | 62.28 | 74.95 | 0.17 | 300K |
| SAM | GAN | ±8.42 | 81.96 | 68.38 | 80.42 | 0.39 | 70K |
| CUSP | GAN | ±9.59 | 85.92 | 64.98 | 80.67 | 0.24 | 30K |
| FADING | Diffusion | ±14.47 | 86.70 | 64.65 | 73.52 | 61.26 | - |
| IPFE | Diffusion | ±11.95 | 75.14 | 63.55 | 72.54 | 8.84 | - |
| **Cradle2Cane** | **Diffusion** | **±7.47** | 81.34 | **72.69** | **81.33** | **0.56** | **10K** |

### 消融实验（Qwen-VL 评估）

| AdaNI | SVR-ArcFace | Rotate-CLIP | Age Diff.↓ | ID Sim.↑ | HCS↑ |
|-------|-------------|-------------|-----------|----------|------|
| ✗ | ✗ | ✗ | ±8.87 | 68.92 | 73.10 |
| ✓ | ✗ | ✗ | ±3.94 | 59.70 | 71.83 |
| ✗ | ✓ | ✗ | ±9.48 | 70.17 | 73.11 |
| ✓ | ✓ | ✗ | ±6.75 | 63.38 | 71.92 |
| ✓ | ✓ | ✓ | **±4.62** | **70.29** | **78.33** |

### 关键发现

1. **年龄精度最优**：Cradle2Cane 的 Age Diff. 在 Face++ 和 Qwen-VL 评估中均最低（±7.47 / ±4.62）
2. **推理速度极快**：0.56s，比 FADING（61.26s）快 100+ 倍，与 GAN 方法相当
3. **训练数据高效**：仅需 10K 训练数据，远少于 HRFAE（300K）和 Lifespan（70K）
4. **综合指标 HCS 最优**：在 Face++ 评估中 HCS 达 81.33，Qwen-VL 达 78.33
5. **消融显示各组件互补**：AdaNI 大幅降低年龄误差（8.87→3.94），SVR-ArcFace 提升身份一致性（59.70→63.38），Rotate-CLIP 进一步将 HCS 从 71.92 提升至 78.33
6. **Wild 场景鲁棒**：在 in-the-wild 数据上 HCS 达 75.94，领先 CUSP（70.94）和 FADING（75.06）
7. **用户研究**：50名志愿者在与 SAM 和 CUSP 的1v1对比中，明显偏好 Cradle2Cane 的结果

## 亮点与洞察

1. **问题定义精准**：将人脸老化的核心矛盾清晰定义为 Age-ID Trade-off，并通过定量分析（60个年龄偏移值 × 100张人脸的 trade-off 曲线）提供了坚实的实证支撑
2. **解耦思路优雅**：将年龄精度和身份保持分配给两个独立阶段，每个阶段只负责一个目标，比端到端统一处理更易优化
3. **自适应噪声注入符合直觉**：大年龄差异需要更多编辑自由度（更多噪声），小年龄差异只需微调（更少噪声）— 这与人类对老化过程的认知高度一致
4. **SVR 去纠缠思路新颖**：利用同一人不同年龄变体的 ArcFace 嵌入做 SVD，通过奇异值重加权放大身份共性、抑制年龄差异性
5. **训练效率突出**：仅 10K 数据 + LoRA fine-tuning 即达到 SOTA，推理速度与 GAN 相当

## 局限性

1. **身份相似度非最优**：虽然 HCS 综合最高，但 ID Sim. 单项指标（81.34）低于 HRFAE（94.32）和 CUSP（85.92），说明身份保持仍有提升空间
2. **三档噪声分区略显粗糙**：以固定阈值 5 和 20 划分三档，可能不是最优的 — 连续自适应噪声调度是否更好？
3. **评估数据集单一**：主要在 CelebA-HQ 上评估，缺少 MORPH、CACD 等常用人脸老化数据集的对比
4. **年龄估计器依赖**：训练依赖 MiVOLO 年龄估计器的准确性，估计器本身的偏差会传导到生成结果
5. **缺少跨种族/跨文化分析**：不同种族的老化模式存在差异，论文未讨论模型在多样化人群上的公平性

## 相关工作与启发

- **与 FADING 的区别**：FADING 基于标准 LDM 的 NTI 逆生成，推理需 61s；本文基于 SDXL-Turbo 少步模型，0.56s 完成推理，且通过两阶段解耦实现更好的 Age-ID 平衡
- **与 SAM 的区别**：SAM 在 StyleGAN2 潜空间中做年龄变换，依赖 GAN 的潜空间结构；本文利用扩散模型的噪声控制灵活性，实现自适应编辑强度
- **启发**：两阶段解耦思路可推广到其他存在多目标冲突的图像编辑任务（如风格-内容权衡、编辑强度-保真度权衡）；SVR 去纠缠技术可用于任何需要从嵌入中分离共享/差异属性的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 两阶段解耦 + AdaNI + SVR-ArcFace 组合设计有新意
- 实验充分度: ⭐⭐⭐⭐ — 双评估协议 + 消融 + 用户研究，但数据集偏少
- 写作质量: ⭐⭐⭐⭐ — 动机阐述清晰，trade-off 分析图表设计好
- 价值: ⭐⭐⭐⭐ — 解决了人脸老化的核心矛盾，实用性强（快速推理 + 少量训练数据）
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](../../CVPR2026/image_generation/high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [\[CVPR 2025\] SVFR: A Unified Framework for Generalized Video Face Restoration](../../CVPR2025/image_generation/svfr_a_unified_framework_for_generalized_video_face_restoration.md)
- [\[CVPR 2026\] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](../../CVPR2026/image_generation/preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)
- [\[CVPR 2025\] GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing](../../CVPR2025/image_generation/glyphmastero_a_glyph_encoder_for_high-fidelity_scene_text_editing.md)
- [\[NeurIPS 2025\] One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting](one_stone_with_two_birds_a_null-text-null_frequency-aware_diffusion_models_for_t.md)

</div>

<!-- RELATED:END -->
