---
title: >-
  [论文解读] UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset
description: >-
  [NeurIPS 2025][图像生成][超高分辨率] 构建了包含 10 万张超高分辨率图像及丰富标注的 UltraHR-100K 数据集，并提出频率感知后训练方法（DOTS + SWFR），通过面向细节的时间步采样和基于 DFT 的软加权频率正则化来增强预训练 T2I 模型的超高分辨率细节生成能力。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "超高分辨率"
  - "数据集"
  - "频率感知"
  - "细节生成"
  - "后训练"
---

# UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset

**会议**: NeurIPS 2025  
**arXiv**: [2510.20661](https://arxiv.org/abs/2510.20661)  
**代码**: 有（论文中标注 code available）  
**领域**: 超高分辨率图像生成 / 扩散模型  
**关键词**: 超高分辨率, 数据集, 频率感知, 细节生成, 后训练

## 一句话总结

构建了包含 10 万张超高分辨率图像及丰富标注的 UltraHR-100K 数据集，并提出频率感知后训练方法（DOTS + SWFR），通过面向细节的时间步采样和基于 DFT 的软加权频率正则化来增强预训练 T2I 模型的超高分辨率细节生成能力。

## 研究背景与动机

文本到图像（T2I）扩散模型在 1024×1024 分辨率下表现优异，但直接缩放到超高分辨率（UHR，如 4K）时会出现明显质量退化和结构伪影。现有方法分两大类：

**免训练方法**（DemofFusion、HiFlow 等）：通过修改推理策略实现 UHR 生成，但往往过度平滑、细节不真实、推理时间长

**训练型方法**（PixArt-σ、SANA 等）：主要关注训练效率，忽视了细节生成质量

两个核心挑战未解决：(1) 缺乏大规模高质量开源 UHR T2I 数据集——现有 Aesthetic-4K 仅约 1 万张且缺乏严格筛选；(2) 缺乏面向 UHR 细粒度细节合成的训练策略——现有模型的语义规划能力强，但在 UHR 场景下的高频细节生成不足。

## 方法详解

### 整体框架

本文工作包含两部分：
1. **UltraHR-100K 数据集**：10 万张精心筛选的 UHR 图像 + 丰富的文本标注
2. **频率感知后训练方法（FAPT）**：两阶段训练策略——第一阶段用 UltraHR-100K 做微调增强语义规划，第二阶段用 DOTS + SWFR 聚焦高频细节学习

### 关键设计

1. **UltraHR-100K 数据集构建**：

    - **数据收集**：使用基于 Scrapy 的爬虫收集约 40 万张高分辨率图像（最低 3840×2160）
    - **初步筛选**：拉普拉斯方差（评估锐度）+ Sobel 算子（评估边缘密度），去除模糊/无纹理图像
    - **三维度精细筛选**：
        - **细节丰富度**：基于 GLCM（灰度共生矩阵）计算对比度、熵和相关性，保留前 50%
        - **内容复杂度**：基于 Shannon 熵衡量像素强度多样性，保留前 50%
        - **美学质量**：使用 LAION Aesthetic Predictor 评分，保留前 50%
    - **最终数据集** = 三个子集的**交集**：UltraHR-100K = S_G ∩ S_E ∩ S_A，确保每张图同时满足高标准
    - **标注**：使用 Gemini 2.0 生成详细长标注，覆盖全局摘要和细粒度描述，标注长度显著超过 Aesthetic-4K
    - 最终规模：104,117 张图像，平均高度 3648、宽度 5119

2. **面向细节的时间步采样（DOTS）**：

    - 观察：去噪过程中早期步骤主要重建低频结构，后期步骤渐进合成高频细节
    - 使用 Beta(α, β) 分布采样去噪时间步，α=2, β=4 使采样偏向后期步骤（接近 t=0）
    - 效果：引导模型在后训练阶段重点学习高频细节相关的去噪步

3. **软加权频率正则化（SWFR）**：

    - 对预测和目标执行 2D DFT（离散傅里叶变换），在频域施加加权约束
    - 频率软加权函数 w(r) = 1 + λ·(exp(γr)-1)/(exp(γ)-1)，r∈[0,1] 为归一化频率距离
    - λ 和 γ 控制高频强调的强度和陡度
    - L_freq = E[|w(r)·x̂ - w(r)·ŷ|²]
    - 总损失 = L_diff + λ_freq · L_freq
    - 比 DWT 方法（Diffusion4K 使用）提供更精细、连续的频率分离

### 损失函数 / 训练策略

- 两阶段训练：第一阶段 4K 步 Logit-Normal 采样微调，第二阶段 8K 步 DOTS+SWFR
- 使用 CAMEWrapper 优化器，恒定学习率 1e-4，混合精度训练，batch size=24
- 基于 SANA 模型训练，4 张 H20 GPU
- 评估基准：自建 UltraHR-eval4K（2000 张 4096×4096 图像）

## 实验关键数据

### 主实验（UltraHR-eval4K，4096×4096）

| 方法 | FID↓ | FID_patch↓ | IS↑ | IS_patch↑ | CLIP↑ | FG-CLIP↑ |
|------|------|-----------|-----|----------|-------|---------|
| FLUX + BSRGAN | 37.65 | 43.14 | 11.77 | 5.39 | 31.45 | 28.02 |
| I-Max(FLUX) | 37.67 | 37.84 | 11.99 | 4.39 | 31.49 | 27.78 |
| HiFlow(FLUX) | 35.89 | 38.33 | 11.77 | 4.62 | 31.52 | 27.75 |
| PixArt-σ | 33.17 | 32.20 | 12.21 | 5.39 | 31.78 | 28.65 |
| SANA | 37.07 | 38.80 | 11.78 | 5.65 | 31.70 | 28.60 |
| Diffusion4K | 39.86 | 38.52 | 10.83 | 3.24 | 31.41 | 26.48 |
| **Ours(UltraHR-100K)** | 34.00 | 20.93 | 12.50 | 5.02 | 31.85 | 28.65 |
| **Ours(+FAPT)** | **31.75** | **15.80** | **13.00** | 5.10 | 31.82 | **28.68** |

### 消融实验

| 模型 | DOTS | SWFR | 数据 | FID↓ | FID_patch↓ | CLIP↑ |
|------|------|------|------|------|-----------|-------|
| LoRA | × | × | Full | 35.07 | 35.02 | 31.80 |
| A（全量微调基线）| × | × | Full | 33.99 | 20.93 | 31.85 |
| B | ✓ | × | Full | 32.57 | 19.95 | 31.79 |
| C | ✓ | ✓ | 15K | 32.75 | 18.42 | 31.81 |
| **D（完整方法）** | ✓ | ✓ | Full | **31.74** | **15.79** | **31.82** |

### 关键发现

- FID_patch 提升幅度最显著（38.80→15.80），证明方法在细粒度细节生成上的优势
- 用户研究中以 70% 总体偏好率、78% 细节质量偏好率大幅领先竞争方法
- 数据规模至关重要：15K 子集 vs 完整 100K 数据集，后者明显优于前者
- DOTS 的 Beta(α=2, β=4) 最优，过大的 α 弱化细节学习，过小的 α 损害语义一致性
- SWFR 对 FID_patch 的贡献最为显著（19.95→15.79），验证高频正则化的有效性
- 在公开的 Aesthetic-Eval@4096 上同样取得最佳结果，证明泛化性

## 亮点与洞察

- **三维度交集筛选策略简洁有效**：细节丰富度 × 内容复杂度 × 美学质量的交集设计确保了数据集质量的高底线
- **DOTS 利用去噪过程的频率特性**：精确利用了"早期生成结构、后期生成细节"的物理特性
- **DFT vs DWT 的选择有道理**：DFT 提供连续频谱，比 DWT 的粗粒度离散分解更适合精细控制
- **后训练范式值得参考**：不需要从头训练，只需少量后训练就能显著提升预训练模型的 UHR 细节能力
- 10 万级 UHR 数据集的贡献本身就很有价值——将推动整个 UHR 生成领域的发展

## 局限与展望

- 频率感知后训练略微降低了文本-图像对齐度（CLIP 分数轻微下降）
- 数据集中人像数据较少，UHR 人像生成仍有提升空间
- 仅在 SANA 上训练和验证，未覆盖 FLUX、SD3 等其他主流模型
- 计算资源限制：4 张 H20 GPU

## 相关工作与启发

- 与 Diffusion4K（使用 DWT 频率分解）相比，本文的 DFT + 软加权方法在所有指标上大幅领先
- 与 PixArt-σ（高效 token 压缩）和 SANA（高效 4K 管线）相比，本文方法关注细节而非效率
- 启发：UHR 生成不仅需要高效架构，更需要高质量数据和面向细节的训练策略

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VINS-120K: Ultra High-Resolution Image Editing with A Large-Scale Dataset](../../CVPR2026/image_generation/vins-120k_ultra_high-resolution_image_editing_with_a_large-scale_dataset.md)
- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)
- [\[NeurIPS 2025\] Hephaestus: Mixture Generative Modeling with Energy Guidance for Large-scale QoS Degradation](hephaestus_mixture_generative_modeling_with_energy_guidance_for_large-scale_qos_.md)
- [\[CVPR 2025\] OmniStyle: Filtering High Quality Style Transfer Data at Scale](../../CVPR2025/image_generation/omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)
- [\[ICCV 2025\] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](../../ICCV2025/image_generation/enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)

</div>

<!-- RELATED:END -->
