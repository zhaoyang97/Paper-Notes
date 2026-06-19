---
title: >-
  [论文解读] DiffEye: Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned on Natural Images
description: >-
  [NeurIPS 2025][图像生成][扩散模型] 提出 DiffEye，首个基于扩散模型直接利用原始眼动轨迹数据、以自然图像为条件生成连续且多样化眼动轨迹的框架，同时引入对应位置嵌入 (CPE) 对齐注视空间与图像语义空间。 - 眼动追踪的重要性：眼动追踪为 VR、发育科学（如自闭症研究）、广告效果分析等领域提供关键的视…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "扩散模型"
  - "眼动轨迹生成"
  - "注视路径预测"
  - "视觉注意力建模"
  - "对应位置嵌入"
---

# DiffEye: Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned on Natural Images

**会议**: NeurIPS 2025  
**arXiv**: [2509.16767](https://arxiv.org/abs/2509.16767)  
**代码**: 有 ([https://diff-eye.github.io/](https://diff-eye.github.io/))  
**领域**: 图像生成 / 扩散模型 / 眼动追踪  
**关键词**: 扩散模型, 眼动轨迹生成, 注视路径预测, 视觉注意力建模, 对应位置嵌入

## 一句话总结

提出 DiffEye，首个基于扩散模型直接利用原始眼动轨迹数据、以自然图像为条件生成连续且多样化眼动轨迹的框架，同时引入对应位置嵌入 (CPE) 对齐注视空间与图像语义空间。

## 研究背景与动机

- **眼动追踪的重要性**：眼动追踪为 VR、发育科学（如自闭症研究）、广告效果分析等领域提供关键的视觉注意力数据，但数据采集成本高、耗时长
- **现有方法的两大缺陷**：
  1. 多数方法在 scanpath（离散注视点序列）上操作，丢失了原始轨迹的丰富时空信息（MIT1003 中平均 scanpath 仅 8.4 步 vs 原始轨迹 723.7 步）
  2. 现有方法要么用自回归采样模拟变异性，要么做确定性预测，无法真正捕获人类注视行为的固有分布
- **核心假设**：利用完整眼动轨迹训练生成模型，可以比 scanpath 方法更有效地表征视觉注意力动态；扩散模型特别适合建模这种内在随机性

## 方法详解

### 整体框架

DiffEye 采用 DDPM 框架学习条件分布 $p_\theta(\mathcal{R}|\mathcal{S})$，即给定视觉刺激 $I$ 下的眼动轨迹分布：

- **输入**：视觉刺激图像 $I \in \mathbb{R}^{H \times W \times 3}$（resize 到 224×224）和固定长度轨迹 $R \in \mathbb{R}^{L \times 2}$（L=720）
- **前向过程**：对轨迹逐步加高斯噪声，共 $T_{diff}=1000$ 步
- **反向过程**：U-Net 预测噪声，以图像为条件逐步去噪恢复真实轨迹
- **推理时**：从纯噪声出发，用 DDIM 采样 50 步生成轨迹，可转换为 scanpath 或 saliency map

### 关键设计

#### 1. 基础模型：1D U-Net

- 下采样、中间、上采样块结构，使用 **1D 卷积**对轨迹进行时间维度的降采样/升采样
- 每个块内含自注意力层捕捉轨迹的时间依赖关系
- 扩散时间步 $t_{diff}$ 通过正弦位置编码 + MLP + SiLU 注入各块

#### 2. 图像条件注入机制（逐步演进）

| 方案 | 方式 | 效果 |
|------|------|------|
| 全局特征 | DINOv2 全局向量拼接到轨迹 token | 生成质量差，缺乏空间语义 |
| Patch 特征 + 单层交叉注意力 | DINOv2 patch token 与轨迹 token 做交叉注意力 | 条件注入不充分 |
| Patch 特征 + 多层交叉注意力 | 每个 U-Net 块末尾加交叉注意力 | 显著提升 |
| **FeatUp 高分辨率特征** | 替换 DINOv2 patch 为 FeatUp（32×32 vs 16×16） | 最终方案，空间精度更高 |

#### 3. 对应位置嵌入 (CPE) — 核心创新

CPE 的关键思想：让轨迹 token 和图像 patch token **共享同一套 2D 正弦位置编码网格**，实现空间对齐：

- 构建位置嵌入网格 $P \in \mathbb{R}^{H \times W \times D}$
- 对于轨迹第 $i$ 步坐标 $(x_i, y_i)$，从网格提取对应位置编码：$R_i^{CPE} = R_{proj}[i] + P[y_i, x_i, :]$
- 对图像特征，插值位置网格到 patch 分辨率后加到特征上：$F_{CPE} = F_{proj} + P'$
- 在交叉注意力中，同一空间位置的轨迹点和图像 patch 获得匹配的位置信号

#### 4. 数据预处理

- 使用 MIT1003 数据集（15 名被试 × 1003 张图 × 3 秒自由观看，240Hz 采样）
- 去除眨眼和 NaN，保留 ≥720 步的序列，最终 8,934 条轨迹
- 统一截取/下采样到 720 步，90%/10% 按图像划分训练/测试集

### 损失函数 / 训练策略

- **损失函数**：标准 DDPM 噪声预测损失 $\min_\theta \mathbb{E}_{t,R^{(0)},\epsilon}\left[\|\epsilon - \epsilon_\theta(R^{(t)}, t, I)\|^2\right]$
- Adam 优化器，固定学习率 $1 \times 10^{-4}$，训练 3000 epochs
- 线性噪声调度 $[1 \times 10^{-4}, 2 \times 10^{-2}]$
- 使用 **Classifier-Free Guidance (CFG)** 增强条件控制
- 推理用 DDIM 50 步采样

## 实验关键数据

### 主实验：Scanpath 生成（MIT1003 + OSIE）

| 测试集 | 方法 | Levenshtein ↓ (Mean/Best) | DFD ↓ ×10² | DTW ↓ ×10³ | TDE ↓ (Mean/Best) |
|--------|------|--------------------------|------------|------------|-------------------|
| MIT1003 | IOR-ROI | 13.574/11.092 | 3.777/2.460 | 1.834/1.317 | 108.284/80.944 |
| MIT1003 | DeepGaze III (seen) | 14.415/11.856 | 3.553/2.160 | 1.757/1.141 | 96.456/65.408 |
| MIT1003 | **DiffEye** | **13.009/9.709** | **3.529/2.449** | **1.573/1.067** | **88.661/53.486** |
| OSIE | DeepGaze III | 15.507/12.532 | 3.206/2.077 | 1.765/1.166 | 84.337/57.786 |
| OSIE | **DiffEye** | **14.771/12.077** | **3.068/2.238** | **1.552/1.089** | **81.925/54.347** |

DiffEye 训练数据量（8,934 轨迹）远小于基线（DeepGaze III 用 615K scanpath），但全面领先。

### 消融实验

| 配置 (Scanpath) | Levenshtein ↓ ×10² | DTW ↓ ×10⁴ | TDE ↓ (Mean) |
|------|---------------------|------------|-------------|
| DiffEye (完整) | **0.130** | **0.157** | **88.661** |
| w/o FeatUp | 0.133 | 0.163 | 91.007 |
| w/o CPE | 0.141 | 0.180 | 100.792 |
| w/o U-Net 交叉注意力 | 0.143 | 0.189 | 107.962 |
| w/o Patch 级特征 (仅全局) | 0.153 | 0.209 | 116.226 |

### 关键发现

1. **每个组件都有贡献**：Patch 级特征 > 多层交叉注意力 > CPE > FeatUp 高分辨率，逐步叠加效果显著
2. **小数据即可超越大规模训练的基线**：DiffEye 仅用 8,934 条轨迹，超越了用 60K+ scanpath 训练的 HAT、GazeFormer
3. **泛化能力强**：在完全未见过的 OSIE 数据集上仍然表现最优
4. **首次实现连续眼动轨迹生成**：此前方法仅生成离散 scanpath

## 亮点与洞察

- **原始数据的价值**：直接利用原始眼动轨迹（720 步）而非压缩的 scanpath（~8 步），信息量提升约 85 倍，显著增强了生成质量
- **CPE 的巧妙设计**：通过共享位置编码网格，零参数开销实现轨迹-图像空间对齐，简洁而有效
- **扩散模型的天然优势**：无需自回归采样即可生成多样化轨迹，真正建模人类注视行为的随机分布特性
- **下游可转换性**：生成的连续轨迹可按需转换为 scanpath 或 saliency map，一个模型服务多种任务

## 局限与展望

1. **仅支持固定长度输出**：当前固定生成 720 步 (240Hz)，无法适配不同采样率和序列长度
2. **数据集单一**：MIT1003 是唯一提供原始眼动轨迹的自然图像数据集，规模有限
3. **仅限自由观看任务**：未验证在视觉搜索、视觉问答等其他视觉任务上的表现
4. **saliency 预测非最优**：间接从轨迹转换的 saliency map 精度不及专用 saliency 模型
5. 未来可探索：迁移学习、可变长度生成、个性化建模（如自闭症人群 vs 正常人群）

## 相关工作与启发

- **DiffGaze** (Jiao et al., 2024)：360° 图像上的扩散眼动生成，但仅用全局特征条件化；DiffEye 证明 patch 级特征 + CPE 远优于全局特征
- **HAT** (Yang et al., 2024)：统一 FV/TP/TA 三种任务的 Transformer 模型，但采样策略类似 DeepGaze III
- **FeatUp** (Fu et al., 2024)：模型无关的特征上采样框架，为 DiffEye 提供高分辨率语义特征
- **启发**：扩散模型 + 空间位置对齐的思路可推广到其他序列-图像条件生成任务

## 评分

| 维度 | 分数 | 评价 |
|------|------|------|
| 新颖性 | ★★★★☆ | 首个自然图像上的扩散眼动轨迹生成，CPE 设计巧妙 |
| 技术深度 | ★★★★☆ | 架构设计合理，消融全面，但理论分析较少 |
| 实验充分性 | ★★★★☆ | 多度量评估、多基线对比、消融研究完善 |
| 实用价值 | ★★★★☆ | 可应用于 VR、发育科学等领域，代码开源 |
| 写作质量 | ★★★★☆ | 图表清晰，动机明确，逻辑流畅 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Detecting Generated Images by Fitting Natural Image Distributions](detecting_generated_images_by_fitting_natural_image_distributions.md)
- [\[NeurIPS 2025\] SAO-Instruct: Free-form Audio Editing using Natural Language Instructions](sao-instruct_free-form_audio_editing_using_natural_language_instructions.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](continuous_diffusion_model_for_language_modeling.md)
- [\[NeurIPS 2025\] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)
- [\[NeurIPS 2025\] Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models](mitigating_sexual_content_generation_via_embedding_distortion_in_text-conditione.md)

</div>

<!-- RELATED:END -->
