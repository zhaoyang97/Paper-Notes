---
title: >-
  [论文解读] Do We Need Perfect Data? Leveraging Noise for Domain Generalized Segmentation
description: >-
  [AAAI 2026][语义分割][域泛化语义分割] 提出 FLEX-Seg 框架，将扩散模型合成数据中图像与语义掩码之间固有的**边界不对齐**(misalignment)转化为学习鲁棒表示的机会，通过粒度自适应原型 (GAP)、不确定性边界强调 (UBE) 和难度感知采样 (HAS) 三个模块，在域泛化语义分割任务上取得 SOTA。
tags:
  - "AAAI 2026"
  - "语义分割"
  - "域泛化语义分割"
  - "边界不对齐"
  - "扩散模型合成数据"
  - "自适应原型"
  - "不确定性加权"
---

# Do We Need Perfect Data? Leveraging Noise for Domain Generalized Segmentation

**会议**: AAAI 2026  
**arXiv**: [2511.22948](https://arxiv.org/abs/2511.22948)  
**代码**: [有](https://github.com/VisualScienceLab-KHU/FLEX-Seg)  
**领域**: 分割  
**关键词**: 域泛化语义分割, 边界不对齐, 扩散模型合成数据, 自适应原型, 不确定性加权

## 一句话总结

提出 FLEX-Seg 框架，将扩散模型合成数据中图像与语义掩码之间固有的**边界不对齐**(misalignment)转化为学习鲁棒表示的机会，通过粒度自适应原型 (GAP)、不确定性边界强调 (UBE) 和难度感知采样 (HAS) 三个模块，在域泛化语义分割任务上取得 SOTA。

## 研究背景与动机

域泛化语义分割 (DGSS) 旨在仅使用源域数据训练模型，使其在未见过的目标域（如不同天气、光照条件）上仍能表现良好。近期基于扩散模型的数据生成方法（如 DGInStyle）通过生成多样化的合成图像来增强泛化能力，但这些方法面临一个根本性挑战：

**生成图像与语义掩码之间的边界不对齐**。与真实数据集（标注从真实图像导出）不同，合成数据管道是从语义掩码生成图像，这一反向过程固有地引入像素级的空间不对齐，尤其在物体边界处。

作者的关键观察：
- 边界区域的错误率在正常条件下就显著高于内部区域
- 在雾、雨、雪、夜间等恶劣条件下，这种差距更加明显
- 现有边界感知方法假设图像与掩码完美对齐，无法处理合成数据的不对齐问题

核心洞察：**与其强制追求完美对齐，不如利用这种不对齐来学习更鲁棒的表示**。

## 方法详解

### 整体框架

FLEX-Seg (FLexible Edge eXploitation for Segmentation) 包含三个协同组件：

1. **GAP (Granular Adaptive Prototypes)**: 多粒度边界原型学习
2. **UBE (Uncertainty Boundary Emphasis)**: 基于预测熵的动态边界加权
3. **HAS (Hardness-Aware Sampling)**: 渐进式难例采样

训练流程：从源域 $\mathcal{D}_S$ 用扩散模型生成合成数据 $\mathcal{D}_G$，合并为统一训练集 → GAP 学习跨域不变的边界表示 → UBE 自适应强调不确定区域 → HAS 渐进聚焦困难样本。

### 关键设计

#### GAP: 粒度自适应原型

**问题分析**：语义边界存在固有的尺度变化——远处小物体呈现薄边界，近处大物体呈现厚边界区域。边界像素同时具备几何变化（厚薄）和风格变化（不同环境条件下的外观）。

**类-形状 Token 坐标系统**：将每个边界像素 $p_i$ 表示为 $(c_i, g_i)$，其中 $c_i$ 为类别语义，$g_i$ 编码几何属性（边界厚度）。

**多粒度边界提取**：通过形态学操作生成三种粒度的边界掩码（薄、中、厚）：
$$B_g = \text{Dilate}(M_d, k_g) \ominus \text{Erode}(M_d, k_g)$$

**原型库构建**：构建 $C \times 3$ 维原型库 $\mathcal{P} = \{p_{c,g}\}$（C 个类 × 3 种粒度），通过动量更新维护：
$$p_{c,g} \leftarrow m \cdot p_{c,g} + (1-m) \cdot f_{c,g}$$

**对比学习**：使用带不平衡感知权重的 InfoNCE loss：
$$\mathcal{L}_{GAP} = -\frac{1}{N} \sum_{i=1}^{N} w_{c_i,g_i} \cdot \log \frac{e^{\langle f_i, p_{c_i,g_i} \rangle / \tau}}{\sum_{(c',g') \in \mathcal{P}} e^{\langle f_i, p_{c',g'} \rangle / \tau}}$$

权重 $w_{c,g}$ 根据原型更新频率自适应调整，为低频类-粒度组合分配更高权重。

#### UBE: 不确定性边界强调

基于预测熵的动态加权机制，无需手动调参即可跨域适应：

1. 计算每个像素的预测熵：$H_{x,y} = -\sum_{c=1}^{C} p_c(x,y) \log p_c(x,y)$
2. 仅对边界区域施加自适应权重（内部像素保持权重 1）：
$$w(x,y) = 1 + \alpha \cdot \text{sigmoid}\left(\frac{H_{x,y} - \mu_H}{\sigma_H + \epsilon}\right), \quad \text{if } (x,y) \in B$$
3. 应用加权交叉熵：$\mathcal{L}_{UBE} = \frac{1}{N} \sum_{(x,y)} w(x,y) \cdot \mathcal{L}_{CE}(x,y)$

熵高的像素（通常在不对齐的边界或模糊区域）获得更大权重，引导模型重点关注困难区域。

#### HAS: 难度感知采样

通过 sigmoid 衰减调度，渐进地从随机采样过渡到基于损失的采样：

- 维护每张图片的难度分数 $h_i(t)$，基于 EMA 更新
- 阈值函数：$\text{threshold}(t) = \frac{1}{1 + e^{k(t-m)}}$
- 每次迭代：若随机数 $r > \text{threshold}(t)$，执行基于损失的采样；否则随机采样
- 采样概率与难度分数成正比（通过 softmax + 温度参数控制）

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{total} = \mathcal{L}_{UBE} + \lambda_{gap} \cdot \mathcal{L}_{GAP}$

- GAP 确保跨域一致的边界表示（对比学习）
- UBE 根据预测置信度自适应调整学习焦点
- 超参数：$\tau = 0.07$, $\alpha = 3.0$, $\lambda_{gap} = 0.5$, $k = 0.05$, $\tau_{HAS} = 1.0$

## 实验关键数据

### 主实验

源域 GTA → 5 个真实驾驶数据集，使用 MiT-B5 backbone + HRDA：

| 方法 | ACDC | DZ | CS | BDD | MV | 平均 |
|------|------|------|------|------|------|------|
| HRDA + DGInStyle | 46.07 | 25.53 | 58.63 | 52.25 | 62.47 | 48.99 |
| **HRDA + FLEX-Seg** | **48.51** | **28.16** | **59.49** | **52.48** | 61.71 | **50.07** |
| DAFormer + DGInStyle | 44.04 | 25.58 | 55.31 | 50.82 | 56.62 | 46.47 |
| **DAFormer + FLEX-Seg** | **46.56** | **29.51** | **56.84** | **52.06** | **57.93** | **48.58** |

在恶劣条件域上的提升尤为显著：ACDC +2.44%，Dark Zurich +2.63%（HRDA）；Dark Zurich +3.93%（DAFormer）。

### 消融实验

各模块贡献（DAFormer + MiT-B5，Avg2 = ACDC + DZ 均值）：

| GAP | UBE | HAS | Avg2 | Avg3 | Avg5 |
|-----|-----|-----|------|------|------|
| ✗ | ✗ | ✗ | 34.81 | 54.25 | 46.47 |
| ✓ | ✗ | ✗ | 36.33 (+1.52) | 55.26 | 47.69 |
| ✗ | ✓ | ✗ | 35.05 (+0.24) | 55.33 | 47.21 |
| ✓ | ✓ | ✗ | 36.15 | 56.07 | 48.10 |
| ✓ | ✓ | ✓ | **38.04** | 55.61 | **48.58** |

合成数据量消融：10,000 张最优，过多反而轻微下降。

Sigmoid 衰减 vs 线性衰减 vs 无衰减：Sigmoid 38.04% > 无衰减 36.82% > 线性 36.28%。

### 关键发现

- **GAP 是核心贡献**：单独引入 GAP 即带来 +1.52% Avg2 提升，多粒度边界原型对于域不变表示至关重要
- **HAS 的权衡效应**：加入 HAS 略降标准域 Avg3 (-0.46%) 但大幅提升恶劣域 Avg2 (+1.89%)，体现其聚焦困难样本的策略
- **框架泛化性强**：在 ALDM 生成的合成数据上也有一致提升 (+1.44% Avg2)

## 亮点与洞察

1. **逆向思维**：将合成数据的固有缺陷（边界不对齐）转化为学习鲁棒表示的机会，而非试图消除
2. **类-粒度二维原型库**：将边界特征分解为语义维度和几何维度，实现细粒度的跨域对齐
3. **自适应学习聚焦**：UBE 基于预测熵自动识别困难区域，免除人工调整边界权重
4. **渐进式课程学习**：HAS 的 sigmoid 衰减调度确保早期探索充分、后期聚焦困难样本

## 局限与展望

- 依赖预训练扩散模型生成合成数据（如 DGInStyle/ALDM），无法脱离合成数据使用
- HAS 在标准域上有轻微性能下降，难例聚焦与全域均衡之间的权衡尚可进一步优化
- 原型库大小 $C \times 3 \times 256$ 随类别数线性增长，大规模类别场景需考虑效率
- 仅在 GTA → 真实驾驶场景验证，其他合成-真实域差距（如室内、卫星图像）尚未探索

## 相关工作与启发

- **DGInStyle** (2024)：利用潜在扩散模型合成多样化图像，是本文的主要合成数据来源
- **FAMix**：使用 CLIP 预训练的 ResNet-50 做域泛化，但在恶劣条件下不如本方法
- **HRDA** (2023)：多分辨率融合的域自适应框架，本文在其基础上叠加 FLEX-Seg 实现提升
- **边界感知方法**（BAPA, InverseForm 等）：假设完美对齐，无法处理合成数据不对齐

## 评分

- 新颖性: ⭐⭐⭐⭐ — "利用噪声而非消除噪声"的思路新颖，三组件设计有针对性
- 技术深度: ⭐⭐⭐⭐ — 多粒度原型对比学习 + 熵引导加权 + 课程学习采样，技术组合合理
- 实验充分度: ⭐⭐⭐⭐⭐ — 5 个目标域、2 种 backbone、详细超参消融、跨生成模型验证
- 写作质量: ⭐⭐⭐⭐ — 动机分析深入，误差率分析图示说服力强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Causal-Tune: Mining Causal Factors from Vision Foundation Models for Domain Generalized Semantic Segmentation](causal-tune_mining_causal_factors_from_vision_foundation_mod.md)
- [\[CVPR 2025\] MambaOut: Do We Really Need Mamba for Vision?](../../CVPR2025/segmentation/mambaout_do_we_really_need_mamba_for_vision.md)
- [\[AAAI 2026\] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)
- [\[CVPR 2026\] Leveraging Class Distributions in CLIP for Weakly Supervised Semantic Segmentation](../../CVPR2026/segmentation/leveraging_class_distributions_in_clip_for_weakly_supervised_semantic_segmentati.md)
- [\[CVPR 2026\] PromptMoE: A Segmentation Refinement Framework Leveraging Mixture of Experts for Improved Prompting](../../CVPR2026/segmentation/promptmoe_a_segmentation_refinement_framework_leveraging_mixture_of_experts_for_.md)

</div>

<!-- RELATED:END -->
