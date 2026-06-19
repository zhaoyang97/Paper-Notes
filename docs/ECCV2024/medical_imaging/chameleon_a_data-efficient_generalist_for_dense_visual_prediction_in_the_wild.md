---
title: >-
  [论文解读] Chameleon: A Data-Efficient Generalist for Dense Visual Prediction in the Wild
description: >-
  [ECCV2024][医学图像][vision generalist] 提出 Chameleon，一个基于 meta-learning 和 token matching 的数据高效视觉通才模型，仅需几十张标注图像即可适应全新的密集预测任务（包括医学图像、视频、3D 等），在六个下游基准上显著超越现有通才方法。
tags:
  - "ECCV2024"
  - "医学图像"
  - "vision generalist"
  - "low-shot learning"
  - "dense prediction"
  - "meta-learning"
  - "token matching"
---

# Chameleon: A Data-Efficient Generalist for Dense Visual Prediction in the Wild

**会议**: ECCV2024  
**arXiv**: [2404.18459](https://arxiv.org/abs/2404.18459)  
**代码**: [GitGyun/chameleon](https://github.com/GitGyun/chameleon)  
**领域**: 医学图像  
**关键词**: vision generalist, low-shot learning, dense prediction, meta-learning, token matching

## 一句话总结

提出 Chameleon，一个基于 meta-learning 和 token matching 的数据高效视觉通才模型，仅需几十张标注图像即可适应全新的密集预测任务（包括医学图像、视频、3D 等），在六个下游基准上显著超越现有通才方法。

## 背景与动机

- 大语言模型凭借通用的语言接口和大规模预训练成为了数据高效的通才，但在密集视觉预测领域构建类似的通才模型面临独特挑战：不同任务的标签结构差异极大（关键点热图、6D 位姿、语义分割等）
- 现有视觉通才方法的局限：
    - **多任务学习方法**（Painter、InvPT 等）：将预定义任务统一到单一模型，但需要大量标注数据且无法泛化到未见任务
    - **In-context learning 方法**（SegGPT 等）：尝试通过少量示例解决新任务，但面对训练时未见过的标签结构和语义（如 6D 位姿、动物关键点）时泛化能力有限
- 核心动机：需要一个能灵活适应任意未见密集标签结构的通用模型，在低数据场景下仍能有效工作

## 核心问题

如何构建一个数据高效的密集视觉预测通才模型，使其能够用极少量标注样本（≤50 张）适应训练时完全未见过的任务类型、标签结构和数据域？

## 方法详解

### 整体框架：Visual Token Matching

Chameleon 基于 Visual Token Matching（VTM）框架，将密集预测建模为 query 和 support 图像之间的 token 级匹配问题：

- 给定 query 图像 $X^q$ 和少量标注的 support 集 $\mathcal{S}_\mathcal{T} = \{(X^i, Y^i)\}_{i \leq N}$
- 通过图像编码器提取 token embedding，计算 query token 与 support token 的相似度
- 根据相似度对 support 标签 embedding 进行插值，得到 query 的预测

### 多模态输入编码器（Section 3.1）

- 采用固定 patch 大小将多模态输入 patchify 为 $I_\mathcal{T} \times M_\mathcal{T}$ 个 token
- 所有 token 通过 Transformer 编码器一次性编码，实现跨模态上下文化
- 设计**可学习的相对位置偏置** $P_\mathcal{T}^{(b)}[m, m', h-h', w-w']$：前两个索引区分模态对，后两个索引编码空间相对位置
- 位置偏置作为任务特定参数的一部分，不同任务学习不同的跨模态交互模式

### 特征调制机制（Section 3.2）

两种自适应方式：

1. **Bias tuning**：每层编码器的偏置参数 $\mathbf{b}_\mathcal{T}$ 按任务单独调整
2. **特征重加权（Feature Re-weighting）**：引入可学习矩阵 $\Lambda_\mathcal{T} \in \mathbb{R}^{L \times L}$，对 $L$ 层图像特征进行任务自适应重加权后送入对应层的 matching 模块

$$F_\mathcal{T} = \Lambda_\mathcal{T} \hat{F}_\mathcal{T}$$

- 每行归一化使贡献总量恒定
- 匹配在 $L$ 个层级进行，输出转换为特征金字塔，由卷积解码器逐级解码
- 任务特定参数 $\theta_\mathcal{T} = (P_\mathcal{T}, \mathbf{b}_\mathcal{T}, \Lambda_\mathcal{T})$ 仅占全部参数的极小比例，抗过拟合

### 训练策略

- **Episodic meta-training**：在包含 17 种密集预测任务的大规模数据集（约 120 万图像，来自 Taskonomy、COCO、MidAir、MPII、DeepFashion、FreiHand 六个数据集）上进行元训练
- **Few-shot fine-tuning**：在目标任务上仅微调任务特定参数和部分标签解码器

### 模型扩展

- 图像编码器扩展至 BEiTv2-Large，标签编码器扩展至 ViT-Large
- 标签解码器卷积通道从 96 增加到 256
- 元训练分辨率 224×224，下游任务自适应分辨率

## 实验关键数据

在六个完全未见的下游任务上评估（使用 ≤50 张标注图像）：

| 任务 | 数据集 | 指标 | Chameleon | 最佳通才基线 | 全监督专家 |
|------|--------|------|-----------|-------------|-----------|
| 动物关键点检测 | AP-10K | AP↑ | **67.2** | 9.1 (VTM) | 69.8 (HRNet) |
| 6D 位姿估计 | LineMOD | ADD↑ | **85.2** | 59.3 (VTM) | 89.9 (CDPN) |
| 皮肤病变分割 | ISIC 2018 | F1↑ | **88.5** | 88.1 (SegGPT+PT) | 89.8 (UNeXt) |
| 视频目标分割 | DAVIS 2017 | J&F↑ | **77.5** | 75.6 (SegGPT+ICL) | 88.2 (ISVOS) |
| 目标计数 | FSC-147 | MAE↓ | **12.0** | - | 10.8 (LOCA) |
| 细胞实例分割 | Cellpose | AP50↑ | **70.3** | - | 70.4 (Cellpose) |

关键发现：

- Chameleon 在所有任务上大幅超越通才基线，尤其在结构未见任务（动物关键点 67.2 vs 9.1，6D 位姿 85.2 vs 59.3）
- 在多个任务上接近甚至匹敌全监督专家模型（如细胞分割 70.3 vs 70.4）
- 消融实验证明每个组件的贡献，feature re-weighting 对 OOD 任务提升最显著

## 亮点

- **极强的 OOD 泛化**：仅用几十张标注图像就能适应训练时完全未见的任务类型和数据域，包括医学图像、3D 理解、视频追踪等
- **灵活的多模态处理**：通过可学习位置偏置优雅地处理不同数量和类型的输入模态
- **层级特征重加权**：简洁地解决了不同任务需要不同层级特征对应关系的问题
- **实验设计全面**：六个差异极大的下游任务覆盖了域外、结构外、多模态等多种挑战场景
- 消融实验发现元训练数据多样性的间接提升效果（如加入无人机合成数据提升动物姿态估计）

## 局限与展望

- 视频目标分割未利用时序信息，面对外观相似的干扰物容易出错
- 元训练和微调的计算开销较大（BEiTv2-Large + ViT-Large）
- 仅在像素级密集预测任务上验证，未扩展到检测、实例分割等需要后处理的任务
- Support set 的选择策略对性能影响可能较大，论文未深入探讨
- 目标计数和细胞分割需要特定的标签重新表示（热图/flow），这种转换本身需要领域知识

## 与相关工作的对比

| 方法 | 框架 | 训练时未见任务泛化 | 多模态输入 | 低数据适应 |
|------|------|----------|----------|----------|
| Painter | In-context learning | 有限（结构内） | ✗ | ICL/PT |
| SegGPT | In-context learning | 有限（仅分割） | ✗ | ICL/PT |
| VTM | Token matching | 中等（窄域） | ✗ | Fine-tuning |
| **Chameleon** | Token matching + meta-learning | **强（跨域+跨结构）** | **✓** | **Fine-tuning** |

与 VTM 的核心区别：VTM 仅在室内场景等窄域验证，Chameleon 通过多模态编码器、特征重加权和大规模多样化元训练将其扩展到真实世界的广泛场景。

## 启发与关联

- Token matching 框架的通用性值得关注：将密集预测统一为 token 级检索+插值，无需对输出结构做假设
- 元训练数据多样性比直接相关性更重要的发现，对构建基础模型有启发
- 层级特征重加权的思想可推广到其他需要多尺度特征对齐的场景
- 对医学图像领域尤其有价值：标注昂贵、任务多样、域差异大，正好匹配 Chameleon 的优势

## 评分

- 新颖性: ⭐⭐⭐⭐ — 在 VTM 基础上的改进逻辑清晰，多模态编码和特征重加权设计简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ — 六个差异极大的下游任务，全面的消融实验和可视化分析
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题动机阐述充分
- 价值: ⭐⭐⭐⭐ — 在低数据密集预测通才方向上推进了 SOTA，对医学图像等实际应用场景价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Cross-Modal Guided Visual Synthesis for Data-Efficient Multimodal Depression Recognition](../../CVPR2026/medical_imaging/cross-modal_guided_visual_synthesis_for_data-efficient_multimodal_depression_rec.md)
- [\[ICML 2026\] Which Anatomy Matters Under Limited Labels? A Data-Efficient Anatomy-Aware Benchmark for Cardiac Pathology Prediction](../../ICML2026/medical_imaging/which_anatomy_matters_under_limited_labels_a_data-efficient_anatomy-aware_benchm.md)
- [\[CVPR 2026\] CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](../../CVPR2026/medical_imaging/chips_efficient_clip_adaptation_via_curvature-aware_hybrid_influence-based_data_.md)
- [\[ECCV 2024\] Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis](radiative_gaussian_splatting_for_efficient_x-ray_novel_view_synthesis.md)
- [\[ICML 2026\] MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](../../ICML2026/medical_imaging/meg-xl_data-efficient_brain-to-text_via_long-context_pre-training.md)

</div>

<!-- RELATED:END -->
