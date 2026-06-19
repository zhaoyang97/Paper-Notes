---
title: >-
  [论文解读] UniFit: Towards Universal Virtual Try-on with MLLM-Guided Semantic Alignment
description: >-
  [AAAI 2026][多模态VLM][虚拟试穿] 提出 UniFit，一个由多模态大语言模型（MLLM）驱动的通用虚拟试穿框架，通过 MLLM 引导的语义对齐模块（MGSA）桥接文本指令与参考图像之间的语义鸿沟，并通过两阶段渐进训练+自合成流水线克服复杂场景的数据稀缺问题，首次在单一框架内支持 6 种 VTON 任务。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "虚拟试穿"
  - "MLLM"
  - "语义对齐"
  - "Transformer"
  - "自合成训练"
---

# UniFit: Towards Universal Virtual Try-on with MLLM-Guided Semantic Alignment

**会议**: AAAI 2026  
**arXiv**: [2511.15831](https://arxiv.org/abs/2511.15831)  
**代码**: [github.com/zwplus/UniFit](https://github.com/zwplus/UniFit)  
**领域**: 多模态VLM  
**关键词**: 虚拟试穿, MLLM, 语义对齐, 扩散 Transformer, 自合成训练

## 一句话总结

提出 UniFit，一个由多模态大语言模型（MLLM）驱动的通用虚拟试穿框架，通过 MLLM 引导的语义对齐模块（MGSA）桥接文本指令与参考图像之间的语义鸿沟，并通过两阶段渐进训练+自合成流水线克服复杂场景的数据稀缺问题，首次在单一框架内支持 6 种 VTON 任务。

## 研究背景与动机

### 问题定义

基于图像的虚拟试穿（VTON）旨在合成一个人穿着指定服装的逼真图像。尽管进展显著，但构建一个能灵活处理多样复杂任务的通用 VTON 框架仍是重大挑战。

### 核心矛盾

现有文本指令引导的 VTON 方法面临两个关键限制：

**语义鸿沟**：文本编码器（如 CLIP 或 T5）提取的抽象文本表示难以精确对应图像中的具体视觉细节（纹理、logo 形状等），导致生成结果保真度低、可控性弱

**数据稀缺**：公开数据集（如 VITON-HD、DressCode）仅提供单件服装-试穿结果对，缺乏多服装试穿、模特到模特试穿等复杂场景的训练数据

### 现有方法的功能对比

| 方法 | 单件试穿 | 无模特试穿 | 服装重建 | 多视角 | 多服装 | 模特到模特 |
|------|---------|-----------|---------|--------|--------|----------|
| AnyFit | ✓ | - | - | - | ✓ | - |
| CatVTON | ✓ | - | - | - | - | ✓ |
| MV-VTON | ✓ | - | - | ✓ | - | - |
| Any2AnyTryon | ✓ | ✓ | ✓ | - | - | - |
| **UniFit** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

UniFit 是首个在单一框架中支持全部 6 种 VTON 任务的方法。

## 方法详解

### 整体框架

UniFit 由三个核心组件构成：

1. **MGSA 模块**（红色）：编码多模态输入为连贯的语义引导
2. **VAE 编码器**（蓝色）：从参考图像提取低层视觉特征
3. **DiT（Diffusion Transformer）**（灰色）：以语义引导和低层视觉特征为条件生成输出图像

生成流程包含两个并行流：
- MGSA 利用 Qwen2-VL 和可学习查询捕获文本指令与参考图像间的语义关系，生成高层语义表示 $T_q$
- VAE 编码器处理参考图像提取细粒度视觉特征 $r = \{r_1, \ldots, r_n\}$
- 将 $T_q$、噪声潜变量 $z_t$、参考 token $r$ 拼接为 DiT 输入 $[T_q; z_t; r_1; \ldots; r_n]$

### 关键设计

#### 1. **MLLM-Guided Semantic Alignment Module (MGSA)**：桥接文本与视觉的语义鸿沟

**核心思路**：直接利用 MLLM（Qwen2-VL-2B-Instruct）联合处理文本指令和视觉输入，而非像现有方法那样分别用文本编码器和图像编码器独立处理。

**可学习查询**：引入 $T_q \in \mathbb{R}^{N_q \times D_q}$（$N_q = 486$，$D_q = 1536$），附加在 Qwen2-VL 输入序列末尾。通过因果注意力机制，查询从冗长的多模态序列中蒸馏任务相关信号为紧凑表示。

**语义对齐损失**：将 $T_q$ 与目标图像的真实视觉表示 $T_v$（通过冻结 ViT 提取）对齐：

$$\mathcal{L}_{\text{align}} = -\frac{1}{N_v} \sum_{n=1}^{N_v} \cos(T_{v,n}, \text{MLP}(T_{q,n}))$$

通过 token 级余弦相似度对齐，确保查询表示语义上对应目标输出。

**设计动机**：
- 可学习查询解决了 MLLM 输出序列过长导致的冗余信息和计算开销问题
- 语义对齐损失使 MGSA 学会融合多模态输入，生成对 DiT 有意义的显式引导
- 相比 CLIP/T5 的抽象文本特征，MLLM 能联合理解文本和图像的语义关系

#### 2. **Spatial Attention Focusing Loss**：引导 DiT 聚焦任务相关区域

**核心思路**：DiT 的交叉注意力往往分散在不相关区域，导致细节退化和视觉伪影。通过显式正则化交叉注意力图，强制模型聚焦关键区域。

计算交叉注意力图 $AttnMap \in \mathbb{R}^{l_{r_i} \times l_{z_t}}$，根据任务类型：
- **试穿任务**：沿参考 token 轴平均，得到输出中心响应图 $M \in \mathbb{R}^{l_{z_t}}$
- **服装重建任务**：沿输出 token 轴平均，得到参考中心响应图 $M \in \mathbb{R}^{l_{r_i}}$
- **模特到模特**：同时计算和监督两种响应图

使用 MSE 损失与真实空间掩码 $M_{\text{target}}$ 对齐：

$$\mathcal{L}_{\text{focus}} = \frac{1}{N_R \times N_L} \sum_{j=1}^{N_L} \sum_{i=1}^{N_R} \|M_i^j - M_{\text{target},i}\|_2^2$$

**设计动机**：虽然 MGSA 提供了强大的高层语义引导，但 DiT 的注意力仍可能分散，需要空间层面的显式约束来确保细粒度细节的忠实传递。

#### 3. **两阶段渐进训练 + 自合成流水线**：克服复杂场景的数据稀缺

**Stage I - Base Pretraining**：
- 在 VITON-HD 和 DressCode 上训练 Base Model，学习三个基础任务（单件试穿、服装重建、无模特试穿）
- 每个任务约 59K 训练对，120K 步

**Self-Synthesis（自合成）**：
- **多服装试穿**：利用 Base Model 的服装重建能力，从全身图像中提取分离的上装/下装，创建 10K 配对
- **模特到模特试穿**：利用无模特试穿能力，合成以现有服装为条件的新人物图像，创建 30K 配对
- 双重筛选：DreamSim 感知相似度 + Qwen2.5-VL-7B-Instruct 一致性检查

**Stage II - Joint Finetuning**：
- 在真实数据 + 合成数据混合数据集上微调 80K 步
- 同时覆盖全部 6 种 VTON 任务

### 损失函数 / 训练策略

总训练目标为三项损失的组合：
1. **Flow matching loss**：标准生成损失
2. **语义对齐损失 $\mathcal{L}_{\text{align}}$**：MGSA 查询与目标视觉 token 的余弦对齐
3. **空间注意力聚焦损失 $\mathcal{L}_{\text{focus}}$**：DiT 交叉注意力图与空间掩码的 MSE

训练细节：
- MGSA 基于 Qwen2-VL-2B-Instruct，冻结前 14 层，微调后 14 层
- DiT backbone：StableDiffusion-3.5 Medium
- 训练分辨率：$1024 \times 768$（大部分任务）
- AdamW 优化器，学习率 $4 \times 10^{-5}$，梯度裁剪 1.0，batch size 16

## 实验关键数据

### 主实验

**服装重建**（VITON-HD）：

| 方法 | SSIM ↑ | LPIPS ↓ | DISTS ↓ | FID ↓ |
|------|--------|---------|---------|-------|
| TryOffDiff | 0.792 | 0.337 | 0.227 | 21.40 |
| Any2AnyTryon | 0.762 | 0.367 | 0.231 | 13.57 |
| **UniFit** | **0.775** | **0.281** | **0.202** | **12.58** |

**单件试穿**（VITON-HD）：

| 方法 | SSIM ↑ | LPIPS ↓ | FID ↓ | KID ↓ |
|------|--------|---------|-------|-------|
| CatVTON | 0.888 | 0.075 | 9.128 | 1.130 |
| FitDiT | 0.895 | 0.067 | 9.326 | 0.913 |
| Any2AnyTryon | 0.839 | 0.088 | 8.965 | 0.981 |
| **UniFit** | **0.883** | **0.065** | **8.799** | **0.702** |

FID 降低至 8.799（最佳），KID 大幅降低至 0.702，提升明显。

**无模特试穿**（VITON-HD）：

| 方法 | CLIP-AS ↑ | CLIP-I ↑ | MP-LPIPS ↓ |
|------|----------|---------|-----------|
| IMAGDressing-v1 | 4.96 | 0.880 | 0.107 |
| Any2AnyTryon | 4.95 | 0.843 | 0.127 |
| **UniFit** | 4.91 | **0.914** | **0.078** |

**多视角试穿**（MVG）：

| 方法 | SSIM ↑ | LPIPS ↓ | FID ↓ | KID ↓ |
|------|--------|---------|-------|-------|
| MV-TON | 0.930 | 0.062 | 37.09 | 3.23 |
| **UniFit** | **0.935** | 0.072 | **35.62** | 3.85 |

### 消融实验

（VITON-HD 单件试穿，Stage I Base Model）：

| 配置 | SSIM ↑ | LPIPS ↓ | FID ↓ | KID ↓ | 说明 |
|------|--------|---------|-------|-------|------|
| w/o MGSA（用 T5 替代） | 0.851 | 0.098 | 9.133 | 1.053 | 全面大幅下降 |
| w/o $\mathcal{L}_{\text{align}}$ | 0.863 | 0.074 | 8.937 | 0.951 | 对齐损失重要 |
| w/o $\mathcal{L}_{\text{focus}}$ | 0.872 | 0.069 | 8.870 | 0.835 | 聚焦损失有贡献 |
| **完整模型 (Stage I)** | **0.887** | **0.071** | **8.813** | **0.785** | 最佳 |

### 关键发现

1. **MGSA 模块是核心贡献**：去掉 MGSA 改用 T5 后，SSIM 从 0.887 降至 0.851，FID 从 8.813 升至 9.133，下降巨大
2. **语义对齐损失关键**：没有 $\mathcal{L}_{\text{align}}$，SSIM 从 0.887 降至 0.863，说明显式语义对齐对引导质量至关重要
3. **自合成流水线有效**：通过 Base Model 自合成数据，使模型能处理训练数据中原本不存在的复杂任务（多服装、模特到模特）
4. **多任务框架效果优异**：在 6 种不同任务上都达到 SOTA 或可比性能，证明通用框架的可行性

## 亮点与洞察

1. **MLLM 应用于 VTON 的创新**：首次将 MLLM 深度集成到 VTON 框架中，不仅用于理解指令，还用于桥接多模态语义鸿沟
2. **自合成训练策略精妙**：利用已训练好的模型生成新任务的训练数据，形成正向循环，巧妙解决了数据稀缺瓶颈
3. **空间注意力聚焦损失**：通过显式监督 DiT 的注意力图，解决了注意力分散导致的细节退化问题
4. **Mask-Free 设计**：不依赖服装掩码，通过图像修补合成三元组训练样本，更实用

## 局限与展望

1. **wild 场景表现不确定**：受限于现有数据集分布（以门店场景为主），极端光照和严重遮挡下可能性能退化
2. **不支持分层试穿**：当前无法处理穿着层次（如外套覆盖内衣）
3. **不支持文本可编辑试穿**：无法通过文本描述修改服装属性（如"改为红色"）
4. **自合成数据质量**：虽有 DreamSim + Qwen2.5-VL 筛选，但合成数据的分布偏差可能限制泛化
5. **计算开销较大**：MLLM（Qwen2-VL-2B）+ DiT（SD3.5-Medium）的组合推理成本可能较高

## 相关工作与启发

- **CatVTON / FitDiT**：基于 DiT 的单任务 VTON 方法，UniFit 将其扩展到多任务
- **Any2AnyTryon**：此前最接近通用 VTON 的工作，但不支持多服装和模特到模特，且文本指令遵循能力较弱
- **DreamO**：空间注意力聚焦损失的灵感来源
- **Qwen2-VL**：UniFit 选择的 MLLM backbone，提供多模态理解能力
- **启发**：MLLM 不仅可以用于理解和生成，还可以作为"语义桥梁"连接不同模态，指导生成模型

## 评分

- 新颖性: ⭐⭐⭐⭐ （MLLM + VTON 的融合方式新颖，自合成训练也很有创意）
- 实验充分度: ⭐⭐⭐⭐⭐ （6 种任务全面评估，多个数据集，完整消融）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图示丰富，问题定义清楚）
- 价值: ⭐⭐⭐⭐⭐ （首个支持 6 种 VTON 任务的通用框架，实用性极高）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Exo2Ego: Exocentric Knowledge Guided MLLM for Egocentric Video Understanding](exo2ego_exocentric_knowledge_guided_mllm_for_egocentric_vide.md)
- [\[CVPR 2026\] Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models](../../CVPR2026/multimodal_vlm/uncertainty-guided_compositional_alignment_with_part-to-whole_semantic_represent.md)
- [\[CVPR 2026\] Gravitation-Driven Semantic Alignment for Text Video Retrieval](../../CVPR2026/multimodal_vlm/gravitation-driven_semantic_alignment_for_text_video_retrieval.md)
- [\[CVPR 2026\] Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning](../../CVPR2026/multimodal_vlm/anchor-guided_gradient_alignment_for_incomplete_multimodal_learning.md)
- [\[CVPR 2026\] Self-guided Semantic Inspection for Zero-Shot Composed Image Retrieval](../../CVPR2026/multimodal_vlm/self-guided_semantic_inspection_for_zero-shot_composed_image_retrieval.md)

</div>

<!-- RELATED:END -->
