---
title: >-
  [论文解读] Uni3DL: Unified Model for 3D and Language Understanding
description: >-
  [ECCV 2024][多模态VLM][3D视觉-语言统一模型] 提出 Uni3DL，一个直接在点云上操作的统一 3D 视觉-语言模型，通过 Query Transformer 学习任务无关的语义/掩码输出，再由 Task Router 组合多个功能头实现语义分割、实例分割、目标检测、视觉定位、3D 描述生成、文本-3D 检索等六大任务，性能达到或超过各任务专用 SOTA。
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "3D视觉-语言统一模型"
  - "点云理解"
  - "Transformer"
  - "多任务学习"
  - "功能统一"
---

# Uni3DL: Unified Model for 3D and Language Understanding

**会议**: ECCV 2024  
**arXiv**: [2312.03026](https://arxiv.org/abs/2312.03026)  
**代码**: [https://uni3dl.github.io/](https://uni3dl.github.io/)  
**领域**: 多模态VLM / 3D视觉  
**关键词**: 3D视觉-语言统一模型, 点云理解, Query Transformer, 多任务学习, 功能统一

## 一句话总结

提出 Uni3DL，一个直接在点云上操作的统一 3D 视觉-语言模型，通过 Query Transformer 学习任务无关的语义/掩码输出，再由 Task Router 组合多个功能头实现语义分割、实例分割、目标检测、视觉定位、3D 描述生成、文本-3D 检索等六大任务，性能达到或超过各任务专用 SOTA。

## 研究背景与动机

**领域现状**：3D 感知技术是机器人导航、自动驾驶、虚拟现实等应用的基础。当前 3D 感知领域存在大量任务专用模型（semantic segmentation、instance segmentation、visual grounding、captioning 等），每个任务独立设计架构和训练。

**现有痛点**：
   - **任务覆盖不全**：现有 3D 视觉-语言统一模型（如 PointCLIP v2、ULIP、3D-VisTA）支持的任务种类有限，特别是密集预测任务（语义/实例分割）关注较少
   - **依赖多视图图像**：大多数方法需要将点云投影为多视图 2D 图像再处理，导致 3D 几何信息丢失，且增加模型复杂度
   - **需要任务特定微调**：如 3D-VisTA 虽然做了视觉-语言预训练，但下游每个任务仍需独立的 task head

**核心矛盾**：2D 领域的统一模型（如 CLIP、X-Decoder、Mask2Former）已取得巨大成功，但 2D→3D 的迁移面临两大障碍——2D/3D 架构差异大 + 3D 大规模预训练数据稀缺。现有 3D 统一模型要么任务有限，要么依赖 2D 投影。

**本文目标**：设计一个直接在点云上操作、支持尽可能多的 3D 视觉和视觉-语言任务的统一模型，实现跨任务的参数共享和无缝任务分解。

**切入角度**：借鉴 2D 领域的"功能统一"（functional unification）范式，而非 I/O 统一的 seq2seq 方式。通过 query-based transformer 生成通用的语义/掩码表示，再通过组合不同功能头来适配不同任务。

**核心 idea**：用 Query Transformer + Task Router 的功能统一架构，在点云上直接实现六类 3D 视觉-语言任务的统一建模。

## 方法详解

### 整体框架

Uni3DL 包含四个核心模块：

- **Text Encoder**：基于 CLIP tokenizer + transformer 的文本编码器，提取文本特征 $\mathbf{F}_T \in \mathbb{R}^{L_T \times C}$
- **Point Encoder**：基于 MinkowskiEngine 的稀疏 3D 卷积 U-Net（Res16UNet34C），输入带颜色的点云 $\mathbf{P} \in \mathbb{R}^{N_0 \times 6}$，输出多层级体素特征 $\{\mathbf{V}_s\}_{s=1}^{S}$
- **Query Transformer Module**：核心模块，让可学习的 latent queries $\mathbf{F}_Q \in \mathbb{R}^{Q \times C}$ 和 text queries 通过交叉注意力关注体素特征，生成任务无关的掩码输出 $\mathbf{O}_m$ 和语义输出 $\mathbf{O}_s$
- **Task Router**：包含多个功能头（分类头、掩码头、定位头、文本生成头、文本-3D 匹配头），不同任务通过组合不同头来完成

整体公式为：$\mathbf{O}_m, \mathbf{O}_s = \mathcal{D}(\langle \mathbf{F}_Q, \mathbf{F}_T \rangle, \mathbf{V})$

### 关键设计

1. **Point Cloud Encoder（3D U-Net）**：

    - 功能：从带颜色的点云中提取多尺度体素特征
    - 核心思路：输入点云量化为 $N_0$ 个体素，经过 $S$ 阶段的卷积-下采样-反卷积-上采样，得到不同分辨率的特征图 $\{\mathbf{V}_s \in \mathbb{R}^{N_s \times C}\}_{s=1}^{S}$。最后一层特征 $\mathbf{V}_S$ 用作逐点掩码计算的点嵌入，其余层 $\{\mathbf{V}_s\}_{s=1}^{S-1}$ 被送入 Query Transformer 增强 queries
    - 设计动机：U-Net 结构保留多尺度信息，有利于同时处理需要全局语义（如分类）和局部精度（如实例分割）的任务。使用预训练的 Mask3D 权重初始化，加速收敛

2. **Query Transformer Module（核心创新）**：

    - 功能：融合 latent queries、text queries 和 3D 视觉特征，生成统一的语义和掩码表示
    - 核心思路：$L=15$ 层 transformer decoder，每层包含：
        - **Masked Cross-Attention**：queries 与体素特征交叉注意力，采用 Mask2Former 的 masked attention 策略，每个 query 只关注上一层预测的掩码区域对应的体素：$\langle \hat{\mathbf{F}}_Q^l, \hat{\mathbf{F}}_T^l \rangle = \text{Cross-Att}(\langle \mathbf{F}_Q^{l-1}, \mathbf{F}_T^{l-1} \rangle, \mathbf{V}_s)$
        - **Self-Attention**：query 间的交互，让 latent queries 和 text queries 相互增强
        - **FFN**：标准前馈层
    - **Voxel Sampling**：由于不同场景点数不同，训练时对每个特征层采样固定数量体素，实现高效 batch 训练
    - 设计动机：masked attention 提升目标定位能力（只关注相关区域），latent queries 捕获对象级信息，text queries 捕获文本语义——两者在同一个 decoder 中联合优化

3. **Task Router（功能统一的关键）**：

    - 功能：通过组合不同功能头，从统一的语义/掩码输出中导出任务特定结果
    - 组合策略（Table 2）：
        - 语义分割 = 分类头 + 掩码头
        - 实例分割 = 分类头 + 掩码头
        - Grounded 分割 = 掩码头 + 定位头
        - 3D 描述生成 = 文本生成头
        - 文本-3D 检索 = 文本-3D 匹配头
    - 设计动机：不同任务共享底层 encoder 和 decoder 参数，仅在最后的 routing 策略上不同，实现了真正的参数共享和任务分解

4. **各功能头细节**：

    - **Object Classification Head**：取前 $Q$ 个语义输出 $\mathbf{O}_s$，将所有 $K+1$ 类名送入文本编码器得到类嵌入 $\mathbf{C}_{emb}$，分类概率 $\mathbf{O}_c = \mathbf{O}_s \cdot \mathbf{C}_{emb}^T$（开放词汇分类）
    - **Mask Head**：掩码输出与全分辨率体素特征做点积 $\mathbf{O}_m = \mathbf{O}_m \cdot \mathbf{V}_S^T$，得到每个 query 的逐点掩码
    - **Grounding Head**：计算文本嵌入与对象嵌入的相似度 $\mathbf{S}_t = \text{Softmax}(e^\eta \cdot \mathbf{T}_{emb} \cdot \mathbf{O}_s^T)$，$\eta$ 为可学习缩放参数；用 Hungarian matching 进行匹配。额外的轻量 MLP 网络预测文本描述中提及的物体类别
    - **Text Generation Head**：取最后 $L_T$ 个语义输出与词表 token 嵌入计算亲和矩阵 $\mathbf{S}_{cap} \in \mathbb{R}^{L_T \times V}$，训练时用 causal masking，推理时自回归生成
    - **Text-3D Matching Head**：取最后一个语义 token 作为形状嵌入，计算与文本嵌入的对比学习损失

### 损失函数 / 训练策略

总损失为五个任务损失之和：

$$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{mask} + \mathcal{L}_{grd} + \mathcal{L}_{cap} + \mathcal{L}_{ret}$$

- $\mathcal{L}_{cls} = \lambda_{cls} \cdot \text{CE}(\mathbf{O}_c, C_{gt})$：分类交叉熵
- $\mathcal{L}_{mask} = \lambda_{bce} \cdot \text{BCE} + \lambda_{dice} \cdot \text{DICE}$：掩码二元交叉熵 + dice loss
- $\mathcal{L}_{grd} = \lambda_{gc} \cdot \mathcal{L}_{gc} + \mathcal{L}_{gtxt} + \mathcal{L}_{gmask}$：定位匹配 + 类别存在性 + 掩码
- $\mathcal{L}_{cap} = \lambda_{cap} \cdot \text{CE}(\mathbf{S}_{cap}, y_{cap})$：描述生成
- $\mathcal{L}_{ret} = \lambda_{ret} \cdot \text{CL}(\mathbf{S}_{ret}, y_{ret})$：CLIP 式对比学习

权重设置：$\lambda_{cls}=2.0$，$\lambda_{bce}=5.0$，$\lambda_{dice}=5.0$，$\lambda_{gc}=0.4$，$\lambda_{cap}=\lambda_{ret}=2.0$

**训练流程**：
- 预训练：ScanNet(v2) + ScanRefer + Cap3D Objaverse 三数据集联合训练 50 epochs，4×A100 约 20 小时
- 微调：各下游任务分别微调 20-30 epochs，学习率 1e-4 或 1e-5
- 150 个 latent queries + 1 个场景级 query；体素大小：3D 扫描 0.02m，归一化 3D 形状 0.01

## 实验关键数据

### 主实验

Uni3DL 在 6 大任务上的表现（Table 3）：

| 任务 | 数据集 | 指标 | Uni3DL | 之前最佳 | 对比 |
|------|--------|------|--------|----------|------|
| 语义分割 | ScanNet Val | mIoU | **76.2** | 75.6 (Swin3D†) | +0.6 |
| 语义分割 | S3DIS Area5 | mIoU | 72.7 | 73.0 (Swin3D†) | -0.3 |
| 目标检测 | ScanNet Val | bAP50 | **67.7** | 63.9 (Mask-Att-Free†) | +3.8 |
| 实例分割 | ScanNet Val | mAP | **60.9** | 58.4 (Mask-Att-Free†) | +2.5 |
| Grounded 分割 | ScanRefer | mIoU/Acc@0.25 | 32.3/39.4 | 27.8/37.5 (TGNN) | +4.5/+1.9 |
| 3D 描述 | Cap3D | B-1/R/M | **31.6/33.1/14.4** | 12.6/15.0/16.0 | B-1 超 19+ |
| 文本-3D 检索 | Text2Shape | R@1/R@5 | 5.8/19.7 | 5.1/17.2 (P2W) | +0.7/+2.5 |

关键发现：
- 语义分割在 ScanNet 上达到最佳 mIoU 76.2，甚至超过使用额外数据的 Swin3D†
- 3D captioning 优势特别大：BLEU-1 和 ROUGE-L 超过之前最佳 20% 以上
- 在 grounded segmentation 上大幅超越唯一的竞品 TGNN
- 所有任务使用一个统一架构，而对比方法均为各任务专用模型

### 消融实验

**预训练效果（Table 4）**：

| 配置 | 语义分割 mIoU | 实例分割 mAP50 | Grounded Acc@0.25 | 检索 R@1 |
|------|--------------|---------------|-------------------|---------|
| From scratch | 72.3 | 61.7 | 33.8 | 2.4 |
| 预训练后微调 | **76.2** | **65.3** | **39.4** | **4.6** |
| 提升 | +3.9 | +3.6 | +5.6 | +2.2 |

**预训练任务组合（Table 5）**：

| 配置 | Grounded Acc@0.25 | Captioning R | Retrieval S2T R@1 |
|------|-------------------|-------------|-------------------|
| Full model | 37.8 | 18.6 | 8.0 |
| 去掉实例分割 | 33.8 (-4.0) | 17.8 | 4.0 |
| 去掉检索 | 37.7 | 15.8 (-2.8) | n/a |
| 去掉描述生成 | 37.9 | n/a | 3.5 (-4.5) |

### 关键发现

- 预训练对所有下游任务都有显著提升，特别是 grounded segmentation (+5.6 Acc@0.25) 和语义分割 (+3.9 mIoU)
- 任务间存在互利关系：实例分割帮助 grounded segmentation（共享实例理解能力），captioning 和 retrieval 互相促进（共享文本-3D 对齐表示）
- 去掉实例分割预训练后，grounded segmentation 和检索都显著下降——说明实例级理解是跨任务迁移的关键
- 零样本 3D 分类（ModelNet10: 70.4%, ModelNet40: 57.0%）虽不及使用 CLIP 的方法，但 Uni3DL 完全不依赖 2D 投影和预训练 foundation model

## 亮点与洞察

- **功能统一而非 I/O 统一**：不走 seq2seq 路线（预测 token 序列），而是让 decoder 输出异构结果（掩码 + 语义 + 文本），由 task router 灵活组合。这比 I/O 统一更适合密集预测任务，因为掩码输出本身就是高维连续量
- **直接操作点云**：摆脱对多视图投影的依赖，保留完整的 3D 几何信息。虽然代价是无法利用 CLIP 等强大的 2D 预训练模型，但换来了更简洁的架构和更好的几何理解
- **Query 设计的双轨制**：latent queries 捕获对象级信息（不需要语言输入），text queries 捕获语言语义——两者在同一个 decoder 中联合优化，天然实现了视觉-语言对齐

## 局限与展望

- **未利用 2D 预训练模型**：作者自己也承认，虽然直接操作点云避免了信息损失，但也放弃了利用 CLIP 等 2D 预训练模型的强大表示。未来可探索 hybrid 方案
- **数据规模受限**：3D 数据集（ScanNet 1.2K 场景、Cap3D 660K 对）远小于 2D 数据集（LAION 400M+），限制了统一模型的泛化能力
- **文本生成质量有限**：自回归文本生成使用自定义小型 transformer，生成质量远不如 LLM-based 方法。可考虑接入 LLM 做文本生成
- **检索性能一般**：在 Text2Shape 上的 R@1 仅 5.8%，远低于使用 part 标签的 Parts2Words (12.7%)。细粒度形状-文本匹配仍是挑战
- **场景规模**：主要在室内场景（ScanNet、S3DIS）上评估，未验证在大规模户外场景的有效性

## 相关工作与启发

- **vs X-Decoder**：X-Decoder 是 2D 领域的功能统一先驱，Uni3DL 将其思路成功迁移到 3D 点云。区别在于 3D 的体素采样策略和 3D U-Net backbone
- **vs 3D-VisTA**：3D-VisTA 也不依赖多视图图像做预训练，但需要为每个任务单独微调不同的 task head。Uni3DL 参数共享程度更高
- **vs PointLLM**：PointLLM 接入 Vicuna 做文本生成，生成质量更好但不支持密集预测。Uni3DL 的优势在于全任务覆盖
- **启发**：这种 encoder-decoder-router 的统一框架思路可以推广到其他模态（如音频-语言），关键是 decoder 输出要足够通用（语义 + 掩码的组合确实很灵活）

## 评分

- 新颖性: ⭐⭐⭐⭐ 功能统一范式在 3D 的首次全面落地，但各组件设计多为已有技术组合
- 实验充分度: ⭐⭐⭐⭐⭐ 6 大任务、5 个数据集、详细消融，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Table 1 的任务-方法对比一目了然
- 价值: ⭐⭐⭐⭐ 为 3D 统一建模提供了强 baseline，但未利用 2D 预训练限制了实际竞争力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] UniCode: Learning a Unified Codebook for Multimodal Large Language Models](unicode_learning_a_unified_codebook_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] OneCAT: Decoder-Only Auto-Regressive Model for Unified Understanding and Generation](../../CVPR2026/multimodal_vlm/onecat_decoder-only_auto-regressive_model_for_unified_understanding_and_generati.md)
- [\[ICLR 2026\] SR-3D: 3D-Aware Region Prompted Vision Language Model](../../ICLR2026/multimodal_vlm/3d_aware_region_prompted_vision_language_model.md)
- [\[ICLR 2026\] UniHM: Unified Dexterous Hand Manipulation with Vision Language Model](../../ICLR2026/multimodal_vlm/unihm_unified_dexterous_hand_manipulation_with_vision_language_model.md)
- [\[CVPR 2026\] Rosetta Stone for Unified MLLMs: A Unified Tokenizer to Decipher Understanding and Generation](../../CVPR2026/multimodal_vlm/rosetta_stone_for_unified_mllms_a_unified_tokenizer_to_decipher_understanding_an.md)

</div>

<!-- RELATED:END -->
