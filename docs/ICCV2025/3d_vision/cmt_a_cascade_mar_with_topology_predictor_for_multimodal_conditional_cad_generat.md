---
title: >-
  [论文解读] CMT: A Cascade MAR with Topology Predictor for Multimodal Conditional CAD Generation
description: >-
  [ICCV 2025][3D视觉][CAD 生成] 提出 CMT，首个基于 B-Rep 表示的多模态 CAD 生成框架，通过级联 MAR（先边后面）和拓扑预测器实现精确拓扑和几何生成，并构建了 130 万级多模态 CAD 数据集 mmABC。 传统 CAD 设计需要人工执行 2D 草图→3D 操作→B-Rep 转换的复杂流程…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "CAD 生成"
  - "B-Rep"
  - "级联自回归"
  - "拓扑预测"
  - "多模态条件生成"
---

# CMT: A Cascade MAR with Topology Predictor for Multimodal Conditional CAD Generation

**会议**: ICCV 2025  
**arXiv**: [2504.20830](https://arxiv.org/abs/2504.20830)  
**代码**: 无（数据集 mmABC 开放）  
**领域**: 3D 视觉 / CAD 生成  
**关键词**: CAD 生成, B-Rep, 级联自回归, 拓扑预测, 多模态条件生成

## 一句话总结

提出 CMT，首个基于 B-Rep 表示的多模态 CAD 生成框架，通过级联 MAR（先边后面）和拓扑预测器实现精确拓扑和几何生成，并构建了 130 万级多模态 CAD 数据集 mmABC。

## 研究背景与动机

传统 CAD 设计需要人工执行 2D 草图→3D 操作→B-Rep 转换的复杂流程，耗时且需专业技能。理想的 CAD 生成工具应同时满足：(1) 拓扑和几何精确性；(2) 支持多模态条件输入（文本、图像、点云）；(3) 对非专业用户友好。

现有方法存在根本性局限：
- **MLLM 方法**（DeepCAD、CAD-MLLM、SolidGen）：将 CAD 序列化为离散 token 逐个生成，但受限于离散表示——仅支持线段/圆弧+拉伸操作，无法建模倒角、圆角和自由曲面
- **扩散方法**（BrepGen）：可以生成连续 B-Rep 模型，但难以处理多模态条件输入

新提出的 MAR (Masked AutoRegressive) 架构统一了自回归的 token 依赖建模与扩散的连续分布学习，为"精确+多模态"的 CAD 生成提供了机会。但原始 MAR 缺乏捕获 B-Rep 拓扑关系（顶点-边-面层级）的内在机制。

## 方法详解

### 整体框架

CMT 由四部分组成：(1) B-Rep 连续 tokenization；(2) 统一多模态条件编码器；(3) 级联自回归生成网络（Edge MAR → Surface MAR）；(4) 拓扑预测器。整体流程为：B-Rep → 连续 token 化 → 条件编码 → 级联生成边→面 → 预测拓扑关系 → 完整 B-Rep。

### 关键设计

1. **连续 Token 化 (Continuous Tokenization)**：

    - **面 token**：通过 Surface VAE 编码面的几何信息（均匀采样点），拼接面包围盒坐标作为拓扑特征
    - **边 token**：通过 Edge VAE 编码边的几何信息，拼接包围盒坐标和相邻顶点坐标。顶点信息集成在边 token 中，无需单独生成
    - **排序**：按包围盒 3D 坐标 $(x_1, y_1, z_1, x_2, y_2, z_2)$ 升序排列，实现确定性序列化

2. **级联自回归网络 (Cascade Autoregressive Network, CAN)**：
   遵循 B-Rep 的拓扑先验"边围成面"，设计先边后面的两阶段生成：
    - **Edge MAR**：给定条件嵌入 $Z$ 和可见边 token $E_v$，通过 edge transformer $\mathcal{G}_e$ 生成被掩码边 token 的特征 $c_e$，再通过 edge diffusion MLP $\mathcal{D}_e$ 去噪得到边 token
    - **Surface MAR**：将变长边序列通过自注意力 + 可学习边 token 的 edge projector 压缩为定长边条件嵌入 $Q$，再结合条件嵌入 $Z$ 和可见面 token $S_v$，通过 surface transformer $\mathcal{G}_s$ + surface diffusion $\mathcal{D}_s$ 生成面 token
   
   级联设计使得面生成可以利用已有边信息，降低建模难度。

3. **拓扑预测器 (Topology Predictor)**：
   使用简单的 cross-attention 层直接从生成的边 token $\hat{E}$ 和面 token $\hat{S}$ 预测邻接矩阵 $A \in \mathbb{R}^{N_e \times N_s}$，阈值 $\tau=0.5$ 判定边-面邻接关系。比 Point2CAD 的后处理算法快 **4200 倍**（256 个 CAD 模型：0.038s vs 161.15s）。

4. **统一多模态条件编码器**：使用 CLIP tokenizer 处理文本/图像，3D 卷积处理点云，统一通过冻结 CLIP-ViT 编码器提取特征，再通过可学习 projector 生成定长条件嵌入 $Z$。

### 损失函数 / 训练策略

总损失 $L = L_{edge} + L_{surf} + L_{topo}$，其中：
- $L_{edge}$ 和 $L_{surf}$：扩散去噪损失，约束预测噪声与真实噪声的 MSE
- $L_{topo}$：邻接矩阵预测的 MSE 损失

训练：无条件生成 2100 epochs + 条件生成 1000 epochs，8×A100 GPU。序列最大长度：DeepCAD 上边 64/面 32，ABC 上边 128/面 64。推理时默认每步生成 1 个 token。

## 实验关键数据

### 主实验

**无条件生成 (DeepCAD & ABC)**：

| 数据集 | 方法 | COV↑ | MMD↓ | JSD↓ | Valid↑ |
|--------|------|------|------|------|--------|
| DeepCAD | DeepCAD | 65.46 | 1.29 | 1.67 | 46.1 |
| DeepCAD | SolidGen | 71.03 | 1.08 | 1.31 | 60.3 |
| DeepCAD | BrepGen | 71.26 | 1.04 | 0.09 | 62.9 |
| DeepCAD | **CMT** | **75.71** | **0.92** | 1.02 | **70.1** |
| ABC | BrepGen | 57.92 | 1.35 | 3.69 | 48.2 |
| ABC | **CMT** | **68.60** | 1.35 | **2.79** | **58.5** |

**条件生成 (mmABC)**：

| 条件 | 方法 | Chamfer↓ | F-score↑ | Normal C↑ |
|------|------|----------|----------|-----------|
| 点云 | DeepCAD | 5.11 | 83.56 | 69.58 |
| 点云 | NVDNet (重建) | 0.77 | 98.17 | 94.36 |
| 点云 | **CMT** | **0.64** | **99.07** | **95.48** |
| 图像 | InstantMesh | 6.18 | 84.71 | 52.72 |
| 图像 | **CMT** | **2.17** | **92.93** | **70.14** |
| 文本 | Michelangelo | 25% (GPT4o win) | 20% (Human win) | - |
| 文本 | **CMT** | **75%** (GPT4o) | **80%** (Human) | - |

### 消融实验

| 级联 | 采样步数 | COV↑ | Valid↑ |
|------|---------|------|--------|
| ✓ | 64/32 | **75.71** | **70.1** |
| ✗ | 64+32 | 65.80 | 47.17 |
| ✓ | 32/16 | 74.97 | 67.93 |
| ✓ | 16/8 | 70.26 | 47.80 |
| ✓ | 8/4 | N.A. | 1.20 |
| ✓ | 1/1 | N.A. | 0.10 |

### 关键发现

- 级联网络带来 **+9.91% Coverage 和 +22.96% Valid**，证明"先边后面"的范式对 B-Rep 生成至关重要
- CMT 在点云条件生成上甚至**超越 SOTA 重建方法** NVDNet（+0.90 F-score, +1.12 Normal C）
- 采样步数与质量强相关：1/1 步（一次生成所有 token）Valid 仅 0.10%，逐 token 生成达 70.1%
- 拓扑预测器速度是后处理算法的 4200 倍
- mmABC 是目前最大的多模态 B-Rep 数据集（130 万+），支持自由曲面建模

## 亮点与洞察

1. **首个多模态 B-Rep 生成框架**：CMT 填补了"精确 B-Rep + 多模态条件"的空白
2. **级联设计符合领域知识**：边围成面的拓扑先验被自然嵌入到架构中
3. **MAR 的成功应用**：将 MAR 从图像生成扩展到 CAD 生成，展示了其广泛适用性
4. **大规模数据集贡献**：mmABC 的构建（多体分解 + 去重 + VLM 标注）为未来 CAD 生成研究奠定基础

## 局限与展望

- 最大序列长度有限（边 128/面 64），仅覆盖 95% 数据，更复杂模型无法生成
- 文本条件的评估依赖人工和 VLM 打分，缺乏自动化定量指标
- 数据集的文本描述由 VLM 自动生成，可能存在质量不一致
- 未支持多体生成和模型补全等更复杂任务（论文明确提到作为未来工作）
- 推理速度：逐 token 生成的采样步数等于序列长度，速度可能成为瓶颈

## 相关工作与启发

- BrepGen：当前 B-Rep 生成的 SOTA，CMT 在所有指标上大幅超越
- MAR (Kaiming He)：CMT 的架构基础，成功展示了 MAR 在非图像领域的应用
- Point2CAD：从点云重建 CAD 的方法，其后处理拓扑算法被 CMT 的拓扑预测器直接替代
- OneLLM：统一多模态编码器的灵感来源

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个多模态 B-Rep 生成框架，级联设计和拓扑预测器创新性强
- **实验充分度**: ⭐⭐⭐⭐ 无条件+三种条件生成、消融充分、对比全面
- **写作质量**: ⭐⭐⭐⭐ 框架图清晰，流程描述条理分明
- **价值**: ⭐⭐⭐⭐⭐ 数据集+方法双重贡献，对 CAD/工业设计领域有重大意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians](sheap_self-supervised_head_geometry_predictor_learned_via_2d_gaussians.md)
- [\[CVPR 2025\] CADDreamer: CAD Object Generation from Single-view Images](../../CVPR2025/3d_vision/caddreamer_cad_object_generation_from_single-view_images.md)
- [\[ICCV 2025\] AnyI2V: Animating Any Conditional Image with Motion Control](anyi2v_animating_any_conditional_image_with_motion_control.md)
- [\[CVPR 2025\] MAR-3D: Progressive Masked Auto-regressor for High-Resolution 3D Generation](../../CVPR2025/3d_vision/mar-3d_progressive_masked_auto-regressor_for_high-resolution_3d_generation.md)
- [\[ICCV 2025\] Zero-Shot Inexact CAD Model Alignment from a Single Image](zero-shot_inexact_cad_model_alignment_from_a_single_image.md)

</div>

<!-- RELATED:END -->
