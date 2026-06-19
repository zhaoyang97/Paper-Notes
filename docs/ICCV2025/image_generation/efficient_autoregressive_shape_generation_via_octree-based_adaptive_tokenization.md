---
title: >-
  [论文解读] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization
description: >-
  [ICCV 2025][图像生成][3D 形状生成] OAT 提出基于二次误差度量（quadric error）的自适应八叉树 tokenization，根据局部几何复杂度动态分配 token 预算，在减少 50% token 的同时保持重建质量，并在此基础上构建 OctreeGPT 实现高质量文本到 3D 生成。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "3D 形状生成"
  - "八叉树"
  - "自适应 tokenization"
  - "自回归模型"
  - "向量量化"
---

# Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization

**会议**: ICCV 2025

**arXiv**: [2504.02817](https://arxiv.org/abs/2504.02817)

**领域**: 3D 生成/图像生成

**关键词**: 3D 形状生成, 八叉树, 自适应 tokenization, 自回归模型, 向量量化

## 一句话总结

OAT 提出基于二次误差度量（quadric error）的自适应八叉树 tokenization，根据局部几何复杂度动态分配 token 预算，在减少 50% token 的同时保持重建质量，并在此基础上构建 OctreeGPT 实现高质量文本到 3D 生成。

## 研究背景与动机

- **固定长度表示的浪费**：现有 3D 形状 VAE（如 3DShape2VecSet、Craftsman）将所有形状编码为固定大小的 latent 向量，不论形状复杂度如何。简单立方体和精细雕塑使用相同的表示容量，导致简单形状浪费 token，复杂形状表示不足
- **传统八叉树的局限**：虽然 OctFusion、XCube 等方法使用稀疏体素/八叉树考虑了稀疏性，但它们仍然将所有包含表面几何的 cell 细分到最深层级。一个简单的立方体（8 个顶点）需要与精细雕塑相似的节点数
- **核心观察**：3D 数据天然稀疏，信息集中在表面；不同物体和同一物体的不同区域具有显著不同的几何复杂度，需要自适应的表示策略

## 方法详解

### 整体框架

OAT 包含两个核心组件：(1) 基于 quadric error 的自适应八叉树 tokenization；(2) OctreeGPT 自回归生成模型。流程为：输入网格 -> 采样点云 -> 构建自适应八叉树 -> Perceiver 编码为变长 latent -> 残差量化 -> OctreeGPT 生成。

### 关键设计

**1. 基于 Quadric Error 的自适应八叉树构建**

借鉴网格简化中的二次误差度量（QEM），衡量每个 cell 内局部几何复杂度。对每个 octree cell，计算 cell 内所有采样点的 quadric 矩阵之和，然后求最小化二次误差。细分条件：(1) 未达最大深度 L；(2) 平均 quadric error 超过预设阈值 T。直觉上，当局部几何接近平面时误差趋近零（无需细分），复杂区域误差较高（需继续细分）。

**2. Perceiver-based 自适应编码**

- 使用位置编码（PE）+ 尺度编码（SE）表示每个叶节点
- 叶节点通过 cross-attention 全局关注所有表面点（而非仅局部 cell 内的点），获取长距离依赖
- 后续 self-attention 内叶节点间信息交换
- 从叶节点向祖先节点自底向上聚合 latent（取子节点均值）

**3. 多尺度八叉树残差量化**

核心思想：子节点只需编码相对于父节点的残差 latent，而非完整表示。

- 根节点直接量化
- 子节点量化残差：Quantize(phi(v) - z_acc(v_parent))
- 维护累积 latent z_acc(v) = z_acc(v_parent) + z(v)
- 使用共享 codebook 和量化函数

**4. OctreeGPT 自回归生成**

- 以广度优先顺序遍历八叉树节点，每个 token 包含量化索引 q(v_i) 和结构编码 chi(v_i)（8-bit 二进制）
- 树结构位置编码编码 3D 坐标和层级深度信息
- 双预测头：同时预测 latent token 和结构代码
- 结构代码的 8-bit 编码天然表示变长生成的终止条件
- 文本条件：前置 77 个 CLIP embedding token

### 损失函数

- VAE 训练：占据场重建（BCE 损失）+ lambda_VQ x VQ 损失
- GPT 训练：两个交叉熵损失（latent token 分类 + 结构代码分类），均为 256-way

## 实验关键数据

### 形状重建（离散 latent）

| 方法 | 平均 Token 数 | IoU | CD (x1e-3) |
|------|------------|------|-------------|
| Craftsman-VQ | 512 | 83.8 | 1.94 |
| OAT (w/o 自适应) | 607 | 88.3 | 1.85 |
| **OAT** | **439** | **88.6** | **1.78** |
| Craftsman-VQ | 1024 | 84.4 | 1.80 |
| **OAT** | **625** | **89.7** | **1.53** |

### 形状生成

| 方法 | FID | KID (x1e-3) | CLIP | 时间(s) |
|------|------|-------------|-------|---------|
| Craftsman (image-to-3D) | 65.18 | 6.42 | 0.27 | 54.8 |
| XCube | 132.56 | 9.83 | 0.23 | 32.3 |
| Craftsman-VQ + GPT | 85.10 | 7.49 | 0.26 | 15.4 |
| **OctreeGPT** | **56.88** | **5.79** | **0.34** | **11.3** |

### 消融实验

- 无自适应细分时，传统八叉树对简单大体积物体浪费 token（512 -> 607 token），对小体积物体偶尔节省
- 随 token 数增加，OAT 始终优于同等 token 数的基线
- 连续 latent（OAT-KL）版本在 439 token 下 IoU 91.6、CD 1.29，优于 Craftsman 512 token 的 IoU 91.0、CD 1.83

## 亮点与洞察

1. **QEM 引导的自适应策略**：将网格简化中成熟的 quadric error 度量创新性地应用于 octree 构建，优雅且计算高效（仅需解线性系统）
2. **残差量化与树结构的完美结合**：利用八叉树的层级关系做残差编码，子节点只编码增量信息，大幅压缩 latent 空间
3. **变长生成的自然表达**：八叉树结构代码的存在使模型可以自然决定何时停止细分，类似文本生成中的 EOS token
4. **50% token 压缩 + 质量提升 + 最快推理**：同时在效率和效果上实现双赢，实际部署价值高
5. **兼容连续/离散 latent**：通过切换 VQ/KL bottleneck 灵活适应不同下游任务
6. **全局注意力而非局部**：编码器的 cross-attention 让每个叶节点关注整个形状的表面点，有效捕获全局几何

## 局限性

- 仅处理几何形状重建和生成，**未涉及纹理信息**
- 自适应细分阈值 T 需要预定义，不同数据集可能需要不同设置
- Objaverse 数据集经过大量筛选（800K -> 207K），对低质量网格的鲁棒性未经验证
- 与 XCube、OctFusion 的对比使用了不同训练数据的预训练模型，公平性有一定局限

## 相关工作

- **Craftsman** [Li et al., 2024]：固定长度 vector set 表示，OAT 的直接基线
- **OctFusion** [Xiong et al., 2024]：八叉树 + 扩散模型，但使用均匀占据细分
- **XCube** [Ren et al., 2024]：稀疏体素特征，也是均匀细分策略
- **3DShape2VecSet** [Zhang et al., 2023]：开创了将 3D 形状编码为 latent vector set 的范式
- **Perceiver** [Jaegle et al., 2021]：query-based transformer，用于灵活长度的编解码

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | 4/5 |
| 有效性 | 4/5 |
| 实用性 | 5/5 |
| 清晰度 | 4/5 |
| 综合 | 4/5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)
- [\[CVPR 2025\] Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction](../../CVPR2025/image_generation/efficient_long_video_tokenization_via_coordinate-based_patch_reconstruction.md)
- [\[CVPR 2025\] Language-Guided Image Tokenization for Generation](../../CVPR2025/image_generation/language-guided_image_tokenization_for_generation.md)
- [\[ICCV 2025\] Holistic Tokenizer for Autoregressive Image Generation](holistic_tokenizer_for_autoregressive_image_generation.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](grouped_speculative_decoding_for_autoregressive_image_generation.md)

</div>

<!-- RELATED:END -->
