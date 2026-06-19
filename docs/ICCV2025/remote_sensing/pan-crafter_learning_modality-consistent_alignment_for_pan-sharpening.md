---
title: >-
  [论文解读] Pan-Crafter: Learning Modality-Consistent Alignment for Pan-Sharpening
description: >-
  [ICCV 2025][遥感][全色锐化] PAN-Crafter 提出模态一致性对齐框架，通过模态自适应重建（MARs）和跨模态对齐感知注意力（CM3A）显式处理 PAN 和 MS 图像的跨模态错位问题，在多个遥感基准数据集上达到 SOTA，且推理速度比扩散模型快 1110×：。 遥感中的全色锐化（Pan-sharpeni…
tags:
  - "ICCV 2025"
  - "遥感"
  - "全色锐化"
  - "跨模态对齐"
  - "模态自适应重建"
  - "注意力机制"
  - "遥感图像融合"
---

# Pan-Crafter: Learning Modality-Consistent Alignment for Pan-Sharpening

**会议**: ICCV 2025  
**arXiv**: [2505.23367](https://arxiv.org/abs/2505.23367)  
**代码**: [https://kaist-viclab.github.io/PAN-Crafter_site](https://kaist-viclab.github.io/PAN-Crafter_site)  
**领域**: 遥感 / 全色锐化  
**关键词**: 全色锐化, 跨模态对齐, 模态自适应重建, 注意力机制, 遥感图像融合

## 一句话总结

PAN-Crafter 提出模态一致性对齐框架，通过模态自适应重建（MARs）和跨模态对齐感知注意力（CM3A）显式处理 PAN 和 MS 图像的跨模态错位问题，在多个遥感基准数据集上达到 SOTA，且推理速度比扩散模型快 **1110×**。

## 研究背景与动机

遥感中的全色锐化（Pan-sharpening）旨在融合高分辨率全色（PAN，单通道）图像和低分辨率多光谱（MS，多通道）图像，生成高分辨率多光谱（HRMS）输出。这是实际卫星图像处理的核心需求。

**核心痛点**：由于传感器位置差异、采集时间偏差和分辨率不匹配，PAN 和 MS 图像之间存在**跨模态空间错位**。然而：

**现有方法假设完美对齐**：大多数深度学习方法使用逐像素损失（$\ell_1$/$\ell_2$），在错位存在时导致光谱失真、双边缘和模糊

**自适应修复不够灵活**：SIPSA 使用固定尺度对齐、LAGConv/CANConv 使用自相似性聚合而非显式几何对齐

**扩散模型虽质量好但过慢**：PanDiff 和 TMDiff 推理时间分别为 2.955s 和 9.997s，难以部署

**切入角度**：设计一个双向对齐机制——不仅将 PAN 结构对齐到 MS 纹理（用于 HRMS 重建），还反过来将 MS 纹理对齐到 PAN 结构（通过 PAN 回重建作为辅助自监督），形成"模态一致性"约束。

## 方法详解

### 整体框架

PAN-Crafter 采用 U-Net 编解码器架构，包含两大核心模块：
1. **MARs**（模态自适应重建）：一个网络同时学习重建 HRMS 和反重建 PAN
2. **CM3A**（跨模态对齐感知注意力）：多尺度双向对齐 PAN 和 MS 特征

### 关键设计

#### 1. 模态自适应重建（MARs）

MARs 让一个网络在两种模式下交替工作：

- **MS 模式**：输入 PAN + LRMS → 输出 HRMS
  $$\hat{\mathbf{I}}_{\text{ms}}^{\text{hr}} = \mathcal{P}_\theta(\mathbf{I}_{\text{pan}}, \mathbf{I}_{\text{ms}}^{\text{lr}}; \text{mode}=\mathsf{MS}) + \mathbf{I}_{\text{ms}}^{\text{lr}}$$

- **PAN 模式**：输入 PAN + LRMS → 回重建 PAN（复制至多通道）
  $$\hat{\mathbf{I}}_{\text{pan}}^{\text{rep}} = \mathcal{P}_\theta(\mathbf{I}_{\text{pan}}, \mathbf{I}_{\text{ms}}^{\text{lr}}; \text{mode}=\mathsf{PAN}) + \mathbf{I}_{\text{pan}}^{\text{lr,rep}}$$

**设计动机**：PAN 回重建作为辅助自监督信号，其优势在于PAN 图像本身就是已知的"ground truth"——无需额外标注。通过强制网络也能重建出清晰的 PAN 图像，迫使其学到更锐利的空间结构，这些知识反过来提升 HRMS 的空间细节质量。

训练时每个 batch 被复制为两份，一份走 MS 模式，另一份走 PAN 模式。推理时只用 MS 模式。

MARs 损失：
$$\mathcal{L}_{\text{MARs}} = \|\hat{\mathbf{I}}_{\text{ms}}^{\text{hr}} - \mathbf{I}_{\text{ms}}^{\text{hr}}\|_1 + \lambda \|\hat{\mathbf{I}}_{\text{pan}}^{\text{rep}} - \mathbf{I}_{\text{pan}}^{\text{rep}}\|_1$$

#### 2. 跨模态对齐感知注意力（CM3A）

CM3A 是 PAN-Crafter 的另一核心创新，在多尺度特征级别进行双向跨模态对齐。

**关键设计**：用局部注意力（$k \times k$ 窗口，$k=3$）替代全局注意力，因为 PAN-MS 对通常已经大致预对齐，只需局部修正。这将计算复杂度从 $O(2(HW)^2 C)$ 降至 $O(2(HW)k^2 C)$。

在 **MS 模式**下，查询由 LRMS 特征构建，同时执行：
- **自注意力**：Query 与 MS key-value 交互，保持 MS 特征一致性
- **对齐注意力**：Query 与 PAN key-value 交互，引入 PAN 的结构信息

$$\mathbf{x}_{\text{ms}} = \text{LocalAttn}(\mathbf{Q}, \mathbf{K}_{\text{ms}}, \mathbf{V}_{\text{ms}})$$
$$\mathbf{x}_{\text{pan}} = \text{LocalAttn}(\mathbf{Q}, \mathbf{K}_{\text{pan}}, \mathbf{V}_{\text{pan}})$$

在 **PAN 模式**下操作镜像——Query 由 PAN 特征构建，确保 PAN 回重建的结构一致性。

**创新点**：用下采样的原始图像替代传统的固定位置编码，拼接到 Q/K 中，隐式学习模态间的相对错位。

#### 3. 模态调制（Modulate）

ResBlock 中通过可学习的 $\gamma, \beta$ 参数根据 MARs mode 调制特征：

$$\mathsf{Modulate}(\mathbf{x}; \mathsf{MS}): \mathbf{x} \leftarrow (1 + \gamma_{\text{ms}}) \odot \mathbf{x} + \beta_{\text{ms}}$$

这确保同一网络能适应不同模态的特征分布。

### 损失函数 / 训练策略

- 使用 AdamW 优化器，初始学习率 $1 \times 10^{-4}$，余弦退火
- Batch size 48（因 MARs 双份有效 batch = 96）
- 训练 50K 迭代，100 步 warmup
- $\lambda = 1.0$，特征维度 $C = 128$

## 实验关键数据

### 主实验

**WV3 数据集性能对比**：

| 方法 | HQNR↑ | ERGAS↓ | PSNR↑ | 推理时间↓(s) | 内存↓(GB) |
|-----|-------|--------|-------|------------|----------|
| CANConv (CVPR24) | 0.951 | 2.163 | 37.441 | 0.451 | 2.713 |
| PanDiff (TGRS23) | 0.952 | 2.276 | 37.029 | 2.955 | 2.328 |
| TMDiff (TGRS24) | 0.924 | 2.151 | 37.477 | 9.997 | 9.910 |
| **PAN-Crafter** | **0.958** | **2.040** | **37.956** | **0.009** | **1.711** |

PAN-Crafter 比 TMDiff 快 **1110×**，比 CANConv 快 **50×**，同时在所有核心指标上达到最佳。

**GF2 数据集**：PSNR 达到 **45.076**，比次优的 CANConv (43.166) 高近 2 dB。

### 消融实验

| CM3A | MARs | HQNR↑ | ERGAS↓ | PSNR↑ | 时间(s) |
|------|------|-------|--------|-------|--------|
| ✗ | ✗ | 0.948 | 2.232 | 37.245 | 0.006 |
| ✗ | ✓ | 0.956 | 2.122 | 37.602 | 0.009 |
| ✓ | ✗ | 0.949 | 2.212 | 37.285 | 0.007 |
| **✓** | **✓** | **0.958** | **2.040** | **37.956** | **0.009** |

关键发现：
- MARs 的贡献大于 CM3A 单独使用（PSNR: +0.357 vs +0.040）
- 但两者组合产生**协同效应**——组合增量（+0.711）远大于各自增量之和

### 关键发现

1. **零样本泛化**：在未见过的 WV2 卫星数据上，PAN-Crafter HQNR 达到 **0.942**，大幅领先其他方法
2. **MARs 的自监督效果**：PAN 回重建作为辅助任务，显著提升了主任务的空间锐度
3. **CM3A + MARs 协同**：MARs 使双向交互成为可能，CM3A 在此基础上实现更精确的对齐
4. **效率优势**：整个框架仅 7.17M 参数，79.03G FLOPs

## 亮点与洞察

- **PAN 回重建作为自监督**：巧妙地利用了 PAN 图像本身作为免费的监督信号，无需额外标注
- **局部注意力替代位置编码**：用原始图像特征替代传统 PE，更适合处理不确定的空间错位
- **实用性极强**：在全面超越扩散模型的同时，推理速度快 3 个数量级，非常适合实际遥感处理流水线

## 局限与展望

- 未显式处理多光谱波段之间的错位（目前仅处理 PAN 和 MS 之间的错位）
- 使用深度可分离卷积处理波段间对齐是一个潜在改进方向（论文中已提到）
- 局部注意力窗口大小 $k=3$ 可能对大幅错位不够

## 相关工作与启发

- **CANConv** [CVPR 2024]：基于聚类的空间自适应卷积，但依赖 k-means 导致速度慢
- **SIPSA** [2022]：首次将错位识别为全色锐化的关键挑战
- **PanDiff / TMDiff**：扩散模型方法，质量好但推理代价过高
- 启发：双任务自监督（主任务 + 逆任务）的思路可推广到其他跨模态融合任务

## 评分

- 新颖性：⭐⭐⭐⭐ — MARs 双向重建和 CM3A 对齐机制设计新颖
- 技术深度：⭐⭐⭐⭐ — 模态调制、局部注意力替代 PE 等细节考虑周到
- 实验充分度：⭐⭐⭐⭐⭐ — 4 个卫星数据集 + 零样本泛化 + 效率对比
- 实用性：⭐⭐⭐⭐⭐ — 速度快、内存小、性能强，极适合实际部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Multigrain-aware Semantic Prototype Scanning and Tri-Token Prompt Learning Embraced High-Order RWKV for Pan-Sharpening](../../CVPR2026/remote_sensing/multigrain-aware_semantic_prototype_scanning_and_tri-token_prompt_learning_embra.md)
- [\[ICCV 2025\] WildSAT: Learning Satellite Image Representations from Wildlife Observations](wildsat_learning_satellite_image_representations_from_wildlife_observations.md)
- [\[NeurIPS 2025\] C3PO: Cross-View Cross-Modality Correspondence by Pointmap Prediction](../../NeurIPS2025/remote_sensing/c3po_cross-view_cross-modality_correspondence_by_pointmap_prediction.md)
- [\[CVPR 2025\] Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning](../../CVPR2025/remote_sensing/hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning.md)
- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](../../CVPR2026/remote_sensing/cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)

</div>

<!-- RELATED:END -->
