---
title: >-
  [论文解读] CoLA: Conditional Dropout and Language-Driven Robust Dual-Modal Salient Object Detection
description: >-
  [ECCV2024][语义分割][目标检测] 提出 CoLA 框架，通过语言驱动的质量评估（LQA）和条件性 Dropout（CD）两个核心模块，首次在双模态显著性目标检测中同时解决噪声输入和模态缺失两大鲁棒性问题。 现有痛点 现有痛点：领域现状：双模态显著性目标检测（SOD）利用 RGB 图像与辅助模态（深度/热红外）来检…
tags:
  - "ECCV2024"
  - "语义分割"
  - "目标检测"
  - "Modality Robustness"
  - "CLIP"
  - "Conditional Dropout"
  - "Quality Assessment"
---

# CoLA: Conditional Dropout and Language-Driven Robust Dual-Modal Salient Object Detection

**会议**: ECCV2024  
**arXiv**: [2407.06780](https://arxiv.org/abs/2407.06780)  
**代码**: 无  
**领域**: 图像分割  
**关键词**: Dual-modal Salient Object Detection, Modality Robustness, CLIP, Conditional Dropout, Quality Assessment

## 一句话总结

提出 CoLA 框架，通过语言驱动的质量评估（LQA）和条件性 Dropout（CD）两个核心模块，首次在双模态显著性目标检测中同时解决噪声输入和模态缺失两大鲁棒性问题。

## 背景与动机

### 现有痛点

**现有痛点**：**领域现状**：双模态显著性目标检测（SOD）利用 RGB 图像与辅助模态（深度/热红外）来检测场景中最显著的目标。现有方法的高精度依赖于高质量且完整的输入，但实际部署中面临两个核心挑战：

1. **输入噪声**：通信故障等原因导致 RGB 或辅助模态图像质量退化，传统方法会产生不理想的预测结果
2. **模态缺失**：设备故障导致某一模态完全不可用，现有模型因过度依赖完整输入而性能急剧下降

现有质量评估方法要么依赖固定参数的预训练网络（无法适应目标数据集），要么使用不精确的伪标签。而处理模态缺失的直接 dropout 方法虽在缺失场景有效，但会显著损害完整模态下的性能。

### 解决思路

**本文目标**：1. 如何在不需要额外质量标注的情况下，自适应地评估各模态输入的质量并重新标定其贡献？
2. 如何在增强模态缺失鲁棒性的同时，不牺牲完整模态下的检测精度？

## 方法详解

CoLA 采用两阶段训练架构，包含四个组件：双分支编码器、LQA 模块、条件性 Dropout 编码器、解码器。

### 阶段一：Language-driven Quality Assessment (LQA)

LQA 利用预训练的 CLIP 视觉-语言模型进行模态质量评估：

- 将双模态图像分别送入 CLIP 图像编码器，得到图像嵌入 $\varepsilon_i \in \mathbb{R}^{1 \times D}$（$D=512$）
- 文本编码器接收固定 prompt "A photo of high quality." 生成文本嵌入 $\varepsilon_t$
- 引入可学习 prompt $\omega$ 添加到文本嵌入上，实现对目标数据集的参数高效微调
- 通过余弦相似度计算各模态的质量分数 $\alpha^{m_1}, \alpha^{m_2}$
- 按质量分数加权融合各层特征：$g_j = g_j^{m_1} \cdot \frac{\alpha^{m_1}}{\alpha^{m_1}+\alpha^{m_2}} + g_j^{m_2} \cdot \frac{\alpha^{m_2}}{\alpha^{m_1}+\alpha^{m_2}}$

与传统方法相比，LQA 既保持了预训练模型的泛化能力，又能适配目标数据集。

### 阶段二：Conditional Dropout (CD)

受条件控制启发，CD 将模态缺失视为条件，避免直接 dropout 带来的性能退化：

- 冻结阶段一训练好的编码器参数 $\theta$
- 复制编码器得到可训练副本（参数 $\theta_f$），通过零卷积 $\mathcal{Z}$（权重和偏置初始化为零）连接
- 训练时从三种条件中随机选择输入：完整双模态 $\{m_1, m_2\}$、缺失模态二 $\{m_1, \phi\}$、缺失模态一 $\{\phi, m_2\}$
- 最终特征：$g = \mathcal{F}(\rho(\mathcal{M});\theta) + \mathcal{Z}(\mathcal{F}(\rho(\mathcal{M});\theta_f);\theta_z)$

零初始化确保阶段二训练初期对原始模型无影响，新学到的特征逐步融合。冻结的第一项保留已有知识，可训练的第二项学习更细粒度的单模态表征。

### 训练目标

采用 BCE loss 和 IoU loss 的组合：$\mathcal{L}_{total} = \mathcal{L}_{bce}(pred, GT) + \mathcal{L}_{iou}(pred, GT)$

## 实验关键数据

### RGB-T 显著性检测（VT821/VT1000/VT5000）

| 条件 | 指标 | TAGFNet | CoLA (本文) |
|------|------|---------|------------|
| 完整模态 (VT5000) | $E_m$ / $F_\beta$ | .913 / .819 | **.927 / .843** |
| 缺失 RGB (VT5000) | $E_m$ / $F_\beta$ | .869 / .742 | **.887 / .774** |
| 缺失 Thermal (VT5000) | $E_m$ / $F_\beta$ | .895 / .791 | **.913 / .822** |
| 平均性能下降 (VT5000) | $E_m$ / $F_\beta$ | -.031 / -.052 | **-.027 / -.045** |

CoLA 在所有数据集的完整模态和缺失模态条件下均取得最佳或次佳性能，平均性能下降最小。

### 消融实验（VT5000）

| 配置 | 完整 $S_\alpha$ | 缺失RGB $S_\alpha$ | 缺失T $S_\alpha$ | 平均 $S_\alpha$ |
|------|----------------|-------------------|-----------------|----------------|
| Baseline | .859 | .820 | .845 | .841 |
| +LQA | .887 | .828 | .849 | .855 |
| +CD | .880 | .833 | .868 | .860 |
| +LQA+CD | **.892** | **.840** | **.874** | **.869** |

LQA 主要提升完整模态性能（+.028），CD 主要提升模态缺失性能（+.023），二者互补。

### 质量评估方法对比

LQA 与 BRISQUE、GIE、CLIP-IQA、CLIP-IQA+ 对比，在 VT821 上 $S_\alpha$ 达到 .888（次佳 .878），证明可学习 prompt 微调策略的有效性。

## 亮点与洞察

1. **首个同时处理噪声和缺失的双模态 SOD 模型**，填补了该领域鲁棒性研究的空白
2. **LQA 设计精巧**：仅用少量可学习参数微调 CLIP，无需质量标注数据，既保留泛化性又适配目标域
3. **CD 的零卷积策略**借鉴 ControlNet 思想，解耦完整模态能力与缺失鲁棒性，两者互不干扰
4. **即插即用特性**：CD 作为训练方案可应用于各种双模态 SOD 模型，具有良好的通用性
5. 编码器和解码器设计故意保持简单，清晰展示了核心模块的贡献

## 局限与展望

1. 两阶段训练增加了整体训练时间，能否合并为端到端单阶段训练值得探索
2. 仅考虑全模态缺失（全零输入），未处理部分区域损坏或渐进退化的情况
3. LQA 依赖 CLIP 预训练权重，对非自然图像（如医学影像）的适用性待验证
4. 质量评估仅产生一个全局标量，缺乏空间维度的细粒度质量感知
5. CD 需要额外复制编码器，推理时参数量翻倍，对资源受限场景不友好

## 相关工作与启发

- **vs 直接 Modality Dropout**：直接 dropout 提升缺失鲁棒性但损害完整模态性能，CD 通过冻结+副本的解耦设计同时保证两者
- **vs GIE / BRISQUE 质量评估**：GIE 使用固定预训练网络无法适配目标数据，BRISQUE 基于手工特征不可训练；LQA 通过 prompt learning 兼具泛化和适应性
- **vs TAGFNet**：此前最强的鲁棒性方法，CoLA 在平均性能下降方面进一步降低约 30-40%
- **vs ControlNet**：将 ControlNet 的零卷积思想从生成模型迁移到判别式的 SOD 任务，处理模态缺失而非条件生成

## 相关工作与启发

1. 零卷积解耦策略具有广泛迁移性，可用于任何需要在保持原有能力的同时适应新条件的多模态任务
2. 用 CLIP 做模态质量评估的思路可推广到其他多模态融合任务（如 RGB-D 语义分割、多模态目标跟踪）
3. 将模态缺失建模为条件输入的视角，与 missing data imputation 和 robust multi-task learning 有潜在联系
4. LQA 的 prompt learning 方式可进一步扩展为多粒度质量描述，如 "blurry"、"dark"、"saturated" 等

## 评分
- 新颖性: 4/5 — 首次在双模态 SOD 同时解决噪声和缺失问题，LQA 和 CD 的设计动机清晰且方案新颖
- 实验充分度: 4/5 — 覆盖 RGB-T 和 RGB-D 两大场景，消融实验和对比分析完整，缺少效率分析
- 写作质量: 4/5 — 问题定义清晰，方法描述系统，图示信息量大
- 价值: 4/5 — CD 的即插即用特性有实际应用价值，但两阶段训练和推理开销限制了落地

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales](self-supervised_co-salient_object_detection_via_feature_correspondences_at_multi.md)
- [\[NeurIPS 2025\] Robust Egocentric Referring Video Object Segmentation via Dual-Modal Causal Intervention](../../NeurIPS2025/segmentation/robust_egocentric_referring_video_object_segmentation_via_dual-modal_causal_inte.md)
- [\[CVPR 2025\] Visual Consensus Prompting for Co-Salient Object Detection](../../CVPR2025/segmentation/visual_consensus_prompting_for_co-salient_object_detection.md)
- [\[ECCV 2024\] Frequency-Spatial Entanglement Learning for Camouflaged Object Detection](frequency-spatial_entanglement_learning_for_camouflaged_object_detection.md)
- [\[ECCV 2024\] Learning Camouflaged Object Detection from Noisy Pseudo Label](learning_camouflaged_object_detection_from_noisy_pseudo_label.md)

</div>

<!-- RELATED:END -->
