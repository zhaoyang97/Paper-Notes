---
title: >-
  [论文解读] AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP
description: >-
  [CVPR 2025][目标检测][零样本异常检测] 提出 AA-CLIP，通过两阶段训练策略（先适配文本编码器建立异常感知锚点，再对齐 patch 级视觉特征），在保留 CLIP 泛化能力的前提下增强其异常判别力，仅需极少训练样本即可在工业和医学多个数据集上达到 SOTA 零样本异常检测性能。 异常检测（AD）旨在建模正常…
tags:
  - "CVPR 2025"
  - "目标检测"
  - "零样本异常检测"
  - "CLIP适配"
  - "文本锚点解耦"
  - "残差适配器"
  - "工业/医学异常"
---

# AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP

**会议**: CVPR 2025  
**arXiv**: [2503.06661](https://arxiv.org/abs/2503.06661)  
**代码**: [https://github.com/Mwxinnn/AA-CLIP](https://github.com/Mwxinnn/AA-CLIP)  
**领域**: 异常检测 / 医学影像  
**关键词**: 零样本异常检测, CLIP适配, 文本锚点解耦, 残差适配器, 工业/医学异常

## 一句话总结

提出 AA-CLIP，通过两阶段训练策略（先适配文本编码器建立异常感知锚点，再对齐 patch 级视觉特征），在保留 CLIP 泛化能力的前提下增强其异常判别力，仅需极少训练样本即可在工业和医学多个数据集上达到 SOTA 零样本异常检测性能。

## 研究背景与动机

异常检测（AD）旨在建模正常分布以识别离群样本，广泛应用于工业缺陷检测和医学病变检测。传统 AD 方法依赖充足标注数据，泛化能力有限。CLIP 凭借大规模图文对比预训练展现了强大的零样本迁移能力，成为 few/zero-shot AD 的热门方案。

然而，现有 CLIP-based AD 方法面临一个核心问题：**CLIP 天生对异常"不感知"（Anomaly Unawareness）**。原因在于 CLIP 在通用数据上训练，缺乏对缺陷/异常的细粒度语义理解。具体表现为：
1. 正常和异常文本嵌入在特征空间高度交织，t-SNE 可视化显示二者几乎无法分离
2. 即使图像存在明显缺陷，其视觉特征与"正常"描述的相似度仍高于"异常"描述
3. 直接使用 CLIP 原始文本嵌入作为 AD 锚点效果不佳

现有解决方案要么只适配视觉特征、忽略文本空间的异常不感知问题（如 VAND、MVFA-AD），要么使用 prompt learning 改造文本编码器但破坏了类别信息（如 AnomalyCLIP、AdaCLIP）。

**核心 idea**：首先在文本空间建立异常感知的"锚点"，将正常与异常语义清晰分离，然后引导视觉特征对齐这些锚点实现精确定位——两阶段顺序适配，用残差适配器保护 CLIP 原始知识。

## 方法详解

### 整体框架

AA-CLIP 采用两阶段训练流程，CLIP 原始参数始终冻结：
- **输入**：图像 + 正常/异常文本 prompt
- **第一阶段**：冻结视觉编码器，适配文本编码器 → 输出异常感知的文本锚点 $T_N$, $T_A$
- **第二阶段**：冻结文本编码器，适配视觉编码器 → 输出与锚点对齐的 patch 级视觉特征
- **推理**：比较视觉特征与锚点的余弦相似度 → 像素级异常分割图 + 图像级异常分数

### 关键设计

1. **残差适配器（Residual Adapter）**:

    - 功能：在编码器浅层注入可训练模块，适配特征的同时保留原始知识
    - 核心思路：将第 $i$ 层 transformer 输出 $x^i$ 通过线性变换+激活+归一化生成残差 $x^i_{residual}$，再与原特征加权融合：
    $x^i_{enhanced} = \lambda \cdot x^i_{residual} + (1-\lambda) \cdot x^i$
      其中 $\lambda=0.1$，确保原始信息占主导
    - 设计动机：直接插入普通 adapter 会严重破坏 CLIP 的泛化能力（消融实验中 pixel-AUROC 暴跌 40 点）。残差连接以较小比例混入新信息，实现"温和适配"

2. **第一阶段：文本锚点解耦**:

    - 功能：在文本编码器前 $K_T=3$ 层插入残差适配器，生成能区分正常/异常语义的文本嵌入
    - 核心思路：正常 prompt 和异常 prompt 的平均嵌入分别作为锚点 $T_N$ 和 $T_A$。计算与视觉特征的余弦相似度得到分类预测 $p_{cls}$ 和分割预测 $p_{seg}$
    - 同时引入 **Disentangle Loss** 强制正常/异常锚点正交：
    $\mathcal{L}_{dis} = |\langle T_N, T_A \rangle|^2$
    - 设计动机：确保两类锚点在特征空间充分分离，减少混淆。t-SNE 显示适配后各类别的正常/异常嵌入清晰解耦，且这种能力可泛化到未见类别

3. **第二阶段：Patch 特征对齐**:

    - 功能：在视觉编码器前 $K_I=6$ 层插入残差适配器，对齐多粒度 patch 特征与文本锚点
    - 核心思路：从视觉编码器第 6/12/18/24 层提取中间特征 $F^i$，通过可训练投影器映射后求和聚合：
    $V_{patch} = \sum_{i=1}^{4} Proj_i(F^i)$
    - 设计动机：多粒度特征融合使不同尺度的异常均可被捕获；文本编码器在此阶段已冻结，避免联合训练导致的类别信息坍塌

### 损失函数 / 训练策略

- 对齐损失：$\mathcal{L}_{align} = \mathcal{L}_{cls} + \mathcal{L}_{seg}$
    - $\mathcal{L}_{cls}$：图像级 BCE 损失
    - $\mathcal{L}_{seg}$：像素级 Dice + Focal 损失
- 总损失：$\mathcal{L}_{total} = \mathcal{L}_{align} + \gamma \mathcal{L}_{dis}$，$\gamma=0.1$
- 第一阶段 5 epochs，lr $1\times10^{-5}$；第二阶段 20 epochs，lr $5\times10^{-4}$
- 两阶段分离训练的核心原因：一阶段联合训练容易导致类别信息坍塌（消融验证），破坏零样本泛化

## 实验关键数据

### 主实验

| 数据集 | 指标 | AA-CLIP (full) | AnomalyCLIP | AdaCLIP | 提升 |
|--------|------|----------------|-------------|---------|------|
| 11数据集平均 | Pixel-AUROC | **93.4** | 91.3 | 90.4 | +2.1 |
| 7数据集平均 | Image-AUROC | **83.1** (64-shot) | 78.4 | 80.6 | +2.5 |
| Liver CT | Pixel-AUROC | **97.8** | 93.9 | 94.5 | +3.3 |
| Retina OCT | Pixel-AUROC | **95.5** | 92.6 | 88.5 | +2.9 |
| ClinicDB | Pixel-AUROC | **89.9** | 85.0 | 85.9 | +4.0 |

- 最突出发现：仅用 **2-shot**（每类1正常+1异常）训练，像素级 AUROC 即达 92.0%，超越所有使用 full-shot 训练的先前方法
- 在医学领域优势更明显：liver CT 达 97.8%、brain MRI 达 96.5%

### 消融实验

| 配置 | Pixel-AUROC | Image-AUROC | 说明 |
|------|-----------|-----------|------|
| CLIP 原始 | 50.3 | 69.3 | 基线 |
| + Linear Proj (VAND) | 88.9 | 69.3 | 仅视觉适配 |
| + 普通 Adapter | 48.9 (-40.0) | 53.4 (-15.9) | 破坏原始知识 |
| + Residual Adapter | 91.3 (+2.4) | 80.7 (+11.4) | 保护泛化能力 |
| + 文本 Residual Adapter | 92.1 (+3.2) | 82.6 (+13.3) | 文本空间适配有效 |
| + Disentangle Loss | **92.7** (+3.8) | **83.3** (+14.0) | 锚点解耦必要 |

### 关键发现

- 普通 adapter 直接插入 transformer 层会导致零样本性能灾难性下降，验证了残差设计的必要性
- 文本空间适配比视觉空间适配更关键——文本锚点解耦为视觉对齐提供了更精确的语义基础
- 一阶段联合训练（如 AdaCLIP 策略）会导致类别信息坍塌，两阶段策略是保护泛化能力的关键

## 亮点与洞察

- **问题定义精准**：首次系统分析 CLIP 的"异常不感知"限制，用 t-SNE/热力图/反例清晰展示问题
- **方法极简高效**：核心就是残差适配器 + 两阶段训练，无复杂架构，单张 RTX 3090 即可复现
- **数据效率极高**：2-shot 即可超越先前 full-shot 方法，对数据稀缺的医学场景意义重大
- **跨域泛化**：在 VisA 上训练，直接迁移到 MVTec-AD、脑 MRI、肝 CT、视网膜 OCT 等完全不同的数据集

## 局限与展望

- full-shot 训练出现过拟合迹象，说明 CLIP 适配存在饱和点，需研究更好的正则化策略
- 只验证了工业和医学两个领域，对自然图像、遥感等领域的泛化性未知
- prompt 设计较为固定，可尝试自动化 prompt 搜索进一步提升

## 相关工作与启发

- CLIP 的 fine-grained 语义感知是其"短板"也是研究热点，AA-CLIP 的思路（先修复文本空间、再引导视觉空间）可迁移至其他需要细粒度理解的任务
- 残差适配器的设计思想与 LoRA 相似但更简单，在保持预训练知识和注入新能力之间取得好平衡
- 两阶段策略的启示：分离适配不同模态可避免互相干扰

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题定义新颖（Anomaly Unawareness），方法设计合理但组件本身不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 11个数据集、多种shot设置、详细消融，industry+medical双域验证
- 写作质量: ⭐⭐⭐⭐⭐ 可视化丰富（t-SNE、热力图、定性结果），逻辑清晰
- 价值: ⭐⭐⭐⭐ 对零样本AD领域有实际价值，特别是医学场景下的低数据需求

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DLVP-CLIP: Enhancing Fine-Grained Zero-Shot Anomaly Detection via Dynamic Local Visual Prompting](../../CVPR2026/object_detection/dlvp-clip_enhancing_fine-grained_zero-shot_anomaly_detection_via_dynamic_local_v.md)
- [\[CVPR 2025\] Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models](towards_zero-shot_anomaly_detection_and_reasoning_with_multimodal_large_language.md)
- [\[CVPR 2026\] FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement](../../CVPR2026/object_detection/fb-clip_fine-grained_zero-shot_anomaly_detection_with_foreground-background_dise.md)
- [\[CVPR 2025\] T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting](t2icount_enhancing_cross-modal_understanding_for_zero-shot_counting.md)
- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](../../CVPR2026/object_detection/gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)

</div>

<!-- RELATED:END -->
