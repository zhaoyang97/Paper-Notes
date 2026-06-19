---
title: >-
  [论文解读] One-Shot Knowledge Transfer for Scalable Person Re-Identification
description: >-
  [ICCV 2025][人体理解][知识迁移] 提出 OSKT（One-Shot Knowledge Transfer），通过将教师模型知识精炼为"权重链"（weight chain）作为中间载体，实现一次计算即可生成任意尺寸学生模型的行人重识别模型压缩方案。 - 行人重识别（ReID）模型部署到边缘设备时需要适配不同资源约…
tags:
  - "ICCV 2025"
  - "人体理解"
  - "知识迁移"
  - "权重链"
  - "行人重识别"
  - "模型压缩"
  - "一次性计算"
---

# One-Shot Knowledge Transfer for Scalable Person Re-Identification

**会议**: ICCV 2025  
**arXiv**: [2511.06016](https://arxiv.org/abs/2511.06016)  
**代码**: 无  
**领域**: 行人重识别 / 模型压缩  
**关键词**: 知识迁移, 权重链, 行人重识别, 模型压缩, 一次性计算

## 一句话总结

提出 OSKT（One-Shot Knowledge Transfer），通过将教师模型知识精炼为"权重链"（weight chain）作为中间载体，实现一次计算即可生成任意尺寸学生模型的行人重识别模型压缩方案。

## 研究背景与动机

- 行人重识别（ReID）模型部署到边缘设备时需要适配不同资源约束的紧凑模型
- 传统压缩方法（蒸馏、剪枝）每种尺寸都需要独立的训练流程，计算成本随目标模型数量线性增长
- 已有的权重选择和学习基因等方法要么不够灵活无法生成密集尺寸，要么效率不足需要大量计算
- 核心问题：**能否仅进行一次计算即可获得适应各种资源场景的不同尺寸学生模型？**

## 方法详解

### 整体框架

OSKT 分为两个阶段：(1) 从教师模型精炼权重链；(2) 通过权重链无需额外计算即可扩展为任意尺寸学生模型。框架适用于 CNN（ResNet）和 ViT 两种主流架构。

### 关键设计

1. **统一的 CNN/ViT 表示**: 将CNN中的卷积滤波器和ViT中的权重矩阵行统一抽象为"行"（rows），将对应的通道/列统一抽象为"列"（columns）。教师模型参数化为 $\{\mathcal{F}_l \in \mathbb{R}^{N_l \times N_{l-1} \times O_l}, 1 \leq l \leq L\}$，其中$N_l$为输出特征维度，$O_l$为操作参数数（卷积层为$K \times K$，全连接层为1）。

2. **权重链设计（Weight Chain）**: 权重链保持与教师模型相同的深度但大幅缩减宽度，参数化为 $\{\mathcal{F}^C_l \in \mathbb{R}^{M_l \times N_{l-1} \times O_l}\}$，其中 $M_l \ll N_l$。关键设计包括：

    - 归一化层与教师模型共享权重，使多个 $(\gamma, \beta)$ 对可复用核心特征维度
    - 基于三个核心洞察：(a) 利用教师权重高效知识迁移；(b) 神经恒等变换——合并相同行并求和对应列可保持网络功能；(c) 权重链作为教师和学生模型的桥梁

3. **权重链精炼**:

    - **初始化**：对教师模型每层的权重行进行聚类，聚类中心初始化权重链对应行（CNN用欧氏距离，ViT用余弦距离）
    - **渐进精炼**：同时训练教师模型和最小学生模型（S-Student，宽度=权重链宽度），梯度回传至权重链
    - 精炼损失：$\mathcal{L}_{ref} = \frac{1}{L}\sum_l \frac{1}{M_l}\sum_j \sum_{k \in \mathcal{I}_l^{(j)}}(\mathcal{F}_{l,k} - \mathcal{F}^C_{l,j})^2$
    - 总损失：$\mathcal{L} = \mathcal{L}_T + \mathcal{L}_S + \alpha \mathcal{L}_{ref}$，其中教师和学生各使用 ID loss + Triplet loss
    - ViT 中 $\alpha$ 为渐进系数 $\alpha = \frac{iter}{n\_iter}$，CNN 中固定为 1

4. **学生模型生成（O(1) 操作）**: 若第 $l$ 层需要 $C_l$ 行（$M_l \leq C_l \leq N_l$），按教师权重行数量比例堆叠权重链的精炼行至 $C_l$ 行，相应合并下一层的列权重（求和），归一化层的 $(\gamma, \beta)$ 对取平均。**无需任何额外计算。**

### 损失函数 / 训练策略

- 教师模型：$\mathcal{L}_T = \mathcal{L}_{id}(\boldsymbol{p}^T) + \mathcal{L}_{tri}(\boldsymbol{f}^T)$（ID loss + hard triplet loss）
- S-Student：$\mathcal{L}_S = \mathcal{L}_{id}(\boldsymbol{p}^S) + \mathcal{L}_{tri}(\boldsymbol{f}^S)$（梯度回传至权重链）
- 精炼损失：$\mathcal{L}_{ref}$（MSE 拉紧聚类中心）
- 有残差连接的层进行联合聚类以对齐特征维度

## 实验关键数据

### 主实验 (表格)

**ResNet50 → 学生模型（Market1501，单场景）：**

| 方法 | Res-50-S1 mAP | Res-50-S1 R1 | Res-50-S3 mAP | Res-50-S3 R1 | Res-50-S5 mAP | Res-50-S5 R1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Scratch | 30.6 | 53.3 | 47.4 | 70.9 | 60.7 | 80.9 |
| WTSel | 48.5 | 72.3 | 63.0 | 81.5 | 84.1 | 93.0 |
| KD++ | 41.5 | 65.7 | 59.9 | 80.4 | 71.4 | 86.9 |
| DepGraph | 61.3 | 80.7 | 83.2 | 92.7 | 86.3 | 94.3 |
| **OSKT** | **75.7** | **89.4** | **84.7** | **93.3** | **87.6** | **94.5** |

**ViT-B/ViT-S → 学生模型（Market1501，单场景）：**

| 方法 | ViT-S-S1 | ViT-S-S2 | ViT-B-S1 | ViT-B-S2 |
|:---:|:---:|:---:|:---:|:---:|
| Scratch | 13.9/23.9 | 18.3/31.2 | 22.2/37.4 | 21.9/36.3 |
| DepGraph | 56.5/74.1 | 69.0/84.2 | 15.3/30.1 | 81.5/91.7 |
| **OSKT** | **74.2/87.1** | **77.2/89.0** | **81.6/91.9** | **82.9/92.4** |

### 消融实验 (表格)

**关键组件消融（Market1501 + MSMT17→CUHK03）：**

| 设置 | 单场景 Res-S1 | 单场景 Res-S2 | 跨场景 Res-S1 | 跨场景 Res-S2 |
|:---:|:---:|:---:|:---:|:---:|
| Scratch | 30.6/53.3 | 41.5/65.1 | 6.1/4.8 | 9.1/6.7 |
| (a) 随机教师 | 54.7/76.5 | 58.2/78.9 | 28.2/28.2 | 30.8/30.9 |
| (b) 随机聚类 | 55.1/76.2 | 59.9/80.2 | 31.4/33.8 | 34.9/35.4 |
| (e) 无精炼 | 52.2/73.3 | 59.8/79.3 | 21.6/20.9 | 29.4/29.0 |
| (f) 无 $\mathcal{L}_T$ | 62.7/81.2 | 64.0/81.5 | 34.2/35.1 | 35.4/37.3 |
| **OSKT** | **75.7/89.4** | **77.6/90.6** | **45.7/47.3** | **49.2/52.2** |

### 关键发现

- OSKT 在最小学生模型上优势最大：Res-50-S1 上 mAP 比 DepGraph 高 **14.4 个点**（75.7 vs 61.3）
- 跨场景迁移效果显著：MSMT17→CUHK03 上，Res-50-S1 的 mAP 比 DepGraph 高 14.4 个点
- 权重精炼至关重要：去掉精炼步骤（设置e），性能大幅下降（75.7→52.2）
- 与轻量级 ReID 架构（OSNet、MSINet）兼容良好，均有稳定提升
- 少样本设置（10%/30%/50% ID）下仍保持优势

## 亮点与洞察

- "权重链"作为知识载体的概念新颖——本质是一种基于聚类的权重共享方案，通过"提起串珠两端"隐式训练所有中间模型
- 学生模型生成是 O(1) 操作，真正实现"一次计算，任意部署"
- 利用神经恒等变换（合并相同行+求和列）的数学性质保证功能等价性
- 框架对 CNN 和 ViT 统一适用，实际工程意义大

## 局限与展望

- 权重链的宽度（最小学生模型尺寸）需要预先设定，影响生成模型的下界
- 聚类距离度量对架构敏感（CNN 用欧氏距离、ViT 用余弦距离），交换后性能下降
- 当前仅在 ReID 任务上验证，泛化到其他视觉任务（检测、分割）的效果未知
- 归一化参数平均合并是一种近似，可能在极端压缩比下引入误差

## 相关工作与启发

- 与 Net2Net、Weight Selection、Learngene 等权重复用方法对比，OSKT 的密集尺寸生成能力和知识迁移效率更优
- 权重链的聚类精炼思路可启发其他"模型权重作为知识载体"的研究
- 对边缘计算场景下的快速模型定制具有重要意义

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 权重链概念原创性强，一次计算生成任意尺寸模型的思路具有突破性
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖 CNN/ViT、单场景/跨场景、少样本设置、轻量级架构兼容性等多维度验证
- **写作质量**: ⭐⭐⭐⭐ 三个核心洞察阐述清晰，但公式较多，阅读门槛略高
- **价值**: ⭐⭐⭐⭐⭐ 解决了边缘部署中的实际痛点，框架通用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Controllable and Expressive One-Shot Video Head Swapping](controllable_and_expressive_one-shot_video_head_swapping.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[NeurIPS 2025\] K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning](../../NeurIPS2025/human_understanding/k-decore_facilitating_knowledge_transfer_in_continual_structured_knowledge_reaso.md)
- [\[ICCV 2025\] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)
- [\[CVPR 2026\] Dynamic Magic: Unleashing Restricted Knowledge for Lifelong Person Re-Identification](../../CVPR2026/human_understanding/dynamic_magic_unleashing_restricted_knowledge_for_lifelong_person_re-identificat.md)

</div>

<!-- RELATED:END -->
