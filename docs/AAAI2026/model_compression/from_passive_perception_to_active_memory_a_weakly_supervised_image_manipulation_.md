---
title: >-
  [论文解读] From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations
description: >-
  [AAAI 2026][模型压缩][图像篡改定位] 提出 BoxPromptIML，一种基于粗粒度框标注的弱监督图像篡改定位（IML）框架，通过冻结的 SAM 教师模型将粗糙边界框转化为高质量伪掩码，结合记忆引导门控融合模块（MGFM）训练轻量级学生模型，仅需 7 秒/张的标注成本即可媲美甚至超越全监督方法。
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "图像篡改定位"
  - "弱监督"
  - "知识蒸馏"
  - "记忆引导"
  - "SAM"
---

# From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations

**会议**: AAAI 2026  
**arXiv**: [2511.20359](https://arxiv.org/abs/2511.20359)  
**代码**: [https://github.com/vpsg-research/BoxPromtIML](https://github.com/vpsg-research/BoxPromtIML)  
**领域**: 机器人  
**关键词**: 图像篡改定位, 弱监督, 知识蒸馏, 记忆引导, SAM

## 一句话总结
提出 BoxPromptIML，一种基于粗粒度框标注的弱监督图像篡改定位（IML）框架，通过冻结的 SAM 教师模型将粗糙边界框转化为高质量伪掩码，结合记忆引导门控融合模块（MGFM）训练轻量级学生模型，仅需 7 秒/张的标注成本即可媲美甚至超越全监督方法。

## 研究背景与动机

图像篡改定位（IML）领域面临一个根本的**标注成本 vs 定位精度**的权衡：

- **全监督方法**（TruFor、PIM、Mesorch 等）：依赖像素级掩码标注，标注一张图平均需要 **23 分钟**，限制了数据集规模和实际部署
- **弱监督方法**（WSCL 等）：仅使用图像级标签（4 秒/张），但缺乏空间定位能力，F1 仅 0.239
- **介于两者之间的空白**：需要一种既保留空间定位信息又大幅降低标注成本的方案

作者通过严格的用户研究（10 位志愿者标注 100 张图）量化了不同标注方式的时间成本，发现**粗糙边界框标注仅需 7 秒/张**，比全监督节省 98% 以上成本，同时保留了关键的空间信息。

**核心 idea**：利用 SAM 的零样本分割能力将低成本的框标注"升级"为高质量伪掩码，然后通过知识蒸馏训练一个无需任何提示即可独立推理的轻量级学生模型，并用记忆引导机制进一步增强定位能力。

## 方法详解

### 整体框架

BoxPromptIML 采用教师-学生知识蒸馏范式：
1. **教师模型**：冻结的 SAM，接收输入图像 + 粗糙边界框，生成高质量伪掩码
2. **学生模型**：轻量级网络（TinyViT 主干），仅接收原始图像，学习预测篡改区域
3. **核心模块**：记忆引导门控融合模块（MGFM）增强学生模型的特征融合能力

### 关键设计

1. **教师模型：基于提示的高保真掩码生成**:

    - 功能：将粗糙的边界框标注转化为精细的像素级伪掩码
    - 核心思路：利用预训练 SAM 的零样本分割能力，给定图像 I 和边界框 B，生成二值掩码 M_teacher
    - 设计动机：SAM 具有强大的通用分割能力，可以从最少的空间线索中恢复精细边界；冻结教师避免了微调开销

2. **记忆引导门控融合模块（MGFM）**:

    - 功能：增强学生模型的多尺度特征融合，结合实时观察和历史记忆原型
    - 核心思路：受人类"集体潜意识"启发，设计了双引导机制：
      * **门控整合（GI）**：为 TinyViT 提取的 4 个尺度特征各生成门控图 G_i = σ(Conv(F'_i))，通过加权聚合得到融合特征 F_fused
      * **记忆库引导**：维护一个可学习的记忆库，存储原型篡改模式 A_mem
      * **双引导细化**：A_final = α(A'_base ⊙ G_avg) + (1-α)·A_mem
    - 设计动机：单张图片的分析可能遗漏细微篡改痕迹，记忆库提供了跨样本的"历史经验"作为先验；不同于标准注意力机制，记忆库解耦了知识聚合与网络权重，提供更稳定的全局先验

3. **粗粒度标注策略**:

    - 功能：设计一种介于像素级掩码和图像级标签之间的标注方式
    - 核心思路：标注者仅需在篡改区域周围画一个粗糙的边界框
    - 设计动机：7 秒 vs 23 分钟的标注时间差异，且不丧失空间定位信息

### 损失函数 / 训练策略

采用标准二元交叉熵损失实现伪监督蒸馏：
$$\mathcal{L}_{loss} = BCE(A_{refined}, \hat{M}_{teacher})$$

- 学生模型预测 A_refined ∈ [0,1]^{H×W}，教师伪掩码 M_teacher 作为软监督
- 推理时仅使用学生模型，无需任何提示或边界框
- 训练集：CASIAv2 + Coverage + Columbia + NIST16 混合

## 实验关键数据

### 主实验

**与全监督方法对比（F1, 阈值 0.5, 20 epochs）**：

| 方法 | 监督 | IND Avg. | OOD Avg. |
|------|------|----------|----------|
| PSCC-Net | Full | 0.535 | 0.334 |
| TruFor | Full | 0.538 | 0.236 |
| Mesorch | Full | 0.544 | 0.219 |
| SparseViT | Full | 0.562 | 0.255 |
| PIM | Full | 0.648 | 0.357 |
| **Ours** | **Weak (Box)** | **0.619** | **0.285** |

**与弱监督方法对比**：

| 方法 | NC16 | C1 | Col | Cov | Avg. |
|------|------|-----|------|------|------|
| WSCL | 0.111 | 0.140 | 0.524 | 0.180 | 0.239 |
| SCAF (scribble) | 0.226 | 0.530 | 0.442 | 0.400 | 0.400 |
| **Ours (box)** | **0.618** | **0.552** | **0.903** | **0.403** | **0.619** |

### 消融实验

| 配置 | 参数量(M) | FLOPs(G) | 关键指标 |
|------|----------|----------|---------|
| Mesorch | 85.8 | 124.9 | IND=0.544 (全监督) |
| PIM | 152.5 | 682.9 | IND=0.648 (全监督) |
| **Ours** | **5.5** | **1.4** | IND=0.619 (弱监督) |
| SCAF | 27.57 | 35.39 | IND=0.400 (弱监督) |

**社交网络鲁棒性测试**：

| 方法 | 无压缩 | Facebook | WeiBo | WeChat | WhatsApp |
|------|--------|----------|-------|--------|----------|
| WSCL | 0.349 | 0.124 | 0.133 | 0.066 | 0.123 |
| Mesorch | 0.703 | 0.671 | 0.655 | 0.583 | 0.677 |
| Ours | 0.552 | 0.532 | 0.536 | 0.477 | 0.530 |

### 关键发现

- **泛化能力**：全监督方法（TruFor、PSCC-Net）随训练轮数增加，OOD 性能反而下降（过拟合）；本文方法 OOD 性能稳步提升（10ep: 0.253 → 20ep: 0.285）
- **极致效率**：5.5M 参数 + 1.4G FLOPs，是所有方法中最轻量的（PIM 的 1/28 参数，1/488 FLOPs）
- **快速收敛**：仅 20 个 epoch 即可达到竞争力的性能，无需像 SCAF 那样训练到完全收敛
- **标注成本**：7 秒/张 vs 全监督 23 分钟/张，成本降低 98%+

## 亮点与洞察

- **"从粗到精"的弱监督范式创新**：首次在 IML 中提出框标注 + SAM 伪掩码 + 知识蒸馏的流水线
- **记忆库的正则化效果**：记忆库不仅提供历史先验，还作为强正则化器防止过拟合，这解释了该方法在 OOD 场景下的持续改善
- **实用性极强**：低标注成本 + 轻量模型 + 快速收敛 + 无推理时提示需求
- **对全监督方法过拟合问题的深刻分析**：揭示了像素级掩码标注可能引入语义偏差

## 局限与展望

- 在社交网络压缩场景下性能仍低于 Mesorch 等全监督方法（如 WeChat: 0.477 vs 0.583）
- 输入分辨率较低（224×224 vs 其他方法的 512×512），可能限制了细粒度定位
- 记忆库的大小和更新策略的影响未充分探讨
- SAM 作为教师模型本身存在局限性——对极细微篡改（如颜色调整）可能无法生成准确伪掩码
- 未评估不同质量的框标注（很粗 vs 较精确）对最终性能的影响

## 相关工作与启发

- SAM 作为通用基础分割模型的"万能教师"角色值得其他领域借鉴
- MGFM 中记忆库 + 门控机制的设计可以推广到异常检测、缺陷检测等依赖先验知识的任务
- 框标注 → 伪掩码 → 蒸馏的范式可以迁移到医学图像分割、遥感变化检测等标注昂贵的领域

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Towards Unified Human Perception and Machine Understanding: Token Flow Guided Compression Framework](../../CVPR2026/model_compression/towards_unified_human_perception_and_machine_understanding_token_flow_guided_com.md)
- [\[AAAI 2026\] HCF: Hierarchical Cascade Framework for Distributed Multi-Stage Image Compression](hcf_hierarchical_cascade_framework_for_distributed_multi-stage_image_compression.md)
- [\[AAAI 2026\] InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression](infocom_kilobyte-scale_communication-efficient_collaborative_perception_with_inf.md)
- [\[ACL 2025\] Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation](../../ACL2025/model_compression/prompt_distill_teacher_student.md)
- [\[ACL 2025\] CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information](../../ACL2025/model_compression/cfsp_an_efficient_structured_pruning_framework_for_llms_with_coarse-to-fine_acti.md)

</div>

<!-- RELATED:END -->
