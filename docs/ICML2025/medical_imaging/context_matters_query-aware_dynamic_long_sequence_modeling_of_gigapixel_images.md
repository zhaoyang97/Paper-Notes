---
title: >-
  [论文解读] Context Matters: Query-aware Dynamic Long Sequence Modeling of Gigapixel Images
description: >-
  [ICML 2025][医学图像][全切片图像] 提出 Querent 框架——通过 query-aware 的动态区域重要性评估实现千亿像素全切片图像（WSI）中的高效长程上下文建模，在理论上有界逼近完整自注意力，在 10+ 个 WSI 数据集的生物标志物预测/基因突变预测/癌症分型/生存分析中超越 SOTA。
tags:
  - "ICML 2025"
  - "医学图像"
  - "全切片图像"
  - "多实例学习"
  - "动态注意力"
  - "计算病理学"
  - "长序列建模"
---

# Context Matters: Query-aware Dynamic Long Sequence Modeling of Gigapixel Images

**会议**: ICML 2025  
**arXiv**: [2501.18984](https://arxiv.org/abs/2501.18984)  
**代码**: 有（GitHub）  
**领域**: 医学图像  
**关键词**: 全切片图像, 多实例学习, 动态注意力, 计算病理学, 长序列建模

## 一句话总结
提出 Querent 框架——通过 query-aware 的动态区域重要性评估实现千亿像素全切片图像（WSI）中的高效长程上下文建模，在理论上有界逼近完整自注意力，在 10+ 个 WSI 数据集的生物标志物预测/基因突变预测/癌症分型/生存分析中超越 SOTA。

## 研究背景与动机

**领域现状**：计算病理学中的全切片图像（WSI）包含 $10000^2 \sim 100000^2$ 像素，需要从数千到数万个 patch 中识别分散的诊断特征——"大海捞针"。多实例学习（MIL）成为主流弱监督框架。

**现有痛点**：
   - Transformer 的完整自注意力有 $O(n^2)$ 复杂度，对万级 patch 不可行
   - 线性注意力（如 TransMIL/Nyströmformer）降低复杂度但损失了建模能力——线性近似创造信息瓶颈
   - 局部-全局注意力（如 HIPT、LongMIL）使用固定窗口，无法适应"哪些区域与当前 patch 相关"的高度变化性

**核心矛盾**：一个关键观察——WSI 中 patch 间的相关性高度依赖上下文。肿瘤边界区域与远处的类似浸润模式高度相关，但与相邻的正常组织不相关。固定的注意力模式无法捕捉这种上下文依赖的异质性关系。

**本文目标**：在保持完整注意力建模能力的同时实现计算效率。

**切入角度**：每个 query patch 动态决定"哪些远处区域与我相关"——通过高效的区域级元数据估计重要性，只对高重要性区域做完整注意力。

**核心 idea**：区域级元数据（min/max 特征压缩）→ 重要性评分 → 选择 top-K 区域 → 稀疏但精准的注意力。

## 方法详解

### 整体框架
Querent 分 4 步处理 WSI：
1. **区域划分+元数据汇总**：将 WSI 的 patch 分成区域（每区域 K 个 patch），用 min-max 网络为每个区域计算紧凑的元数据表示
2. **区域重要性评估**：给定一个 query patch，用元数据高效评估所有区域的重要性分数
3. **选择性自注意力**：仅在 query patch 和 top-K 最相关区域的 patch 之间计算完整自注意力
4. **注意力池化**：聚合特征进行幻灯片级预测

### 关键设计

1. **区域级 Min-Max 元数据**:

    - 功能：将每个区域的 K 个 patch 特征压缩为两个向量（min 和 max）
    - 核心思路：
        - 对区域 $R_i$ 中所有 patch $\{x_{i1}, ..., x_{iK}\}$，计算逐元素最小值 $m_i^{\min}$ 和最大值 $m_i^{\max}$
        - 通过可学习的投影 $f_{\min}, f_{\max}$ 映射到共享嵌入空间
    - 设计动机：min-max 范围隐式编码了区域中 patch 特征的"可达范围"——如果一个 query 的投影落在某区域的 min-max 范围内，说明该区域中可能存在与 query 高度相关的 patch
    - 复杂度：$O(M)$（$M$ 为区域数），远小于 $O(N)$（$N$ 为 patch 数）

2. **Query-Aware 重要性评分**:

    - 功能：为每个 query patch 动态评估所有区域的相关性
    - 核心思路：$s_i = \max(|\langle \hat{q}, \hat{m}_i^{\min} \rangle|, |\langle \hat{q}, \hat{m}_i^{\max} \rangle|)$
    - 选择得分最高的 top-K 区域进行完整注意力
    - 设计动机：每个 query 有独特的注意力模式——肿瘤 patch 关注远处的肿瘤区域，正常 patch 关注局部上下文
    - 理论保证：证明 Querent 的注意力输出与完整自注意力的差异有常数上界（Theorem 1）

3. **选择性完整注意力**:

    - 功能：仅在 query 和选定区域之间计算标准自注意力
    - 核心思路：对每个 query，只使用选定区域的 K/V 参与注意力计算
    - 复杂度：$O(N \cdot K_{\text{sel}} \cdot K)$，其中 $K_{\text{sel}}$ 是选定区域数（远小于总区域数 $M$）
    - 设计动机：在计算精确注意力的区域内保持完整建模能力，只在"选哪些区域"层面做近似

### 损失函数 / 训练策略
- 分类任务：交叉熵损失
- 生存分析：Cox 比例风险损失
- 端到端训练（包括元数据投影和重要性评分网络）
- patch 特征由预训练 CPath 基础模型（PLIP）提取

## 实验关键数据

### 主实验
跨 10+ 个 WSI 数据集的综合评估：

| 任务 | 数据集 | Querent (AUC) | 最佳基线 (AUC) | 提升 |
|------|--------|-------------|-------------|------|
| 生物标志物预测 | TCGA-BRCA | 0.847 | 0.812 (TransMIL) | +3.5% |
| 基因突变预测 | TCGA-LUNG | 0.721 | 0.693 (ABMIL) | +2.8% |
| 癌症分型 | TCGA-NSCLC | 0.966 | 0.951 (LongMIL) | +1.5% |
| 生存分析 | TCGA-COAD | 0.672 | 0.641 (DSMIL) | +3.1% |
| 生存分析 | TCGA-UCEC | 0.718 | 0.689 (WiKG) | +2.9% |

### 效率对比

| 方法 | 内存 (GB) | 延迟 (ms) | AUC (BRCA) |
|------|----------|----------|-----------|
| 完整自注意力 | OOM | - | - |
| TransMIL (线性) | 2.1 | 45 | 0.812 |
| LongMIL (局部-全局) | 3.5 | 82 | 0.831 |
| **Querent** | **2.8** | **65** | **0.847** |

### 消融实验

| 配置 | AUC (BRCA) | 说明 |
|------|-----------|------|
| 均匀注意力（固定区域） | 0.823 | 不根据 query 选择 |
| 随机区域选择 | 0.815 | 未利用重要性信息 |
| 仅 max 元数据 | 0.838 | 缺少下界信息 |
| **Min-Max 元数据 + Query-Aware** | **0.847** | 完整方法 |
| Top-3 区域 | 0.840 | 稍少上下文 |
| **Top-5 区域** | **0.847** | 最优选择数 |
| Top-10 区域 | 0.846 | 边际收益递减 |

### 关键发现
- 在所有 10+ 个数据集和 4 种任务上一致超越 SOTA——表明方法的通用性
- Query-aware 选择比固定/随机选择提升 2-3% AUC——验证了"上下文依赖相关性"的核心假设
- 内存和延迟位于完整注意力和线性注意力之间——在效率和效果间取得最优平衡
- Top-5 区域已涵盖大部分有价值的长程依赖——说明 WSI 中的相关性本质上是稀疏的
- 理论上界在实践中确实很紧——逼近完整注意力的承诺得到验证

## 亮点与洞察
- **"上下文决定相关性"**的观察精准——同一个肿瘤 patch 可能与远处100+个 patch 外的类似区域最相关，固定窗口根本无法捕捉
- Min-Max 元数据的设计极其巧妙——用两个向量（min 和 max）就能有效估计区域与 query 的最大潜在交互
- 理论保证+实践验证的双重支持增强了方法的可信度
- 方法论具有通用性——不仅适用于 WSI，任何超长序列+稀疏相关性的场景都可借鉴
- 端到端可微——重要性评分网络也参与训练，使区域选择自适应不同任务

## 局限与展望
- 区域大小 K 是固定的，自适应区域划分可能更好
- 元数据仅用了 min/max，更丰富的统计量（如分位数、方差）可能提供更多信息量
- 仅在 2D WSI 上验证，3D 体积数据（如 CT）需要扩展
- 预训练 encoder 的选择对性能有较大影响——PLIP vs 其他 CPath 基础模型的比较可更全面
- 注意力选择的"Top-K 硬截断"可能丢失边界重要的区域——软选择机制值得探索

## 相关工作与启发
- **vs TransMIL**: 线性注意力，牺牲建模能力换效率；Querent 保持完整注意力但只用在重要区域
- **vs HIPT/LongMIL**: 固定局部-全局窗口，不适应上下文变化；Querent 动态适应每个 query
- **vs MambaMIL**: SSM 序列建模保持线性复杂度但顺序依赖强；Querent 支持任意位置的跳跃关注
- **vs WiKG**: 图结构建模 patch 关系但图构建依赖预定义邻接；Querent 动态构建 query-specific 邻接
- **启发**：区域级元数据+动态选择的框架可推广到文档理解（跨段落关注）、视频分析（跨帧关注）等长序列场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Query-aware 动态稀疏注意力是 WSI 分析的重要突破
- 实验充分度: ⭐⭐⭐⭐⭐ 10+ 数据集、4 种任务、理论+实验双验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、图示精美、分析透彻
- 价值: ⭐⭐⭐⭐⭐ 对计算病理学和长序列建模都有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](../../NeurIPS2025/medical_imaging/few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)
- [\[AAAI 2026\] Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](../../AAAI2026/medical_imaging/towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)
- [\[CVPR 2025\] Reanimating Images using Neural Representations of Dynamic Stimuli](../../CVPR2025/medical_imaging/reanimating_images_using_neural_representations_of_dynamic_stimuli.md)
- [\[NeurIPS 2025\] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](../../NeurIPS2025/medical_imaging/dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)
- [\[CVPR 2025\] SALIENT: Frequency-Aware Paired Diffusion for Controllable Long-Tail CT Detection](../../CVPR2025/medical_imaging/salient_frequency-aware_paired_diffusion_for_controllable_long-tail_ct_detection.md)

</div>

<!-- RELATED:END -->
