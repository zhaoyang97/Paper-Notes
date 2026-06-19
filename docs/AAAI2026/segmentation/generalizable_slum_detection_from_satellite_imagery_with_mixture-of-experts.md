---
title: >-
  [论文解读] Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts
description: >-
  [AAAI 2026][语义分割][贫民窟检测] 提出 GRAM（Generalized Region-Aware Mixture-of-Experts），一个两阶段测试时自适应框架：第一阶段用 MoE 架构在12个城市的百万级卫星图像上训练区域特化专家，第二阶段通过跨区域预测一致性筛选可靠伪标签进行自训练，实现对未见非洲城市的贫民窟分割泛化。
tags:
  - "AAAI 2026"
  - "语义分割"
  - "贫民窟检测"
  - "卫星图像分割"
  - "混合专家"
  - "测试时自适应"
  - "域泛化"
---

# Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts

**会议**: AAAI 2026  
**arXiv**: [2511.10300](https://arxiv.org/abs/2511.10300)  
**代码**: [GitHub](https://github.com/DS4H-GIS/GRAM)  
**领域**: 分割  
**关键词**: 贫民窟检测, 卫星图像分割, 混合专家, 测试时自适应, 域泛化

## 一句话总结

提出 GRAM（Generalized Region-Aware Mixture-of-Experts），一个两阶段测试时自适应框架：第一阶段用 MoE 架构在12个城市的百万级卫星图像上训练区域特化专家，第二阶段通过跨区域预测一致性筛选可靠伪标签进行自训练，实现对未见非洲城市的贫民窟分割泛化。

## 研究背景与动机

贫民窟（非正式定居点）的精确检测对城市规划和可持续发展至关重要。然而：

**形态异质性极高**：不同国家/城市的贫民窟在建筑风格、屋顶材料、空间组织上差异巨大（如沿海城市的灰色方形屋顶 vs 内陆城市的棕色长方形屋顶），单一模型难以跨区域泛化

**标注资源稀缺**：像素级标注成本高昂，尤其在低收入地区，传统调查方法受限于政治敏感性和后勤挑战

**现有方法迁移性差**：在一个国家训练的模型无法可靠地应用到其他区域，尤其在源域和目标域分布差异大的情况下

关键目标：在目标区域**无需任何标注数据**的情况下，实现跨区域贫民窟分割泛化。

## 方法详解

### 整体框架

GRAM 分两阶段：

**Step 1 - 源域训练**：在12个城市的多区域数据集上，使用 MoE 架构训练分割模型。MoE 层集成到 SegFormer 的 Transformer 编码器中，每个区域有专门的门控网络路由到最相关的专家。

**Step 2 - 目标自适应**：对未标注的目标图像，通过外部区域分类器确定最相似的源域区域，利用对应专家生成伪标签，并通过跨区域预测一致性过滤不可靠样本，最后在高置信度子集上自训练微调。

### 关键设计

#### MoE 架构与自适应路由

在 Transformer 编码器的 $L$ 个中间层集成轻量 MoE block $\mathcal{F}$，每个 block 包含 $E$ 个 MLP 专家适配器：

- **区域特定门控**：每个源域区域 $d$ 有独立的门控网络 $g_d$，对token特征 $z$ 计算各专家的相关性
- **噪声Top-k路由**：加高斯噪声防止专家选择过度集中，softmax归一化top-k专家权重：

$$\text{MoE}(z) = \sum_{e \in \text{top-}k(\tilde{g}_d(z))} \alpha_e \cdot \mathcal{E}_e(z)$$

- 共享backbone学习跨区域通用特征，MoE专家学习区域特化特征

#### 区域感知正则化

两个关键正则项确保专家多样性和区域特化：

1. **互信息正则** $\mathcal{L}_{MI}$：最大化区域与专家选择之间的互信息，确保不同区域激活不同专家集，防止模式坍塌：

$$I^l(d;e) = \sum_{d=1}^{D}\sum_{e=1}^{E} P^l(d,e) \log\frac{P^l(d,e)}{P^l(d)P^l(e)}$$

2. **区域分类损失** $\mathcal{L}_{dom}$：辅助区域分类器从共享backbone的中间特征预测区域标签，增强路由质量

总训练目标：$\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda_{MI} \cdot \mathcal{L}_{MI} + \lambda_{dom} \cdot \mathcal{L}_{dom}$

#### 跨区域预测一致性的伪标签筛选

目标自适应阶段的核心技术：

1. 外部区域分类器 $h_\psi$ 预测目标图像最相似的源域区域 $d_t$，通过对应路由生成伪标签 $\bar{y}_{d_t}$
2. 计算**稳定性得分**：将同一图像通过所有源域区域路由，评估不同路由预测之间的mIoU一致性：

$$s(x) = \sum_{d \neq d_t} \text{mIoU}(\bar{y}_{d_t}, \bar{y}_d)$$

3. 选择稳定性最高的 $\rho_s$ 比例样本构成可靠数据集 $\bar{\mathcal{D}}_t$，在其上自训练微调

### 损失函数 / 训练策略

- 源域训练：像素级交叉熵 + MI正则 + 区域分类损失
- 目标自适应：在筛选后的高置信度伪标签上微调，使用像素级交叉熵
- Backbone：SegFormer，SGD优化（lr=0.0001, momentum=0.99）
- 超参数：$\rho_s = 0.5$（选50%高置信度样本），$E=12$ 个专家，$k=2$ top-k路由

## 实验关键数据

### 主实验

**三个非洲测试城市的 mIoU**：

| 方法 | Dar es Salaam | Kampala | Maputo |
|------|--------------|---------|--------|
| Vanilla Source | 0.681 | 0.716 | 0.800 |
| MoE Source（无TTA） | 0.806 | 0.800 | 0.900 |
| TENT | 0.691 | 0.716 | 0.802 |
| CoTTA | 0.762 | 0.821 | 0.821 |
| BeCoTTA | 0.741 | 0.844 | 0.904 |
| **GRAM** | **0.859** | **0.870** | **0.907** |

GRAM 在所有城市全面领先，特别是在 Dar es Salaam 提升最为显著（+5.3% vs MoE Source, +11.8% vs CoTTA）。

**贫民窟类（少数类）IoU 对比（Dar es Salaam）**：

| 方法 | Slum IoU | Slum F1 |
|------|----------|---------|
| Vanilla Source | 0.476 | 0.645 |
| BeCoTTA | 0.540 | 0.702 |
| **GRAM** | **0.752** | **0.859** |

### 消融实验

**Dar es Salaam 消融**：

| 配置 | mIoU | F1 |
|------|------|-----|
| w/o $\mathcal{L}_{dom}$ | 0.836 | 0.906 |
| w/o $\mathcal{L}_{MI}$ | 0.734 | 0.823 |
| No Filtering | 0.818 | 0.893 |
| Confidence Filtering | 0.463 | 0.501 |
| Temporal Consistency | 0.837 | 0.907 |
| **Full GRAM** | **0.859** | **0.921** |

关键结论：$\mathcal{L}_{MI}$ 是最重要的组件（去除后mIoU大降12.5%）。置信度过滤在域偏移下完全失败（mIoU仅0.463），而跨区域一致性过滤远更可靠。

### 关键发现

- MoE Source vs Vanilla Source：仅加MoE就大幅提升泛化（+12.5% mIoU in Dar es Salaam），证明区域特化的价值
- 区域分类器的预测与地理图像相似度高度一致：沿海城市（Cape Town↔Dar es Salaam↔Maputo）和内陆城市（Nairobi↔Kampala）自然聚类
- 时序追踪应用：Kampala贫民窟率从2015年8.4%增至2023年8.6%；Maputo从35.3%增至41.2%；Dar es Salaam从17.3%降至12.6%

## 亮点与洞察

1. **百万级数据集贡献**：构建了12个城市、270万+图像块的大规模贫民窟分割数据集，是该领域规模最大的基准
2. **MoE+TTA的巧妙结合**：MoE天然适合处理多区域异质性，跨专家一致性天然提供了伪标签质量的度量
3. **置信度过滤的失败**：清晰展示域偏移下entropy-based方法的脆弱性，为TTA社区提供了重要教训
4. **社会价值**：支持对缺乏官方统计数据的地区进行贫民窟监测，对城市政策制定有直接实用价值

## 局限与展望

- 目标域仅测试了3个非洲城市，缺少亚洲/南美未见城市的评估
- 区域分类器 $h_\psi$ 的质量直接影响路由准确性，但分类器本身的鲁棒性未深入分析
- $\rho_s = 0.5$ 的筛选比例对所有城市统一设置，可考虑自适应调整
- 稳定性得分仅基于mIoU，可探索更细粒度的像素级不确定性估计
- 数据集标注部分依赖半监督伪标签，可能引入标注噪声

## 相关工作与启发

- **MoE在分割中的应用**：本文是MoE+TTA在遥感分割中的首次成功应用，证明了区域特化专家在跨域泛化中的价值
- **与BeCoTTA的关系**：BeCoTTA同样使用MoE适配器，但缺乏有效的伪标签筛选；GRAM的跨区域一致性机制是关键差异
- **半监督+全监督两阶段**：先用ST++生成伪标签再训练全监督模型的策略，为大规模标注稀缺场景提供了实用范式
- 启发：跨区域MoE+一致性筛选的思路可迁移到其他遥感任务（如建筑检测、土地利用分类）

## 评分

- 新颖性: ⭐⭐⭐⭐ — MoE+跨专家一致性TTA的组合新颖，稳定性得分设计巧妙
- 实验充分度: ⭐⭐⭐⭐ — 大规模数据集,多基线对比,消融充分,时序追踪展示
- 写作质量: ⭐⭐⭐⭐ — 方法动机清晰，地理可视化直观
- 价值: ⭐⭐⭐⭐⭐ — 开源数据集+代码，实际社会影响力大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] PromptMoE: A Segmentation Refinement Framework Leveraging Mixture of Experts for Improved Prompting](../../CVPR2026/segmentation/promptmoe_a_segmentation_refinement_framework_leveraging_mixture_of_experts_for_.md)
- [\[CVPR 2025\] Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](../../CVPR2025/segmentation/spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)
- [\[ICCV 2025\] Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling](../../ICCV2025/segmentation/harnessing_massive_satellite_imagery_with_efficient_masked_image_modeling.md)
- [\[CVPR 2026\] Generalizable Co-Salient Object Detection via Mixed Content-Style Modulation](../../CVPR2026/segmentation/generalizable_co-salient_object_detection_via_mixed_content-style_modulation.md)
- [\[CVPR 2025\] Towards Generalizable Scene Change Detection](../../CVPR2025/segmentation/towards_generalizable_scene_change_detection.md)

</div>

<!-- RELATED:END -->
