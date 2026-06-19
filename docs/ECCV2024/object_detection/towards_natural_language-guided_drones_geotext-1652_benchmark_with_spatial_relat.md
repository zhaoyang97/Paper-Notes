---
title: >-
  [论文解读] Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching
description: >-
  [ECCV 2024][目标检测][drone navigation] 构建了首个自然语言引导的无人机地理定位基准 GeoText-1652（276K bbox-text 对，316K 描述），并提出 blending spatial matching 方法通过 grounding loss + spatial relation loss 实现区域级空间关系匹配，文本检索 Recall@10 达到 31.2%。
tags:
  - "ECCV 2024"
  - "目标检测"
  - "drone navigation"
  - "geolocalization"
  - "spatial relation matching"
  - "视觉语言"
  - "benchmark"
---

# Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching

**会议**: ECCV 2024  
**arXiv**: [2311.12751](https://arxiv.org/abs/2311.12751)  
**代码**: [https://multimodalgeo.github.io/GeoText/](https://multimodalgeo.github.io/GeoText/)  
**领域**: 目标检测 / 视觉-语言 / 无人机导航  
**关键词**: drone navigation, geolocalization, spatial relation matching, vision-language, benchmark

## 一句话总结
构建了首个自然语言引导的无人机地理定位基准 GeoText-1652（276K bbox-text 对，316K 描述），并提出 blending spatial matching 方法通过 grounding loss + spatial relation loss 实现区域级空间关系匹配，文本检索 Recall@10 达到 31.2%。

## 研究背景与动机

**领域现状**：无人机导航主要依赖图像匹配（跨视角地理定位），如 University-1652 等数据集提供无人机-卫星图像对。然而实际场景中，用户更自然的输入方式是自然语言描述而非查询图像。

**现有痛点**：(1) 缺乏公开的自然语言引导无人机导航数据集——现有地理定位数据集仅有 GPS 标签而无文本描述。(2) 无人机视角的航拍场景中，建筑物外观高度相似（相邻区域可能有多栋相似建筑），仅靠视觉特征和物体描述难以区分，需要空间关系信息。

**核心矛盾**：传统跨模态匹配方法（如 CLIP、ALBEF）不建模区域间的空间关系，而航拍场景中空间位置（"左上角"、"右下方"）是区分相似目标的关键信息。

**本文目标**：建立语言引导无人机导航的基础——数据集和方法。

**切入角度**：(1) 利用 VLM + 人机交互标注流程半自动构建高质量 image-text-bbox 数据集；(2) 在跨模态匹配框架中引入 grounding + spatial relation 两个优化目标。

**核心 idea**：通过区域级空间关系描述和匹配，使自然语言能够精确引导无人机定位到目标建筑。

## 方法详解

### 整体框架
框架包含图像编码器（Swin）、文本编码器（BERT）和跨模态编码器三部分。图像经编码后通过 ROI Pooling 提取区域特征，文本分为图像级描述和区域级描述分别编码。框架同时优化四个损失：Image-Text Contrastive (ITC)、Image-Text Matching (ITM)、Grounding Loss 和 Spatial Loss。

### 关键设计

1. **GeoText-1652 数据集构建**

    - 功能：扩展 University-1652 图像数据集，添加细粒度的文本-bbox 标注
    - 核心思路：两阶段人机交互标注流程：
        - **模态扩展阶段**：使用 Visual-LLM 生成图像级和区域级描述，配合 referee 模型自动过滤（正样本关键词检查 + 负样本排除主观表述），人工仅需审核关键词列表
        - **空间精炼阶段**：用预训练 visual grounding 模型根据区域描述生成 bbox，设置空间规则过滤错误位置，5 轮人工评估迭代优化，最终 >90% 标注为优秀
    - 设计动机：纯人工标注成本过高，Visual-LLM 存在幻觉问题，需要 referee 模型 + 人工验证的混合方案
    - 数据规模：训练集 37,854 张无人机图 + 701 张卫星图 + 11,663 张地面图；每张图平均 3 个全局描述、2.62 个 bbox-text 对；全局描述平均 70.23 词

2. **Image-Text Contrastive Learning (ITC)**

    - 功能：全局图像-文本对齐
    - 核心思路：标准的 batch 内对比学习：
    $\mathcal{L}_{\text{itc}} = -\frac{1}{2}\mathbb{E}[\log(\boldsymbol{p}_{\text{t2v}}) + \log(\boldsymbol{p}_{\text{v2t}})]$
      其中 $\boldsymbol{p}_{\text{v2t}} = \frac{\exp(s(V,T)/\tau)}{\sum_i \exp(s(V,T^i)/\tau)}$
    - 设计动机：建立图像与文本描述的全局语义对齐基础

3. **Image-Text Matching (ITM)**

    - 功能：判断图像-文本对是否匹配
    - 核心思路：对每个视觉概念采样 batch 内最高相似度的难负例文本，用跨模态编码器预测匹配概率：
    $\mathcal{L}_{\text{itm}} = -\mathbb{E}[\boldsymbol{y_m}\log(\boldsymbol{p}_{\text{match}}) + (1-\boldsymbol{y_m})\log(1-\boldsymbol{p}_{\text{match}})]$
    - 设计动机：补充 ITC 的粗粒度对齐，提供细粒度的匹配判别能力

4. **Grounding Prediction Loss**

    - 功能：根据区域级文本描述预测对应 bounding box
    - 核心思路：用 6 层 Transformer + MLP 预测归一化 bbox $\hat{\boldsymbol{b}}_j = (c_x, c_y, w, h)$，训练损失：
    $\mathcal{L}_{\text{grounding}} = \mathbb{E}[\mathcal{L}_{\text{iou}}(\boldsymbol{b}_j, \hat{\boldsymbol{b}}_j) + \|\boldsymbol{b}_j - \hat{\boldsymbol{b}}_j\|_1]$
    - 设计动机：建模文本描述与图像区域的精确空间对应关系，是空间关系匹配的基础

5. **Spatial Relation Matching Loss**

    - 功能：预测多个 ROI 之间的相对空间关系
    - 核心思路：给定多个 bbox（如 $b_1, b_2, b_3$），通过 ROI Pooling 提取区域特征 $R_i$，拼接成对特征 $R_{ij}$（$i \neq j$），用 MLP 预测 9 类空间关系（3 水平 × 3 垂直：left/middle/right × top/middle/bottom）：
    $\mathcal{L}_{\text{spatial}} = \mathbb{E}[-\boldsymbol{y_r}^{ij}\log(\hat{\boldsymbol{p_r}}^{ij})]$
      水平关系判定：$|\Delta x| < w/2$ → middle；$\Delta x > w/2$ → left；$\Delta x < -w/2$ → right
    - 设计动机：单独的 grounding loss 只关注单个区域的绝对定位，无法建模区域间的相对位置关系。而航拍场景中，"主楼在左边"这类相对位置描述是区分相似建筑的关键

6. **总体优化目标**

    - 功能：整合所有损失
    - 核心思路：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{itc}} + \mathcal{L}_{\text{itm}} + \lambda(\mathcal{L}_{\text{grounding}} + \mathcal{L}_{\text{spatial}})$，其中 $\lambda = 0.1$
    - 设计动机：blending spatial matching（混合空间匹配）权重较小，作为语义匹配的补充而非替代

### 训练策略
- Backbone：XVLM 预训练在 16M 图像
- 图像编码器：Swin；文本编码器：BERT
- 所有图像 resize 到 384×384，patch size 32
- 不使用随机旋转或水平翻转（会破坏空间信息）
- AdamW 优化器，学习率 $3e^{-5}$，weight decay 0.01

## 实验关键数据

### 主实验

| 方法 | 参数量 | Text→Image R@1 | R@5 | R@10 | Image→Text R@1 | R@5 | R@10 |
|------|--------|---------------|------|------|----------------|------|------|
| UNITER | 300M | 0.9 | 2.7 | 4.2 | 2.5 | 7.4 | 11.8 |
| ALBEF (14M) | 210M | 1.1 | 3.5 | 5.3 | 3.0 | 9.1 | 14.2 |
| XVLM (16M) | 216M | 4.5 | 9.9 | 13.4 | 5.0 | 14.4 | 21.4 |
| XVLM_finetuned | 216M | 13.2 | 23.7 | 29.6 | 25.0 | 52.3 | 65.1 |
| **Ours** | **217M** | **13.6** | **24.6** | **31.2** | **26.3** | **53.7** | **66.9** |

### 消融实验

| 方法 | Text→Image R@1 | R@10 | Image→Text R@1 | R@10 |
|------|---------------|------|----------------|------|
| Baseline (XVLM_finetuned) | 13.2 | 29.6 | 25.0 | 65.1 |
| + grounding loss only | 13.5 | 30.9 | 25.9 | 66.3 |
| + spatial loss only | 13.4 | 30.1 | 25.3 | 65.6 |
| **+ Both (Ours)** | **13.6** | **31.2** | **26.3** | **66.9** |

### 训练数据分析

| 训练集 | 图像数 | Text→Image R@1 | Image→Text R@1 |
|--------|--------|---------------|----------------|
| Drone only | 37,854 | 12.9 | 25.7 |
| Satellite + Ground | 12,364 | 10.1 | 18.7 |
| **All (Sat+Drone+Ground)** | **50,218** | **13.6** | **26.3** |

### 关键发现
- 预训练模型在航拍数据集上表现很差（XVLM 未微调仅 4.5% R@1），说明航拍域与通用数据存在巨大域差距
- 微调后性能大幅提升（XVLM: 4.5%→13.2% R@1），证明 GeoText-1652 数据集的价值
- Grounding loss 是主要贡献因素，spatial loss 作为补充可叠加提升
- 模型对小角度旋转（15°）鲁棒，对大角度旋转（90°/270°）性能下降可接受
- 在真实无人机视角图像上泛化良好

## 亮点与洞察
- **相对位置是航拍场景的关键区分信号**：当多栋相似建筑出现时，仅描述建筑外观不足以定位，"左侧的停车场"、"右上方的塔楼"等空间关系描述才是区分关键。9 类方向分类的设计虽然粗糙但足够有效
- **人机交互标注范式的实用性**：referee 模型 + Visual-LLM + 人工审核的流水线在降低标注成本的同时保持了质量（>90% 优秀），这种半自动标注框架可以推广到其他细粒度 VL 数据集构建

## 局限与展望
- Recall@1 仅 13.6%，绝对性能较低，实际导航应用中仍需大幅提升
- 9 类空间关系分类（3×3网格）过于粗糙，无法表达如"紧邻"、"远离"等精细空间关系
- 数据集仅覆盖大学校园场景，对城市、自然环境等多样场景的泛化能力未验证
- 仅考虑了 2D 空间关系，未利用建筑高度等 3D 信息

## 相关工作与启发
- **vs University-1652 [Zheng et al.]**：GeoText-1652 在其图像基础上增加了文本-bbox 标注，从图像检索任务扩展到自然语言引导任务
- **vs CLIP/ALBEF/XVLM**：通用 VL 模型在航拍域表现差，强调了领域专用数据的必要性
- **vs GeoDTR [Zhang et al.]**：GeoDTR 做视觉增强的跨视角定位，本文做自然语言引导定位，互为补充
- **vs VLN (Vision-Language Navigation)**：VLN 在室内/街景环境，本文聚焦于航拍无人机导航这一未充分探索的领域

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个自然语言引导无人机定位的benchmark，spatial relation matching 设计直觉合理
- 实验充分度: ⭐⭐⭐⭐ 多种基线对比、消融分析、训练集分析、旋转鲁棒性、真实场景泛化
- 写作质量: ⭐⭐⭐⭐ 数据集构建流程描述详尽，方法部分条理清晰
- 价值: ⭐⭐⭐⭐ 数据集对社区有持久价值，但绝对性能仍有较大提升空间

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] ReGround: Improving Textual and Spatial Grounding at No Cost](reground_improving_textual_and_spatial_grounding_at_no_cost.md)
- [\[CVPR 2025\] VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](../../CVPR2025/object_detection/vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)
- [\[ECCV 2024\] Tensorial Template Matching for Fast Cross-Correlation with Rotations and Its Application for Tomography](tensorial_template_matching_for_fast_cross-correlation_with_rotations_and_its_ap.md)
- [\[ECCV 2024\] LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction](lami-detr_open-vocabulary_detection_with_language_model_instruction.md)
- [\[ECCV 2024\] Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection](weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)

</div>

<!-- RELATED:END -->
