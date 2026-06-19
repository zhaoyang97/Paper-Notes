---
title: >-
  [论文解读] Omni-AD: A Large-scale and Versatile Benchmark for Industrial Anomaly Detection
description: >-
  [CVPR 2026][目标检测][工业异常检测] Omni-AD 是一个从真实产线采集、覆盖 16 个行业 150 个品类、约 3.5 万张像素级标注图像的工业异常检测（IAD）基准；它既支持传统无监督 IAD 评测，又首次为多模态大模型（MLLM）设计了「判别—分类—定位」三个递进难度的子任务，实验证明现有方法与 MLLM 在这个数据集上都远未饱和。
tags:
  - "CVPR 2026"
  - "目标检测"
  - "工业异常检测"
  - "大规模基准"
  - "MLLM 评测"
  - "视觉问答"
  - "视觉定位"
---

# Omni-AD: A Large-scale and Versatile Benchmark for Industrial Anomaly Detection

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_Omni-AD_A_Large-scale_and_Versatile_Benchmark_for_Industrial_Anomaly_Detection_CVPR_2026_paper.html)  
**代码**: 项目页 https://omni-ad.github.io  
**领域**: 目标检测 / 工业异常检测 / 基准数据集  
**关键词**: 工业异常检测, 大规模基准, MLLM 评测, 视觉问答, 视觉定位

## 一句话总结
Omni-AD 是一个从真实产线采集、覆盖 16 个行业 150 个品类、约 3.5 万张像素级标注图像的工业异常检测（IAD）基准；它既支持传统无监督 IAD 评测，又首次为多模态大模型（MLLM）设计了「判别—分类—定位」三个递进难度的子任务，实验证明现有方法与 MLLM 在这个数据集上都远未饱和。

## 研究背景与动机

**领域现状**：工业异常检测的主流范式是无监督学习——只用正常（无缺陷）图像训练，让模型在测试时检出未知缺陷，从而省去昂贵的缺陷数据采集与标注。这一范式的发展深度依赖基准数据集，MVTec AD（5,354 张、15 品类）催生了大量算法，VisA、Real-IAD 等随后把规模逐步推大。

**现有痛点**：作者指出两个卡脖子的问题。其一是**性能饱和**：当下顶尖算法在 MVTec AD 上的 image-level / pixel-level AUROC 都已超过 99%，不同算法的优劣几乎无法区分，基准失去了甄别力。其二是**缺少面向 MLLM 的评测**：MLLM 在通用领域展现出强大能力，但它们到底能不能做工业质检几乎无人系统评估；已有的 MMAD 用「多选 VQA」形式做了初步尝试，但多选题与真实质检需求并不对齐。

**核心矛盾**：现有数据集大多在实验室里人工构造，品类与缺陷多样性有限，从实验室迁移到真实产线时差距巨大；同时既有评测协议（无监督 + 多选 VQA）无法同时满足「甄别难度足够」和「贴近 MLLM 实际用法」两个要求。

**本文目标**：构建一个（i）规模与多样性显著更大、（ii）同时支持无监督与 MLLM 两套协议、（iii）对当前 SOTA 仍有挑战性的统一基准。

**切入角度**：从真实产线直接采集数据（而非人工合成缺陷），并把 MLLM 的能力拆成贴近工业现场的三个问题——「有没有缺陷 / 是什么缺陷 / 缺陷在哪」。

**核心 idea**：用一个大规模真实产线数据集 + 三个递进的 MLLM 子任务（两个 VQA + 一个视觉定位），把 IAD 评测从「饱和、只针对无监督」推进到「有甄别力、原生支持 MLLM」。

## 方法详解

### 整体框架
Omni-AD 本质是「数据集 + 双协议基准」，不含可训练的新模型，因此核心工作是四块：**真实产线数据采集与对齐** → **循环两阶段像素级标注** → **无监督 IAD 协议** + **MLLM 三子任务协议（含层级化语义 QA 生成）**。数据集最终包含 34,886 张图像、150 个品类、16 个行业，正常/异常约 1:1（16,989 正常 / 17,897 异常），每个品类含 1–7 种缺陷类型。基准侧把评测拆成两条互补的路径：传统无监督路径沿用「只用正常图训练、正常+异常一起测」的设定；MLLM 路径则把缺陷理解组织成「判别（Discrimination）→ 分类（Classification）→ 定位（Localization）」三个难度递增的子任务。这是一篇纯基准/数据集论文，下面按四个关键设计展开。

### 关键设计

**1. 真实产线数据采集与前景对齐：让基准更贴近落地、对比更公平**

针对「实验室合成数据多样性不足、迁移到产线就崩」的痛点，作者直接从多条真实生产线采集约 3.5 万张高分辨率图，覆盖金属、塑料、陶瓷、纸、织物、木材等材料的 150 个品类、16 个行业。与人工合成数据不同，真实数据**事先并不知道缺陷类型与位置**，因此作者为每个品类按厂商对产品用途和质量要求的描述，定义一套专属缺陷类别集合 $A_{cat}$，再据此把图像人工划分为正常/异常。采集后还做**前景对齐**：对刚性产品用模板匹配把前景对齐，对可形变/纹理类产品用裁剪把 RoI 推到图像中心。这一步削弱了背景干扰与目标错位对模型（尤其是基于重建的方法）的负面影响，让不同算法的横向比较更公平。

**2. 循环两阶段像素级标注：在大规模下保证标注质量一致性**

跨众多工业场景做细粒度像素标注极易出错且难以一致。作者设计了一个**循环的两阶段**标注流水线。第一阶段是**人工主导标注**：专业标注员勾勒缺陷类型与多边形掩码；先用 500 张精标图建立「控制集」，所有标注员必须通过资格测试（与控制集的差异率低于严格阈值）才能上岗；团队分为标注员与检查员，每个品类的图均分成三批顺序标注，检查员逐批复核，任何含错的批次被打回原标注员修正——既纠错又强制同一品类内规则一致。第二阶段是**模型辅助标注**：把每品类人工标注数据均分两份（正常/异常等量），训练一个有监督语义分割模型（如 U-Net）来发现潜在标注错误，标注员核查模型预测与人工标注不一致的区域并修正；这个「模型审、人工改」的过程反复迭代，直到差异率低于预设阈值。

**3. MLLM 三子任务协议：把缺陷理解拆成贴近质检的递进难度**

已有 MMAD 用多选 VQA，与真实需求不对齐。Omni-AD 把 MLLM 评测拆成难度递增、且直接对应现场三问的子任务：**缺陷判别**——给定图像或指定框 + 富语义问题，模型二选一回答 Yes/No；**缺陷分类**——在若干候选缺陷类型中选出正确类型（或 Not any），形式上是 VQA；**缺陷定位**——把任务设为视觉定位（visual grounding），模型须输出 JSON 格式 `{'box':[xmin,ymin,xmax,ymax],'label':'defect type'}`。每个子任务都在 **0-shot 与 1-shot** 两种设定下评测，1-shot 允许模型参考正常集中最相似的样本作为视觉指引。判别/分类用 VQA 准确率（分 image-level 与 box-level），定位因为 MLLM 缺乏可靠置信度而改用 IoU=0.5 下的 Recall / Precision / F1。

**4. 层级化语义 QA 生成流水线：把结构化元数据转成语义丰富的问答对**

数据集每个样本虽有结构化元数据（产品、缺陷信息），但格式化文本缺乏 MLLM 评测所需的语义丰富度。作者在 MMAD 流水线基础上，用 GPT-4o 做**层级化数据增强**，分三步：（a）**先验增强（Prior Enrichment）**——构造视觉提示（原图叠加由掩码得到的边缘高亮缺陷轮廓 + 同产品的一张正常图做对比），结合实例结构化元数据，引导 GPT-4o 补全使用场景、功能、缺陷形态、成因机制、影响等高层语义，得到「增强后的实例先验」；（b）**知识整合（Knowledge Consolidation）**——把同一产品所有实例的增强先验聚合，用 GPT-4o 过滤掉位置/背景噪声等实例特有变化，蒸馏成一致的「产品级先验」，再经人工筛选保证可靠；（c）**QA 生成**——以视觉提示 + 实例先验 + 产品先验作为上下文，按任务定义与格式要求生成语义丰富的问答对。最终产出 12K 个 QA 数据和 3K 个 grounding 数据。

## 实验关键数据

> 评测指标说明：**I-AUROC / I-AUPR** 为 image-level 的 ROC/PR 曲线下面积（异常分类）；**P-AUROC / P-AUPRO** 为 pixel-level 的 ROC 曲线下面积与 Per-Region-Overlap 曲线下面积（异常分割，PRO 更能兼顾不同缺陷尺寸与正负像素不平衡）；MLLM 侧 **Img-Acc / Box-Acc** 为整图/指定框输入下的 VQA 准确率，定位用 IoU=0.5 下的 Recall/Precision/F1。指标越低代表数据集越难。

### 数据集规模对比

| 数据集 | 品类数 | 正常 | 异常 | 总图像 | 像素标注 |
|--------|--------|------|------|--------|----------|
| MVTec AD | 15 | 4,096 | 1,258 | 5,354 | ✓ |
| VisA | 12 | 9,621 | 1,200 | 10,821 | ✓ |
| GoodsAD | 6 | 4,464 | 1,660 | 6,124 | ✓ |
| Real-IAD（单视角） | 30 | 14,568 | 15,642 | 30,210 | ✓ |
| **Omni-AD** | **150** | **16,989** | **17,897** | **34,886** | ✓ |

品类数比 MVTec AD 多一个数量级（150 vs. 15）、比 VisA 多约 12.5 倍；总图像约为 MVTec AD 的 6 倍、VisA 的 3 倍。

### 无监督 IAD：在 Omni-AD 上普遍掉点且分化更明显

| 方法 | 范式 | MVTec AD (I-AUROC) | Real-IAD (I-AUROC) | Omni-AD (I-AUROC) | Omni-AD (P-AUPRO) |
|------|------|--------------------|--------------------|--------------------|--------------------|
| PatchCore | MemoryBank | 99.1 | 90.4 | **87.8** | 81.1 |
| Dinomaly | 重建 | 99.6 | 89.3 | 84.5 | 71.4 |
| SimpleNet | 数据增强 | 99.6 | 91.7 | 82.1 | 76.7 |
| GLASS | 数据增强 | 99.9 | 92.3 | 83.1 | 74.8 |

关键观察：在 MVTec AD 上所有方法 I-AUROC 挤在 94.9%–99.9% 几乎无法区分，到 Omni-AD 上区间拉开到 82.1%–87.8%、P-AUPRO 更是低至 71.4%–81.6%；号称「多类媲美单类」的 Dinomaly 在 Omni-AD 上 P-AUPRO 掉到 71.4，分化最显著，说明该基准既更难也更有甄别力。

### MLLM 三子任务：随难度递增逐级下滑，定位是最大短板

| 模型 | 设定 | 判别 Img-Acc | 分类 Img-Acc | 定位 F1 |
|------|------|-------------|-------------|---------|
| 随机基线 | - | 50.00 | 20.00 | N/A |
| 人类专家 | - | 95.67 | 91.67 | 79.32 |
| LLaVA-NeXT 34B | 0-shot | 59.43 | 38.84 | N/A（未输出合法 JSON）|
| InternVL-3.5 38B | 0-shot | 57.58 | 40.70 | 10.07 |
| Qwen3-VL-Instruct 30B-A3B | 1-shot | 64.65 | 65.94 | 14.84 |
| Qwen3-VL-Thinking 30B-A3B | 1-shot | **69.38** | 64.29 | 25.64 |

### 关键发现
- **任务难度递进有效**：所有模型在「判别 → 分类 → 定位」上性能依次下滑，最难的定位子任务上最佳模型（Qwen3-VL-30B-A3B-Thinking）F1 仅 26.20（0-shot），远低于人类的 79.32，说明精确定位仍是 IAD 的硬骨头。
- **Box-level 普遍优于 Image-level**：把注意力聚焦到指定区域后，判别与分类准确率都更高，说明给定 RoI 能帮 MLLM 抽到更有意义的特征。
- **Thinking 模式利于定位**：Qwen3-VL-30B-A3B-Thinking 在定位上 F1 26.20 明显高于 Instruct 版的 19.12，提示带「思考」的 MLLM 有更强的视觉定位潜力。
- **1-shot 对判别/分类多为正向，对定位偶有反效果**：参考正常模板样本能帮助多数模型做配对比较提升判别/分类，但在更难的定位子任务上多数方法反而略降。
- **与人类差距巨大**：人类专家在全部 7 个指标上大幅领先所有 MLLM，既说明基准之难，也说明 MLLM 异常检测仍有巨大提升空间。

## 亮点与洞察
- **「真实产线 + 双协议」一次到位**：既用规模和多样性把无监督 IAD 的饱和度打回去，又原生支持 MLLM 评测，避免了「换个数据集只是变大、评测方式还是老一套」的窠臼。
- **三子任务的设计直接映射现场三问**（有没有/是什么/在哪），并刻意把定位做成 grounding 输出 JSON——这个设定一下子暴露了 MLLM「能判断却定不准」的能力断层，比多选 VQA 更有诊断价值。
- **层级化 QA 生成把结构化标注「语义化」**：用「实例先验 → 产品先验」的两级整合再生成问答，思路可迁移到任何「有结构化元数据但缺语义描述」的垂直领域基准构建（如医学、遥感质检）。
- **循环两阶段标注 + 控制集资格测试**是大规模像素标注保质的实用工程范式，可直接复用。

## 局限与展望
- **本质是基准而非方法**：论文不提出新检测算法，价值在于「提供更难、更全的尺子」，而非推动单个模型性能。
- **MLLM 评测依赖 GPT-4o 生成 QA**：QA 质量受 GPT-4o 能力与提示设计影响，虽有人工筛选，但「用一个强 MLLM 生成题目去考别的 MLLM」存在潜在偏置（⚠️ 论文未量化该偏置）。
- **定位评测仅用 IoU=0.5 单阈值的 R/P/F1**，对「轻微偏移但语义正确」的预测较严苛；不同 MLLM 输出格式合规率差异（如 LLaVA-NeXT 直接 N/A）也会放大分数差距。
- **可改进方向**：补充 few-shot/微调后的 MLLM 上限、加入多视角或多光照设定、对 QA 生成偏置做对照实验。

## 相关工作与启发
- **vs MVTec AD / VisA / Real-IAD**：它们是无监督 IAD 的经典尺子，但 MVTec AD 已饱和、VisA/Real-IAD 仍源自受控或合成场景；Omni-AD 在品类（150）、行业（16）、规模（35K）上全面更大，且数据来自真实产线，更具挑战与现实性。
- **vs MMAD**：MMAD 首个系统评测 MLLM 的 IAD 能力，但用多选 VQA、定义 7 个子任务、题目来自 4 个公开数据集；Omni-AD 改用「判别/分类/定位」三个贴近现场的递进子任务，并把定位做成 grounding，评测更对齐真实需求。
- **vs MANTA / PIAD**：它们分别面向微小目标多视角文本数据、位姿与光照无关检测；Omni-AD 的差异在于「真实产线来源 + 原生 MLLM 协议」，互为补充。

## 评分
- 新颖性: ⭐⭐⭐⭐ 数据集本身是「更大更真」的延续，但「原生支持 MLLM 三子任务 + 层级化 QA 生成」是实质性的新角度
- 实验充分度: ⭐⭐⭐⭐ 覆盖 6 种无监督方法、4 个 MLLM 家族、双协议 + 人类对照，但缺少 MLLM 微调/few-shot 上限与 QA 偏置对照
- 写作质量: ⭐⭐⭐⭐ 动机清晰、协议与指标定义完整、图表自洽
- 价值: ⭐⭐⭐⭐⭐ 提供了一把更难、能甄别、且原生面向 MLLM 的工业异常检测尺子，对推动该方向落地很有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](mmrad_multimodal_anomaly_detection.md)
- [\[CVPR 2026\] FastRef: Fast Prototype Refinement for Few-shot Industrial Anomaly Detection](fastref_fast_prototype_refinement_for_few-shot_industrial_anomaly_detection.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[CVPR 2026\] SRA-Det: Learning Omni-Grained Open-Vocabulary Detection Beyond Category Names](sra-det_learning_omni-grained_open-vocabulary_detection_beyond_category_names.md)
- [\[ICLR 2026\] ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](../../ICLR2026/object_detection/forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)

</div>

<!-- RELATED:END -->
