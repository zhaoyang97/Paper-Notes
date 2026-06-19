---
title: >-
  [论文解读] Trust but Verify: Programmatic VLM Evaluation in the Wild
description: >-
  [ICCV 2025][多模态VLM][VLM评测] 提出 PROVE（Programmatic VLM Evaluation）评测范式，通过从超详细图像描述构建高保真场景图，并利用 LLM 生成可编程验证的开放式视觉问答对，在统一的场景图框架内同时评估 VLM 回答的**有用性**（helpfulness）和**真实性**（truthfulness），揭示当前模型在两者之间难以取得良好平衡。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "VLM评测"
  - "幻觉检测"
  - "场景图"
  - "程序化验证"
  - "开放式问答"
  - "有用性-真实性权衡"
---

# Trust but Verify: Programmatic VLM Evaluation in the Wild

**会议**: ICCV 2025  
**arXiv**: [2410.13121](https://arxiv.org/abs/2410.13121)  
**代码**: [项目页面](https://prove-explorer.netlify.app/)  
**领域**: 多模态VLM  
**关键词**: VLM评测, 幻觉检测, 场景图, 程序化验证, 开放式问答, 有用性-真实性权衡

## 一句话总结

提出 PROVE（Programmatic VLM Evaluation）评测范式，通过从超详细图像描述构建高保真场景图，并利用 LLM 生成可编程验证的开放式视觉问答对，在统一的场景图框架内同时评估 VLM 回答的**有用性**（helpfulness）和**真实性**（truthfulness），揭示当前模型在两者之间难以取得良好平衡。

## 研究背景与动机

### 核心问题

VLM 经常生成看似合理但实际错误的回答（幻觉），然而现有评测方法存在明显不足：

**判别式基准**（如 POPE）：仅测试二元存在性问题（"图中是否有人？"），不模拟真实使用场景

**生成式基准**（如 CHAIR、MMHal）：评估开放式回答，但依赖外部 LLM 打分，且评判时提供的上下文往往不足以验证所有声明

**LLM-as-judge 的局限**：缺少清晰评分标准、对 prompt 变化敏感，导致评分不一致和任意性

### 具体例子

当 VLM 回答"图中有四只拉布拉多贵宾犬"时，需要同时验证 `<数量==4>` 和 `<品种==拉布拉多>` 两个声明，但若 LLM 评判员仅获得简短描述（"四只小狗在浅蓝色地毯上"），则无法完成完整验证。

### 研究动机

需要一种既能测试开放式问答（贴近真实使用），又能可靠、可解释地评估回答质量的方法。关键在于：场景描述的高召回率 + 程序化验证 + 统一评估框架。

## 方法详解

### 整体框架

PROVE 包含两大部分：**数据集构建**和**程序化评估**。

#### 1. 数据集构建流程

```
超详细图像描述(DOCCI) → 场景图构建 → LLM生成QA对+验证程序 → 过滤 → 10.5K高质量QA对
```

**Step 1: 构建场景图表示**
- 使用 DOCCI 测试集的 5K 图像-描述对（平均描述长度 136 词，远超竞品数据集）
- 从描述中提取实体-属性-关系三元组，构建有向图 $g(\mathcal{C})$
- 场景图实现为 Python 类，提供查询实体、属性、关系以及提取子图的 API

**Step 2: 生成可验证的 QA 对**
- 用 GPT-4o 对每张图像生成 10-15 个多样化、有挑战性的开放式 QA 对
- 同时生成伴随的 Python 验证程序，可在场景图对象上执行以验证 QA 对的正确性

**Step 3: 双重过滤**
- **程序化过滤**：执行验证程序，丢弃程序失败（18.3%）或返回错误答案（9.8%）的 QA 对
- **文本过滤**：排除低质量 QA 对——琐碎/不明确/不完整的（LLM 判断）、不被图像蕴含的（视觉蕴含模型）、包含禁忌词的、语义重复的（SemDeDup）

最终保留约 50% 的 QA 对，共 **10.5K** 高质量样本。

#### 2. 程序化评估方法

给定模型回答 $\hat{\mathcal{A}} = m_\theta(\mathcal{Q}, \mathcal{I})$，分两个维度评估：

**有用性评分（hscore）**——回答对 GT 答案的场景图召回率：

$$\text{hscore}(\hat{\mathcal{A}}) = \frac{\sum_{t \in g(\mathcal{A}) - g(\mathcal{Q})} \max_{t' \in g(\hat{\mathcal{A}})} \text{sim}(t, t')}{|g(\mathcal{A}) - g(\mathcal{Q})|}$$

- 从 GT 答案和模型回答分别提取场景图元组
- 排除问题中已出现的前提元组
- 计算 GT 元组到回答元组的最近余弦相似度的平均值

**真实性评分（tscore）**——回答元组对完整场景的精确率：

$$\text{tscore}(\hat{\mathcal{A}}) = \frac{\sum_{t' \in g(\hat{\mathcal{A}})} \max\left(\max_{t \in g(\mathcal{C})} \text{sim}(t', t),\ p(\mathcal{I} \models t')\right)}{|g(\hat{\mathcal{A}})|}$$

- 不仅匹配完整描述的场景图，还利用视觉蕴含模型检查图像本身
- 减少因描述不完整导致的假阳性幻觉检测

### 关键设计特点

1. **双重验证来源**：tscore 同时利用文本场景图和视觉蕴含，因为再详细的描述也无法覆盖图像的所有方面
2. **hscore 与 tscore 解耦**：两者不必正相关——回答可以有用但不完全真实（含幻觉），也可以真实但不够有用
3. **可解释性**：基于场景图匹配的具体评分标准，而非不透明的 LLM 打分

## 实验关键数据

### 主实验：VLM 在 PROVE 上的 helpfulness-truthfulness 权衡

| 模型 | 参数量 | hscore ↑ | tscore ↑ | 平均 ↑ |
|------|--------|----------|----------|--------|
| Qwen2-VL | 2B | 69.36 | 80.64 | 75.00 |
| InternVL2 | 2B | 73.96 | 79.51 | 76.74 |
| Phi-3.5-Vision | 4B | 73.35 | 82.27 | 77.81 |
| LLaVA-1.5 | 7B | 72.67 | **82.58** | 77.62 |
| LLaVA-Next | 7B | 74.28 | 80.03 | 77.15 |
| InternVL2 | 8B | 74.55 | 80.56 | 77.56 |
| Pixtral | 12B | 73.34 | 82.43 | 77.88 |
| LLaVA-1.5 | 13B | 72.46 | 82.40 | 77.43 |
| InternVL2 | 26B | 74.63 | 79.23 | 76.93 |
| Claude-3.5-Sonnet† | - | 71.06 | 77.31 | 74.19 |
| GPT-4o-mini† | - | 73.18 | 79.24 | 76.21 |
| Gemini-1.5-Flash† | - | 72.73 | 81.74 | 77.23 |
| **GPT-4o†** | - | **76.53** | 80.92 | **78.72** |
| Oracle* | - | 82.84 | 85.59 | 84.22 |

### 关键消融/分析结果

| 分析维度 | 发现 |
|---------|------|
| hscore vs tscore 相关性 | 模型间平均线性相关仅 **0.03**，几乎无相关 |
| 模型大小与真实性 | InternVL2(2B→8B→26B)：hscore 提升但 tscore 未必提升 |
| LLaVA 系列对比 | LLaVA-1.5 系列获得整体最佳 tscore，但 hscore 较低 |
| 人工评估-QA质量 | 95.9% 问题被判为相关，98.2% 答案被判为正确 |
| 人工评估-指标相关性 | hscore 与人类判断相关 0.81，tscore 相关 0.45 |
| Oracle vs 最佳模型 | Oracle 平均 84.22 vs GPT-4o 78.72，仍有显著提升空间 |

### 关键发现

1. **极少模型在两者间取得良好平衡**：仅 GPT-4o、Phi-3.5-Vision、Pixtral 表现较均衡
2. **高排名模型不一定高真实性**：Claude-3.5-Sonnet 和 InternVL2-26B 在聚合排行榜上排名高，但 tscore 落后于更简单的 LLaVA-1.5
3. **模型以不同方式失败**：GPT-4o 的错误更"轻微"（如 6 个字母读对 3 个），LLaVA 错误更严重（仅读对 1 个）；GPT-4o 生成更描述性的答案提升了 hscore
4. **常见幻觉对象**：tree, building, wall, sign 等常见物体

## 亮点与洞察

1. **范式创新**：首次将程序化验证引入开放式 VLM 评估，通过"生成QA→编程验证→场景图评估"的闭环实现高可靠性
2. **揭示重要权衡**：helpfulness 与 truthfulness 之间的弱相关（0.03）说明近年模型在"更好"方向的改进主要是有用性提升而非真实性提升
3. **评估的可扩展性**：整个基准构建流程完全自动化，可以轻松扩展到更大的图像-描述源
4. **量化反直觉发现**：更大更新的模型不一定更真实——挑战了"scaling = better"的朴素认知

## 局限性

1. **召回率代价**：为确保高精度而过滤掉约 50% 的 QA 对，某些难以验证的问题类型可能被排除
2. **描述覆盖度**：即使高召回率的描述也无法捕获图像的所有方面，可能遗漏某些幻觉
3. **依赖外部模型**：文本嵌入（Sentence-BERT）、场景图提取、视觉蕴含（OFA）都可能引入各自的误差
4. **未测试缓解方法**：未评估微调、偏好优化、无训练解码等幻觉缓解策略在 PROVE 上的效果

## 相关工作与启发

- **与 CHAIR 的区别**：CHAIR 仅评估描述中对象的精确率/召回率，限于图像描述模板；PROVE 支持任意开放式问题
- **与 MMHal-Bench 的区别**：MMHal 依赖一系列现成模型引入噪声，且 GPT-4 评分可能因缺少上下文而惩罚正确回答
- **与 GAVIE 的区别**：GAVIE 依赖稠密描述和边界框，问题多关注局部区域和空间关系，回答质量不自然
- **对未来工作的启发**：结合 agentic VLM（能规划、推理、自我反思）可能是同时提升 hscore 和 tscore 的方向

## 评分 ⭐⭐⭐⭐

**创新性**: ⭐⭐⭐⭐⭐ — 程序化验证 + 场景图评估的范式新颖，极具影响力  
**实用性**: ⭐⭐⭐⭐ — 提供了可扩展的自动化评测流程  
**实验深度**: ⭐⭐⭐⭐ — 覆盖 14 个模型，含人工评估验证  
**写作质量**: ⭐⭐⭐⭐ — 动机清晰，示例充分，表述严谨

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Reading Recognition in the Wild](../../NeurIPS2025/multimodal_vlm/reading_recognition_in_the_wild.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)
- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[ICCV 2025\] GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks](geobench-vlm_benchmarking_vision-language_models_for_geospatial_tasks.md)
- [\[ICCV 2025\] Dynamic Group Detection using VLM-augmented Temporal Groupness Graph](dynamic_group_detection_using_vlm-augmented_temporal_groupness_graph.md)

</div>

<!-- RELATED:END -->
