---
title: >-
  [论文解读] Around the World in 24 Hours: Probing LLM Knowledge of Time and Place
description: >-
  [ACL2025][可解释性][时空推理] 本文提出 GeoTemp 数据集（320k 提示，覆盖 289 个城市和 37 个时区），首次评估 LLM 联合时间和空间推理的能力，发现模型能独立处理时间计算和地理知识，但在需要结合两者时性能急剧下降。 时间和空间推理是理解世界的基本能力。在全球化工作环境中…
tags:
  - "ACL2025"
  - "可解释性"
  - "时空推理"
  - "地理知识"
  - "时间推理"
  - "LLM评估"
  - "基准测试"
---

# Around the World in 24 Hours: Probing LLM Knowledge of Time and Place

**会议**: ACL2025  
**arXiv**: [2506.03984](https://arxiv.org/abs/2506.03984)  
**代码**: [UhhDS/GeoTemp](https://github.com/UhhDS/GeoTemp)  
**领域**: 可解释性  
**关键词**: 时空推理, 地理知识, 时间推理, LLM评估, 基准测试  

## 一句话总结

本文提出 GeoTemp 数据集（320k 提示，覆盖 289 个城市和 37 个时区），首次评估 LLM 联合时间和空间推理的能力，发现模型能独立处理时间计算和地理知识，但在需要结合两者时性能急剧下降。

## 研究背景与动机

时间和空间推理是理解世界的基本能力。在全球化工作环境中，如跨境物流规划、商务出行安排等任务，都需要将时间和地理信息纳入推理过程。

已有工作分别研究了 LLM 的时间推理能力（如 TempLama、McTaco）和空间理解能力（如 WorldBench），但很少有工作评估它们**联合推理时间和空间**的能力。仅有的两项相关工作（TRAM 和 TOT）要么用多选题格式强化引导模型，要么使用完全合成的环境不含真实地点信息。

本文以**全球时区**为测试场景，首次系统性评估 LLM 在不同时间和地理知识组合下的推理能力。

## 方法详解

### GeoTemp 数据集构建

#### 步骤 1：收集时区和地点
- 从 Olson Time Zone Database (OTZD) 提取时区信息
- 过滤掉代表整个区域或已废弃的时区
- 通过 Opendatasoft API 获取城市级数据（人口、经纬度）
- 最终选取 289 个地点，覆盖 37 个 UTC 时区和 217 个 ISO 国家代码

#### 步骤 2：设计四类任务模板

| 任务 | 模板 | 所需知识 |
|------|------|---------|
| **Verification** | 在 $l_1$ 现在几点？ | 基本理解（回显时间） |
| **TimeTime** | 再过 x 小时几点？ | 仅时间计算 |
| **TimePlace** | 在 $l_2$ 现在几点？ | 时间+地理知识 |
| **TimeTimePlace** | 再过 x 小时在 $l_2$ 几点？ | 时间计算+地理知识 |

每个提示前缀为："Today is {Time&Date} in {$l_1$}"

#### 步骤 3：组合测试集
- 生成所有地点对的笛卡尔积 $l_1 \times l_2$
- 为每个组合应用所有任务模板
- 时间日期在 2023 年内随机选取
- 最终生成 **332,928** 个测试提示

### 数据集特点
- 覆盖所有大陆（含南极洲和多个岛屿）
- 6%+ 的地点人口 ≤500（如加拿大 Atikokan）
- 50% 城市人口 ≤50 万

### 评估方案

#### 模型
评测 8 个开源 chat 模型：Llama2-Chat (7B/13B/70B)、Llama3-Instruct (8B/70B)、Qwen2-Instruct (1.5B/7B/72B)

#### 指令类型
- **Neutral**：无额外指导
- **CoT**：附加 "Think step by step."
- **Short**：附加 "Keep your answer short..."

#### 评估方法
使用自定义 regex 算法从开放式回答中提取时间日期。验证集上准确率 ≥98%，远比 LLM-as-judge 高效。

## 实验

### 主实验结果

最佳模型 Llama3-70B 的性能：
- **Verification**: ~95%
- **TimeTime**: ~99%
- **TimePlace**: 56.1%
- **TimeTimePlace**: 25.4%（最高准确率！）

**核心发现**：大多数模型能很好地完成仅涉及时间知识的任务，但在需要结合时间和地理信息时性能显著下降。

### 模型规模效应

- Llama3 从 8B 到 70B：TimePlace 提升 29.3%，TimeTimePlace 提升 23.0%
- Qwen2 的规模提升带来的改善非常有限（TimePlace 最终仅 18.0%）
- Llama2 的提升更不明显
- 仅靠规模扩展可能无法解决时空联合推理问题

### 指令类型的影响

- **Short 指令**：简单任务提升、复杂任务下降
- **CoT 指令**：意外地在简单任务上降低性能（Llama2-70B 在 Verification 上显著下降）
- 定性分析发现模型在 CoT 下试图解决比实际更难的问题，反而陷入自身解释的矛盾中

### 地理偏差分析

- **未发现**明显的西方国家偏好
- Llama3-70B 对非洲国家表现最好，对北美、大洋洲表现反而更差
- 按大陆、人口、收入水平聚合均未发现清晰模式

### 地点名称困惑度分析

| 困惑度组合 | Llama3-70B 准确率 |
|-----------|-------------------|
| Very Low × Very Low | ~53.9% |
| Very High × Very High | ~29.9% |

低困惑度和高困惑度之间的性能差距达 **22.5%**。这表明模型性能偏向于训练数据中频繁出现的地点，而非偏向西方国家。

### 时区知识直接探测

| 模型 | 正确率 |
|------|--------|
| Llama3-70B | **90.0%** |
| Llama3-8B | 84.1% |
| Qwen2-72B | 86.3% |
| Qwen2-1.5B | 39.3% |

当**直接问**模型某个地点的时区时，大多数模型能正确回答 65%+（Llama3-70B 达 90%）。但在需要组合使用这些知识时却失败了——**知识存在但无法被有效检索和组合**。

### 错误分析（Llama3-70B，200 个错误样本）

| 错误类型 | 比例 |
|---------|------|
| DST/UTC 转换错误 | 25.3% |
| 时差计算错误（UTC 正确） | 22.3% |
| 至少一个地点的 UTC 知识错误 | **48.2%** |

### 知识注入实验

| 设置 | Llama3-70B 准确率 |
|------|-------------------|
| 原始 | ~33.4% |
| 添加时区信息 | **76.3%** |
| 仅用时区替换城市名 | ~65% |

注入地理知识后性能大幅提升。但"替换"方式不如"添加"方式好，说明模型确实会利用其地理知识来辅助推理。

## 亮点与洞察

1. **知识存在但无法组合**：这是最核心的发现——模型分别"知道"时间计算和时区知识，但在需要同时使用两者时失败。这指向 LLM 的一个基本局限：复杂推理中知识的检索与整合
2. **CoT 在简单任务上反而有害**：验证了 Sprague et al. 的发现（CoT 主要在数学和符号推理上有效），对不需要深度推理的任务可能引入干扰
3. **困惑度比地理区域更好地解释性能差异**：与预期的"西方偏差"不同，训练数据中地点的出现频率才是关键因素
4. **数据集设计精巧**：通过逐步增加复杂度的四级任务，精准定位模型的失败节点

## 局限性

1. regex 评估算法虽然准确率高（≥98%），但仍存在少量噪声
2. 鲁棒性检查仅在子集上进行
3. 无法获取模型预训练数据，使用困惑度作为频率代理可能不完全可靠
4. 未涉及纯地理任务（如预测位置），但这与已有工作重叠
5. 仅评估开源模型，未包含 GPT-4 等闭源模型的全面评测（仅在鲁棒性实验中测试了 gpt4o-mini）

## 相关工作

- **时间知识测试**：TimeBank、TempLama、McTaco 等聚焦时间推理的不同方面
- **地理知识测试**：GeoLLM、WorldBench 等评估 LLM 的地理知识和偏差
- **时空知识测试**：TRAM（多选题格式，分析有限）和 TOT（合成数据，无真实地点）

## 评分

⭐⭐⭐⭐ — 研究问题新颖且有实际意义，数据集设计精巧，分析深入系统。核心发现（知识存在但无法组合使用）具有广泛的启示意义。局限在于评测模型范围和任务类型相对有限，且解决方案的探索不够深入。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models](../../ACL2026/interpretability/mined_probing_and_updating_with_multimodal_time-sensitive_knowledge_for_large_mu.md)
- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](../../NeurIPS2025/interpretability/llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)
- [\[ACL 2025\] Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models](degenerate_knowledge_neurons.md)
- [\[NeurIPS 2025\] LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS](../../NeurIPS2025/interpretability/llm_probing_with_contrastive_eigenproblems_improving_understanding_and_applicabi.md)
- [\[ACL 2025\] Probing Subphonemes in Morphology Models](probing_subphonemes_in_morphology_models.md)

</div>

<!-- RELATED:END -->
